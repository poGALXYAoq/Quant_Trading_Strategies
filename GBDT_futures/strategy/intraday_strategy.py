# 策略：日内 T+1 开盘进、当日收盘平

import os
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


DATE_COL = "date"


@dataclass
class BacktestConfig:
    predictions_dir: str
    price_csv_path: str
    output_dir: str
    start_date: str = "2024-06-01"
    end_date: str = "2025-08-31"
    initial_capital: float = 500_000.0
    lot_multiplier: float = 5.0
    lots: int = 1
    proba_threshold: float = 0.5  # 概率阈值，超过才开仓
    price_date_col: Optional[str] = None
    price_open_col: Optional[str] = None
    price_close_col: Optional[str] = None
    splits_to_use: Tuple[str, ...] = ("train", "valid", "test")


# ===== 可在此处直接配置默认参数（便于 IDE 一键运行） =====
_BASE_DIR = os.path.dirname(os.path.dirname(__file__))
USER_CONFIG = BacktestConfig(
    predictions_dir=os.path.join(_BASE_DIR, "results", "predictions"),
    price_csv_path="GBDT_futures/data/CU/CU888.csv",
    output_dir=os.path.join(_BASE_DIR, "results", "strategy", "intraday"),
    proba_threshold=0.49,  # 默认0.5表示只要预测概率偏向一边就交易
)


def ensure_dirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _resolve_path_human_style(p: str) -> str:
    """支持以下几种写法：
    1) 绝对路径：原样返回
    2) 以 "GBDT_futures/" 开头：替换为 _BASE_DIR 下的相对路径
    3) 相对路径：默认相对到 _BASE_DIR（项目的 GBDT_futures 目录）
    """
    if os.path.isabs(p):
        return p
    p_norm = p.replace("\\", "/")
    if p_norm.startswith("GBDT_futures/"):
        rel = p_norm.split("/", 1)[1]
        return os.path.normpath(os.path.join(_BASE_DIR, rel))
    return os.path.normpath(os.path.join(_BASE_DIR, p))


def _read_predictions(predictions_dir: str, splits: Tuple[str, ...]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for name in ("train", "valid", "test"):
        if name not in splits:
            continue
        f = os.path.join(predictions_dir, f"{name}.csv")
        if os.path.exists(f):
            df = pd.read_csv(f, encoding="utf-8-sig")
            required_cols = [DATE_COL, "y_pred_p0", "y_pred_p1", "y_pred_p2"]
            if not all(c in df.columns for c in required_cols):
                raise ValueError(f"预测文件缺少必要的概率列: {f}")
            df[DATE_COL] = pd.to_datetime(df[DATE_COL])
            df["split"] = name
            frames.append(df[required_cols + ["split"]].copy())
    if not frames:
        raise FileNotFoundError(
            f"未在 {predictions_dir} 找到 train/valid/test 预测CSV。请先运行模型脚本生成预测。"
        )
    out = pd.concat(frames, ignore_index=True).sort_values(DATE_COL).reset_index(drop=True)
    return out


def _auto_find_price_cols(df: pd.DataFrame, cfg: BacktestConfig) -> Tuple[str, str, str]:
    if cfg.price_date_col and cfg.price_open_col and cfg.price_close_col:
        return cfg.price_date_col, cfg.price_open_col, cfg.price_close_col

    cols = {c: str(c).strip().lower() for c in df.columns}
    inv = {v: k for k, v in cols.items()}

    date_candidates = [
        "date", "交易日期", "日期", "交易日", "day"
    ]
    open_candidates = [
        "open", "开盘", "开盘价"
    ]
    close_candidates = [
        "close", "收盘", "收盘价"
    ]

    def find_by_candidates(cands: List[str], fallback_contains: List[str]) -> Optional[str]:
        for k, v in cols.items():
            if v in inv and v in cands:
                return inv[v]
        # contains match (for中文带后缀列名)
        for k in df.columns:
            name = str(k).lower()
            if all(x in name for x in fallback_contains):
                return k
        return None

    date_col = None
    for cand in date_candidates:
        if cand in inv:
            date_col = inv[cand]
            break
    if date_col is None:
        # 常见中文日期列包含“日期”
        for k in df.columns:
            if "日期" in str(k):
                date_col = k
                break

    open_col = find_by_candidates(open_candidates, ["开", "盘"])  # 例如 含有“开盘”
    close_col = find_by_candidates(close_candidates, ["收", "盘"])  # 例如 含有“收盘”

    if not date_col or not open_col or not close_col:
        raise ValueError(
            "无法自动识别开收盘列名，请在 USER_CONFIG 中显式设置 price_date_col/price_open_col/price_close_col。"
        )
    return date_col, open_col, close_col


def _read_prices(price_csv_path: str, cfg: BacktestConfig) -> pd.DataFrame:
    price_csv_path = _resolve_path_human_style(price_csv_path)
    if not os.path.exists(price_csv_path):
        raise FileNotFoundError(
            f"未找到开收盘CSV: {price_csv_path}。请将文件放入该路径或在USER_CONFIG中修改。"
        )
    df = pd.read_csv(price_csv_path, encoding="utf-8-sig")
    date_col, open_col, close_col = _auto_find_price_cols(df, cfg)
    out = df[[date_col, open_col, close_col]].copy()
    out.columns = [DATE_COL, "open", "close"]
    out[DATE_COL] = pd.to_datetime(out[DATE_COL])
    out = out.sort_values(DATE_COL).reset_index(drop=True)
    return out


def _apply_signal(df_pred: pd.DataFrame, threshold: float) -> np.ndarray:
    """根据预测概率和阈值生成交易信号。"""
    p_down = df_pred["y_pred_p0"].values
    p_up = df_pred["y_pred_p2"].values
    
    sig = np.zeros(len(df_pred), dtype=int)
    sig[p_up > threshold] = 1
    sig[p_down > threshold] = -1
    return sig


def _max_drawdown(equity: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
    roll_max = equity.cummax()
    drawdown = equity - roll_max
    # idxmin 返回的是标签（当索引为日期时是 Timestamp），直接使用即可
    end = drawdown.idxmin()
    start = equity.loc[:end].idxmax()
    return float(drawdown.min()), pd.Timestamp(start), pd.Timestamp(end)


def _safe_div(a: float, b: float) -> Optional[float]:
    try:
        if b == 0:
            return None
        return float(a) / float(b)
    except Exception:
        return None


def _compute_consecutive_stats(pnls: np.ndarray) -> Tuple[int, int]:
    max_wins = 0
    max_losses = 0
    cur = 0
    for v in pnls:
        if v > 0:
            cur = cur + 1 if cur >= 0 else 1
        elif v < 0:
            cur = cur - 1 if cur <= 0 else -1
        else:
            cur = 0
        max_wins = max(max_wins, cur if cur > 0 else 0)
        max_losses = min(max_losses, cur if cur < 0 else 0)
    return int(max_wins), int(abs(max_losses))


def backtest_intraday(cfg: BacktestConfig) -> Tuple[pd.DataFrame, Dict[str, float]]:
    preds = _read_predictions(cfg.predictions_dir, cfg.splits_to_use)
    preds["signal_int"] = _apply_signal(preds, cfg.proba_threshold)

    prices = _read_prices(cfg.price_csv_path, cfg)
    # 构造下一交易日映射
    prices = prices.sort_values(DATE_COL).reset_index(drop=True)
    prices["next_date"] = prices[DATE_COL].shift(-1)

    # 对齐信号到执行日（下一交易日）
    merged = preds.merge(prices[[DATE_COL, "next_date"]], on=DATE_COL, how="left")
    merged.rename(columns={"next_date": "exec_date"}, inplace=True)
    merged = merged.dropna(subset=["exec_date"]).copy()
    merged["exec_date"] = pd.to_datetime(merged["exec_date"])  # type: ignore

    # 过滤回测窗口基于执行日
    start_dt = pd.to_datetime(cfg.start_date)
    end_dt = pd.to_datetime(cfg.end_date)
    merged = merged[(merged["exec_date"] >= start_dt) & (merged["exec_date"] <= end_dt)].copy()
    if len(merged) == 0:
        raise ValueError("在指定回测区间内没有可执行的信号。")

    # 拿到执行日的开收盘
    px = prices[[DATE_COL, "open", "close"]].copy()
    px.columns = ["exec_date", "exec_open", "exec_close"]
    merged = merged.merge(px, on="exec_date", how="left").dropna(subset=["exec_open", "exec_close"]).copy()

    # 仅交易非None信号
    trade_df = merged[merged["signal_int"] != 0].copy().reset_index(drop=True)
    if len(trade_df) == 0:
        raise ValueError("所有信号均为None，未产生任何交易。")

    # 计算逐笔结果
    direction = trade_df["signal_int"].values.astype(int)
    open_px = trade_df["exec_open"].values.astype(float)
    close_px = trade_df["exec_close"].values.astype(float)
    price_diff = (close_px - open_px) * direction
    pnl = price_diff * cfg.lot_multiplier * cfg.lots

    results = trade_df[["exec_date", "signal_int", "exec_open", "exec_close"]].copy()
    results.rename(columns={"signal_int": "signal"}, inplace=True)
    results["pnl"] = pnl
    results["equity"] = cfg.initial_capital + np.cumsum(pnl)
    results["ret"] = results["pnl"] / cfg.initial_capital

    # 汇总指标
    total_pnl = float(results["pnl"].sum())
    total_ret = float(total_pnl / cfg.initial_capital)
    wins = float((results["pnl"] > 0).mean()) if len(results) > 0 else 0.0
    mdd, mdd_start, mdd_end = _max_drawdown(results.set_index("exec_date")["equity"]) if len(results) > 0 else (0.0, pd.NaT, pd.NaT)

    # 专业评测指标（以交易序列近似日度）：
    # 年化收益、年化波动、夏普、索提诺、卡玛、利润因子、期望值、最大连赢/连亏、平均盈亏、偏度、峰度
    # 近似将每笔视为一个“周期”，若有日度数据可切换到按日计算
    periods_per_year = 252.0
    rets = results["pnl"].values.astype(float) / float(cfg.initial_capital)
    mean_ret = float(np.mean(rets)) if len(rets) > 0 else 0.0
    std_ret = float(np.std(rets, ddof=1)) if len(rets) > 1 else 0.0
    downside = rets[rets < 0]
    downside_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else 0.0
    ann_ret = (1.0 + mean_ret) ** periods_per_year - 1.0 if len(rets) > 0 else 0.0
    ann_vol = std_ret * np.sqrt(periods_per_year) if std_ret > 0 else 0.0
    sharpe = _safe_div(ann_ret, ann_vol)
    sortino = _safe_div(ann_ret, downside_std * np.sqrt(periods_per_year) if downside_std > 0 else 0.0)
    mdd_pct = (mdd / float(cfg.initial_capital)) if cfg.initial_capital != 0 else 0.0
    calmar = _safe_div(ann_ret, abs(mdd_pct) if mdd_pct != 0 else 0.0)
    gross_profit = float(results.loc[results["pnl"] > 0, "pnl"].sum())
    gross_loss = float(-results.loc[results["pnl"] < 0, "pnl"].sum())
    profit_factor = _safe_div(gross_profit, gross_loss if gross_loss > 0 else 0.0)
    exp_value = float(np.mean(results["pnl"].values)) if len(results) > 0 else 0.0
    max_win_streak, max_lose_streak = _compute_consecutive_stats(results["pnl"].values.astype(float))
    avg_win = float(results.loc[results["pnl"] > 0, "pnl"].mean()) if (results["pnl"] > 0).any() else 0.0
    avg_loss = float(results.loc[results["pnl"] < 0, "pnl"].mean()) if (results["pnl"] < 0).any() else 0.0
    avg_win_loss_ratio = _safe_div(avg_win, abs(avg_loss) if avg_loss != 0 else 0.0)
    # 偏度峰度
    try:
        from scipy.stats import skew, kurtosis  # type: ignore
        skewness = float(skew(rets)) if len(rets) > 0 else None
        excess_kurtosis = float(kurtosis(rets)) if len(rets) > 0 else None
    except Exception:
        skewness = None
        excess_kurtosis = None

    metrics: Dict[str, float | int | None] = {
        "trades": int(len(results)),
        "total_pnl": total_pnl,
        "total_return": total_ret,
        "win_rate": wins,
        "max_drawdown": float(mdd),
        "proba_threshold": float(cfg.proba_threshold),
        "lots": int(cfg.lots),
        "multiplier": float(cfg.lot_multiplier),
        # 扩展指标
        "annual_return": float(ann_ret),
        "annual_volatility": float(ann_vol),
        "sharpe": None if sharpe is None else float(sharpe),
        "sortino": None if sortino is None else float(sortino),
        "calmar": None if calmar is None else float(calmar),
        "max_drawdown_pct": float(mdd_pct),
        "profit_factor": None if profit_factor is None else float(profit_factor),
        "expected_pnl_per_trade": float(exp_value),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "avg_win_loss_ratio": None if avg_win_loss_ratio is None else float(avg_win_loss_ratio),
        "max_win_streak": int(max_win_streak),
        "max_loss_streak": int(max_lose_streak),
        "skew": skewness,
        "excess_kurtosis": excess_kurtosis,
    }
    return results, metrics


def run(cfg: BacktestConfig) -> None:
    ensure_dirs(cfg.output_dir)
    results, metrics = backtest_intraday(cfg)
    # 保存
    results_path = os.path.join(cfg.output_dir, "trades.csv")
    metrics_path = os.path.join(cfg.output_dir, "metrics.json")
    results.to_csv(results_path, index=False, encoding="utf-8-sig")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run(USER_CONFIG)


