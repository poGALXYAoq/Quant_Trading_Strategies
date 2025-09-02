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
    start_date: str = "2024-07-01"
    end_date: str = "2025-08-31"
    initial_capital: float = 500_000.0
    lot_multiplier: float = 5.0
    lots: int = 1
    neutral_band_q: float = 0.0
    tau_mode: str = "zero"  # or "zero" or median_train
    price_date_col: Optional[str] = None
    price_open_col: Optional[str] = None
    price_close_col: Optional[str] = None
    splits_to_use: Tuple[str, ...] = ("train", "valid", "test")


# ===== 可在此处直接配置默认参数（便于 IDE 一键运行） =====
_BASE_DIR = os.path.dirname(os.path.dirname(__file__))
USER_CONFIG = BacktestConfig(
    predictions_dir=os.path.join(_BASE_DIR, "results", "predictions"),
    price_csv_path="GBDT_futures/data/CU/CU888.csv",
    output_dir=os.path.join(_BASE_DIR, "results", "strategy", "swing"),
    tau_mode="zero",
    neutral_band_q=0.2,
)


def ensure_dirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _resolve_path_human_style(p: str) -> str:
    if os.path.isabs(p):
        return p
    p_norm = p.replace("\\", "/")
    if p_norm.startswith("GBDT_futures/"):
        rel = p_norm.split("/", 1)[1]
        return os.path.normpath(os.path.join(_BASE_DIR, rel))
    return os.path.normpath(os.path.join(_BASE_DIR, p))


def _get_open_on_or_after(px: pd.DataFrame, date: pd.Timestamp, date_col: str = DATE_COL) -> Tuple[Optional[float], Optional[pd.Timestamp]]:
    row = px.loc[px[date_col] == date]
    if len(row) > 0:
        return float(row["open"].iloc[0]), pd.to_datetime(row[date_col].iloc[0])
    row2 = px.loc[px[date_col] > date].head(1)
    if len(row2) > 0:
        return float(row2["open"].iloc[0]), pd.to_datetime(row2[date_col].iloc[0])
    if len(px) > 0:
        last_row = px.tail(1)
        return float(last_row["close"].iloc[0]), pd.to_datetime(last_row[date_col].iloc[0])
    return None, None


def _read_predictions(predictions_dir: str, splits: Tuple[str, ...]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for name in ("train", "valid", "test"):
        if name not in splits:
            continue
        f = os.path.join(predictions_dir, f"{name}.csv")
        if os.path.exists(f):
            df = pd.read_csv(f, encoding="utf-8-sig")
            if DATE_COL not in df.columns or "y_pred" not in df.columns:
                raise ValueError(f"预测文件缺少必要列: {f}")
            df[DATE_COL] = pd.to_datetime(df[DATE_COL])
            df["split"] = name
            frames.append(df[[DATE_COL, "y_pred", "split"]].copy())
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
        for k in df.columns:
            if "日期" in str(k):
                date_col = k
                break

    open_col = find_by_candidates(open_candidates, ["开", "盘"])  # 包含“开盘”
    close_col = find_by_candidates(close_candidates, ["收", "盘"])  # 包含“收盘”

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


def _compute_tau_band(df_pred: pd.DataFrame, q: float, mode: str) -> Tuple[float, float]:
    if q <= 0.0:
        return (0.0 if mode == "zero" else float(df_pred["y_pred"].median()), 0.0)
    df_train = df_pred[df_pred["split"] == "train"]
    base = df_train["y_pred"].values if len(df_train) > 0 else df_pred["y_pred"].values
    tau = 0.0 if mode == "zero" else float(np.median(base))
    band = float(np.quantile(np.abs(base - tau), q))
    return tau, band


def _apply_signal(y_score: np.ndarray, tau: float, band: float) -> np.ndarray:
    z = y_score - float(tau)
    sig = np.zeros_like(z, dtype=int)
    if band > 0.0:
        sig[z > band] = 1
        sig[z < -band] = -1
    else:
        sig[z > 0] = 1
        sig[z < 0] = -1
    return sig


def _max_drawdown(equity: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
    roll_max = equity.cummax()
    drawdown = equity - roll_max
    end = drawdown.idxmin()
    start = equity.loc[:end].idxmax()
    return float(drawdown.min()), pd.Timestamp(start), pd.Timestamp(end)


def backtest_swing(cfg: BacktestConfig) -> Tuple[pd.DataFrame, Dict[str, float]]:
    preds = _read_predictions(cfg.predictions_dir, cfg.splits_to_use)
    tau, band = _compute_tau_band(preds, cfg.neutral_band_q, cfg.tau_mode)
    preds["signal_int"] = _apply_signal(preds["y_pred"].values.astype(float), tau, band)

    prices = _read_prices(cfg.price_csv_path, cfg)
    prices = prices.sort_values(DATE_COL).reset_index(drop=True)
    prices["next_date"] = prices[DATE_COL].shift(-1)

    # 将信号对齐到 T+1 开盘执行
    sig = preds.merge(prices[[DATE_COL, "next_date"]], on=DATE_COL, how="left")
    sig = sig.dropna(subset=["next_date"]).copy()
    sig.rename(columns={"next_date": "exec_date"}, inplace=True)
    sig["exec_date"] = pd.to_datetime(sig["exec_date"])  # type: ignore

    # 回测窗口（基于执行日）
    start_dt = pd.to_datetime(cfg.start_date)
    end_dt = pd.to_datetime(cfg.end_date)
    sig = sig[(sig["exec_date"] >= start_dt) & (sig["exec_date"] <= end_dt)].copy()

    # 不直接合并开盘价，执行时使用 _get_open_on_or_after 以兼容非交易日

    # 生成持仓区间：当 signal 变化或转为0(None)时在下一交易日开/平
    # step1: 去重仅保留信号变点（忽略 None 信号之间的重复）
    sig = sig.sort_values("exec_date").reset_index(drop=True)
    sig["signal_run"] = sig["signal_int"].fillna(0).astype(int)
    # 合并连续重复信号
    sig = sig.loc[(sig["signal_run"] != sig["signal_run"].shift(1))].copy()
    change_rows = sig.copy()

    # step2: 遍历变点，构造从该变点起到下个变点前一日的持仓
    px = prices[[DATE_COL, "open", "close"]].copy()
    px = px[(px[DATE_COL] >= start_dt) & (px[DATE_COL] <= end_dt)].reset_index(drop=True)

    records: List[Dict[str, object]] = []
    current_pos = 0  # -1/0/1
    entry_price = None
    entry_date = None

    # 为了在最后一个区间收尾，需要一个哨兵日期
    change_dates = list(change_rows["exec_date"].values)  # numpy datetime64
    change_signals = list(change_rows["signal_run"].values)
    # 在尾部添加一个比最后价格日期晚一天的哨兵；若 change_rows 为空，使用回测开始日开仓逻辑也能被跳过
    if len(px) > 0:
        sentinel_date = pd.to_datetime(px[DATE_COL].iloc[-1]) + pd.Timedelta(days=1)
    else:
        sentinel_date = pd.to_datetime(cfg.end_date) + pd.Timedelta(days=1)
    change_dates.append(sentinel_date)
    change_signals.append(0)

    for i in range(len(change_dates) - 1):
        exec_date = pd.to_datetime(change_dates[i])
        new_sig = int(change_signals[i])
        next_change_date = pd.to_datetime(change_dates[i + 1])

        # 平掉旧仓并按规则在exec_date开新仓
        if current_pos != 0:
            # 在exec_date（或其后的最近交易日）开盘平旧仓
            close_price, eff_exit_date = _get_open_on_or_after(px, exec_date, DATE_COL)
            if eff_exit_date is not None and close_price is not None:
                pnl = (close_price - float(entry_price)) * current_pos * cfg.lot_multiplier * cfg.lots
                records.append({
                    "entry_date": entry_date,
                    "exit_date": eff_exit_date,
                    "direction": int(current_pos),
                    "entry_price": float(entry_price),
                    "exit_price": float(close_price),
                    "pnl": float(pnl),
                })
            entry_price = None
            entry_date = None
            current_pos = 0

        # 根据新信号在exec_date开新仓（若new_sig为0则不开）
        if new_sig != 0:
            open_price, eff_entry_date = _get_open_on_or_after(px, exec_date, DATE_COL)
            if eff_entry_date is not None and open_price is not None:
                entry_price = float(open_price)
                entry_date = eff_entry_date
                current_pos = int(new_sig)
            else:
                current_pos = 0
                entry_price = None
                entry_date = None

        # 若持有仓位，则从exec_date开始一直持有到下一变点前一日，期间不记录逐日，只在下一变点日开盘平仓
        # 逻辑已在上面平仓时处理

    # 若回测结束仍有未平仓位，使用最后一个有效交易日的收盘进行强制平仓
    if current_pos != 0 and entry_price is not None and len(px) > 0:
        last_close = float(px["close"].iloc[-1])
        last_date = pd.to_datetime(px[DATE_COL].iloc[-1])
        pnl = (last_close - float(entry_price)) * current_pos * cfg.lot_multiplier * cfg.lots
        records.append({
            "entry_date": entry_date,
            "exit_date": last_date,
            "direction": int(current_pos),
            "entry_price": float(entry_price),
            "exit_price": float(last_close),
            "pnl": float(pnl),
        })

    # 汇总逐笔到逐日权益
    trades = pd.DataFrame.from_records(records)
    # 生成逐日结果（即使无交易也提供完整区间）
    px_period = px[(px[DATE_COL] >= start_dt) & (px[DATE_COL] <= end_dt)].copy()
    if len(trades) == 0:
        daily = px_period[[DATE_COL]].copy()
        daily["pnl"] = 0.0
        daily["equity"] = cfg.initial_capital
    else:
        trades["pnl"] = trades["pnl"].astype(float)
        pnl_by_day = trades.groupby("exit_date")["pnl"].sum().reset_index()
        daily = px_period[[DATE_COL]].copy()
        daily = daily.merge(pnl_by_day, left_on=DATE_COL, right_on="exit_date", how="left")
        daily["pnl"] = daily["pnl"].fillna(0.0)
        daily["equity"] = cfg.initial_capital + daily["pnl"].cumsum()
        daily.drop(columns=["exit_date"], inplace=True)

    # 指标（与日内保持一致的专业化输出）
    total_pnl = float(daily["pnl"].sum())
    total_ret = float(total_pnl / cfg.initial_capital)
    mdd_val, _, _ = _max_drawdown(daily.set_index(DATE_COL)["equity"]) if len(daily) > 0 else (0.0, pd.NaT, pd.NaT)

    # 年化收益/波动/夏普/索提诺/卡玛/利润因子/期望/连赢连亏/均值/偏度峰度
    periods_per_year = 252.0
    rets = daily["pnl"].values.astype(float) / float(cfg.initial_capital)
    mean_ret = float(np.mean(rets)) if len(rets) > 0 else 0.0
    std_ret = float(np.std(rets, ddof=1)) if len(rets) > 1 else 0.0
    downside = rets[rets < 0]
    downside_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else 0.0
    ann_ret = (1.0 + mean_ret) ** periods_per_year - 1.0 if len(rets) > 0 else 0.0
    ann_vol = std_ret * np.sqrt(periods_per_year) if std_ret > 0 else 0.0

    def _safe_div(a: float, b: float) -> Optional[float]:
        try:
            if b == 0:
                return None
            return float(a) / float(b)
        except Exception:
            return None

    sharpe = _safe_div(ann_ret, ann_vol)
    sortino = _safe_div(ann_ret, downside_std * np.sqrt(periods_per_year) if downside_std > 0 else 0.0)
    mdd_pct = (mdd_val / float(cfg.initial_capital)) if cfg.initial_capital != 0 else 0.0
    calmar = _safe_div(ann_ret, abs(mdd_pct) if mdd_pct != 0 else 0.0)

    if len(trades) > 0:
        gross_profit = float(trades.loc[trades["pnl"] > 0, "pnl"].sum())
        gross_loss = float(-trades.loc[trades["pnl"] < 0, "pnl"].sum())
        profit_factor = _safe_div(gross_profit, gross_loss if gross_loss > 0 else 0.0)
        exp_value = float(np.mean(trades["pnl"].values))
        # 连赢连亏
        def _cons(pnls: np.ndarray) -> Tuple[int, int]:
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
        max_win_streak, max_lose_streak = _cons(trades["pnl"].values.astype(float))
        avg_win = float(trades.loc[trades["pnl"] > 0, "pnl"].mean()) if (trades["pnl"] > 0).any() else 0.0
        avg_loss = float(trades.loc[trades["pnl"] < 0, "pnl"].mean()) if (trades["pnl"] < 0).any() else 0.0
    else:
        profit_factor = None
        exp_value = 0.0
        max_win_streak = 0
        max_lose_streak = 0
        avg_win = 0.0
        avg_loss = 0.0

    try:
        from scipy.stats import skew, kurtosis  # type: ignore
        skewness = float(skew(rets)) if len(rets) > 0 else None
        excess_kurtosis = float(kurtosis(rets)) if len(rets) > 0 else None
    except Exception:
        skewness = None
        excess_kurtosis = None

    metrics: Dict[str, float | int | None] = {
        "trades": int(len(trades)),
        "total_pnl": total_pnl,
        "total_return": total_ret,
        "max_drawdown": float(mdd_val),
        "tau": float(tau),
        "band": float(band),
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
        "max_win_streak": int(max_win_streak),
        "max_loss_streak": int(max_lose_streak),
        "skew": skewness,
        "excess_kurtosis": excess_kurtosis,
    }

    return trades, daily, metrics


def run(cfg: BacktestConfig) -> None:
    ensure_dirs(cfg.output_dir)
    trades, daily, metrics = backtest_swing(cfg)
    # 保存
    trades_path = os.path.join(cfg.output_dir, "trades.csv")
    daily_path = os.path.join(cfg.output_dir, "daily_equity.csv")
    metrics_path = os.path.join(cfg.output_dir, "metrics.json")
    trades.to_csv(trades_path, index=False, encoding="utf-8-sig")
    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run(USER_CONFIG)


