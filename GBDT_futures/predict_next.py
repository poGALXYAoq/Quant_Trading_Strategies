import os
import json
import argparse
from typing import Dict

import pandas as pd
from xgboost import XGBRegressor

DATE_COL = "date"
PRICE_COL = "加权平均价(主力合约):沪铜(9:00-15:00)"


# ===== 可在此处直接修改默认运行配置（无需命令行） =====
USER_CONFIG: Dict[str, object] = {
    "data_path": os.path.join(os.path.dirname(__file__), "data", "以收盘价为label.csv"),
    # 指向 train_refit.py 生成的部署目录根。若为空，将自动从 results_deploy 下选择最新 cut_ 目录。
    "deploy_root": "",
    # 方式A：最近 N 天（保留兼容）
    "days": 0,  # 设为 >0 启用此模式；否则使用日期范围
    # 方式B：按日期范围选择（包含端点）。若输入非交易日，将自动向内收敛到最近可用日期。
    "start_date": "2025-08-01",
    "end_date": "2025-08-26",
}

# 是否启用命令行参数。False 表示忽略命令行，仅使用上面的 USER_CONFIG
USE_CLI_ARGS = False


def _find_latest_deploy_root(base_dir: str) -> str:
    rd = os.path.join(base_dir, "results_deploy")
    if not os.path.isdir(rd):
        raise FileNotFoundError("找不到 results_deploy 目录，请先运行 train_refit.py")
    # 选择按目录名排序后最新的 cut_*
    cut_dirs = [d for d in os.listdir(rd) if d.startswith("cut_") and os.path.isdir(os.path.join(rd, d))]
    if not cut_dirs:
        raise FileNotFoundError("results_deploy 下未发现 cut_* 目录，请先运行 train_refit.py")
    cut_dirs.sort()
    return os.path.join(rd, cut_dirs[-1])


def _slice_by_date(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    # 包含端点，若端点非交易日，则向内收敛到最近可用交易日
    if start_date:
        sd = pd.to_datetime(start_date)
        # 找到 >= sd 的第一天；若所有交易日都 < sd，则返回空
        mask = df[DATE_COL] >= sd
        if mask.any():
            left = df.index[mask][0]
        else:
            return df.iloc[0:0]
    else:
        left = 0
    if end_date:
        ed = pd.to_datetime(end_date)
        mask = df[DATE_COL] <= ed
        if mask.any():
            right = df.index[mask][-1]
        else:
            return df.iloc[0:0]
    else:
        right = len(df) - 1
    if right < left:
        return df.iloc[0:0]
    return df.iloc[left:right + 1]


def run_predict(data_path: str, deploy_root: str | None, days: int, start_date: str = "", end_date: str = "") -> None:
    base_dir = os.path.dirname(__file__)
    if not deploy_root:
        deploy_root = _find_latest_deploy_root(base_dir)
    arts = os.path.join(deploy_root, "artifacts")
    models = os.path.join(deploy_root, "models")

    feature_cols = pd.read_csv(os.path.join(arts, "feature_names.csv"))
    feature_cols = feature_cols[feature_cols.columns[0]].tolist()
    med_df = pd.read_csv(os.path.join(arts, "feature_medians.csv"))
    medians = med_df.set_index(med_df.columns[0]).iloc[:, 0]

    df = pd.read_csv(data_path, encoding="utf-8-sig")
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    X = df[feature_cols].copy()
    X = X.fillna(medians.reindex(feature_cols))

    model = XGBRegressor()
    model.load_model(os.path.join(models, "xgb_model_final.json"))

    # 选择预测区间
    eff_start = ""
    eff_end = ""
    if int(days) > 0:
        n = int(days)
        n = max(1, min(n, len(df)))
        df_sel = df.tail(n).reset_index(drop=True)
        X_sel = X.tail(n).reset_index(drop=True)
        eff_start = str(df_sel[DATE_COL].iloc[0].date())
        eff_end = str(df_sel[DATE_COL].iloc[-1].date())
    else:
        df_sel = _slice_by_date(df, start_date, end_date).reset_index(drop=True)
        if len(df_sel) == 0:
            print(json.dumps({
                "error": "所选日期范围内无可用交易日",
                "start_date": start_date,
                "end_date": end_date
            }, ensure_ascii=False, indent=2))
            return
        X_sel = X.loc[df_sel.index]
        eff_start = str(df_sel[DATE_COL].iloc[0].date())
        eff_end = str(df_sel[DATE_COL].iloc[-1].date())

    rows = X_sel
    dates = df_sel[DATE_COL]
    closes = df_sel[PRICE_COL]

    outputs = []
    for i in range(n):
        x = rows.iloc[[i]].values
        d = dates.iloc[i]
        p = float(closes.iloc[i])
        y_hat = float(model.predict(x)[0])
        p_next = p + y_hat
        outputs.append({
            "date": str(d.date()),
            "close": p,
            "predicted_delta": y_hat,
            "predicted_next_close": p_next
        })

    print(json.dumps({
        "deploy_root": deploy_root,
        "effective_start": eff_start,
        "effective_end": eff_end,
        "predictions": outputs
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    if USE_CLI_ARGS:
        parser = argparse.ArgumentParser("Predict next-day delta for a chosen date range or last N days")
        parser.add_argument("--data_path", type=str, required=True)
        parser.add_argument("--deploy_root", type=str, default="", help="train_refit.py 输出目录（results_deploy/cut_*）")
        parser.add_argument("--days", type=int, default=0, help="最近 N 天。若为 0 则使用 --start_date/--end_date")
        parser.add_argument("--start_date", type=str, default="", help="起始日期 YYYY-MM-DD（可非交易日，自动收敛）")
        parser.add_argument("--end_date", type=str, default="", help="结束日期 YYYY-MM-DD（可非交易日，自动收敛）")
        args = parser.parse_args()
        run_predict(args.data_path, args.deploy_root or None, int(args.days), args.start_date, args.end_date)
    else:
        cfg = USER_CONFIG
        run_predict(
            str(cfg["data_path"]),
            (str(cfg.get("deploy_root")) or None),
            int(cfg.get("days", 0)),
            str(cfg.get("start_date", "")),
            str(cfg.get("end_date", "")),
        )


