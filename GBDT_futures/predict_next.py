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
    "data_path": os.path.join(os.path.dirname(__file__), "data", "沪铜_20170101_20250731.csv"),
    # 指向 train_refit.py 生成的部署目录根。若为空，将自动从 results_deploy 下选择最新 cut_ 目录。
    "deploy_root": "",
    "days": 1,
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


def run_predict(data_path: str, deploy_root: str | None, days: int) -> None:
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

    n = int(days)
    n = max(1, min(n, len(df)))
    rows = X.tail(n)
    dates = df[DATE_COL].tail(n).reset_index(drop=True)
    closes = df[PRICE_COL].tail(n).reset_index(drop=True)

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

    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    if USE_CLI_ARGS:
        parser = argparse.ArgumentParser("Load saved model and predict next-day delta for the latest N days")
        parser.add_argument("--data_path", type=str, required=True)
        parser.add_argument("--deploy_root", type=str, default="", help="train_refit.py 输出目录（results_deploy/cut_*）")
        parser.add_argument("--days", type=int, default=1, help="对最近 N 天做次日预测（默认1天，仅最后一天）")
        args = parser.parse_args()
        run_predict(args.data_path, args.deploy_root or None, int(args.days))
    else:
        cfg = USER_CONFIG
        run_predict(str(cfg["data_path"]), (str(cfg.get("deploy_root")) or None), int(cfg.get("days", 1)))


