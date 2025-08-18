import os
import pandas as pd
import numpy as np

# 并行配置预留（当前本脚本主要为列运算，不需要并行）
NUM_WORKERS = int(os.getenv("VP_NUM_WORKERS", "30"))


def realized_vol(close: pd.Series, window: int, annualization_days: int = 252) -> pd.Series:
	ret = np.log(close).diff()
	# 前瞻窗口：用未来 [t+1, t+window] 的收益，先向前移位，再 rolling，然后再对齐到 t
	rv = ret.shift(-1).rolling(window).std() * np.sqrt(annualization_days)
	return rv.shift(-(window - 1))


def build_labels(underlying_csv: str, factors_csv: str, output_csv: str,
				 rv_window_days: int = 30, epsilon_quantile: float = 0.6,
				 rolling_window_days: int = 252) -> None:
	spot = pd.read_csv(underlying_csv)
	spot["date"] = pd.to_datetime(spot["date"]).dt.date
	spot = spot.sort_values("date").reset_index(drop=True)

	fac = pd.read_csv(factors_csv)
	fac["date"] = pd.to_datetime(fac["date"]).dt.date
	fac = fac.sort_values("date").reset_index(drop=True)

	rv = realized_vol(spot["close"], rv_window_days)
	tmp = spot[["date"]].copy()
	tmp["RV_30D"] = rv

	df = fac.merge(tmp, on="date", how="left")

	df["diff"] = df["RV_30D"] - df["ATM_IV_30D"]

	# rolling epsilon threshold using past data only (shifted)
	rolling = df["diff"].rolling(rolling_window_days, min_periods=50)
	eps = rolling.quantile(epsilon_quantile).shift(1)
	df["epsilon"] = eps

	# symmetric band around 0 using epsilon; define -epsilon for short threshold
	df["signal"] = 0
	df.loc[df["diff"] > df["epsilon"], "signal"] = 1
	df.loc[df["diff"] < -df["epsilon"], "signal"] = -1

	os.makedirs(os.path.dirname(output_csv), exist_ok=True)
	df.to_csv(output_csv, index=False)


def main():
	import yaml
	cfg = yaml.safe_load(open("Volatility_Prediction_Strategy/config/params.yaml", "r", encoding="utf-8"))
	paths = cfg["paths"]
	labels_cfg = cfg["labels"]

	build_labels(
		underlying_csv=paths["underlying_csv"],
		factors_csv=os.path.join(paths["output_dir"], "factors.csv"),
		output_csv=os.path.join(paths["output_dir"], "labels.csv"),
		rv_window_days=labels_cfg["realized_vol_window_days"],
		epsilon_quantile=labels_cfg["epsilon_quantile"],
		rolling_window_days=labels_cfg["rolling_window_days"],
	)
	print("written labels.csv")


if __name__ == "__main__":
	main()
