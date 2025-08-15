import os
import glob
import datetime as dt
from typing import List

import pandas as pd
import numpy as np
from joblib import Parallel, delayed

# Make local imports work when running as a script
import sys as _sys
_sys.path.append(os.path.dirname(__file__))

from iv_utils import (
	parse_ts_code,
	infer_expiration_date,
	compute_dte,
	implied_vol,
	bsm_greeks,
	choose_option_price_for_iv,
	rate_lookup_1m,
)

# 全局并行度（由环境变量控制，或直接修改常量）
NUM_WORKERS = int(os.getenv("VP_NUM_WORKERS", "30"))


def load_underlying(csv_path: str) -> pd.DataFrame:
	df = pd.read_csv(csv_path)
	df["date"] = pd.to_datetime(df["date"]).dt.date
	df = df.sort_values("date").reset_index(drop=True)
	return df


def load_rates(csv_path: str) -> pd.DataFrame:
	return pd.read_csv(csv_path)


def _prepare_base(df: pd.DataFrame,
				   underlying_df: pd.DataFrame,
				   rates_df: pd.DataFrame,
				   cal: pd.Index,
				   tenor_preference: str,
				   q_annual: float,
				   annualization_days: int) -> pd.DataFrame:
	# 解析基本字段
	cp, strike, cm = zip(*df["ts_code"].map(parse_ts_code))
	df["call_put"] = list(cp)
	df["strike"] = list(strike)
	df["contract_month"] = list(cm)
	df["expiration_date"] = df["contract_month"].map(infer_expiration_date)
	# DTE/Spot/RF
	spot_series = underlying_df.set_index("date")["close"]
	df["dte"] = [compute_dte(t, e, cal) for t, e in zip(df["trade_date"], df["expiration_date"])]
	df["spot"] = df["trade_date"].map(spot_series)
	df["rf_annual"] = df["trade_date"].map(lambda d: rate_lookup_1m(rates_df, d, tenor_preference))
	df["q_annual"] = q_annual
	df["t_years"] = df["dte"].astype(float) / float(annualization_days)
	df["price_for_iv"] = df.apply(choose_option_price_for_iv, axis=1)
	return df


def _compute_iv_greeks_chunk(sub: pd.DataFrame) -> pd.DataFrame:
	# 计算 IV
	def _iv_row(r):
		if not pd.notna(r["price_for_iv"]) or r["price_for_iv"] <= 0 or r["spot"] <= 0 or r["strike"] <= 0 or r["t_years"] <= 0:
			return float("nan")
		is_call = r["call_put"] == "C"
		return implied_vol(
			price=float(r["price_for_iv"]),
			is_call=is_call,
			s=float(r["spot"]),
			k=float(r["strike"]),
			t=float(r["t_years"]),
			r=float(r["rf_annual"]),
			q=float(r["q_annual"]),
		)

	sub["iv"] = [
		_iv_row(row)
		for _, row in sub.iterrows()
	]
	# 计算 Greeks
	deltas, gammas, vegas, thetas = [], [], [], []
	for _, row in sub.iterrows():
		if not pd.notna(row["iv"]):
			deltas.append(np.nan); gammas.append(np.nan); vegas.append(np.nan); thetas.append(np.nan)
			continue
		is_call = row["call_put"] == "C"
		d, g, v, th = bsm_greeks(
			is_call,
			float(row["spot"]),
			float(row["strike"]),
			float(row["t_years"]),
			float(row["rf_annual"]),
			float(row["q_annual"]),
			float(row["iv"]),
		)
		deltas.append(d); gammas.append(g); vegas.append(v); thetas.append(th)
	sub["delta"], sub["gamma"], sub["vega"], sub["theta"] = deltas, gammas, vegas, thetas
	return sub


def _process_file_single(file_path: str,
						   underlying_df: pd.DataFrame,
						   rates_df: pd.DataFrame,
						   cal: pd.Index,
						   tenor_preference: str,
						   q_annual: float,
						   annualization_days: int,
						   chunk_jobs: int = 1) -> pd.DataFrame:
	df = pd.read_csv(file_path)
	df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d").dt.date
	df = _prepare_base(df, underlying_df, rates_df, cal, tenor_preference, q_annual, annualization_days)

	if chunk_jobs <= 1:
		return _compute_iv_greeks_chunk(df)
	# 单文件或超大文件：按索引切块并行
	parts = np.array_split(df.index.values, chunk_jobs)
	chunks = Parallel(n_jobs=chunk_jobs, backend="loky")(delayed(_compute_iv_greeks_chunk)(df.loc[idxs].copy()) for idxs in parts)
	return pd.concat(chunks).sort_index()


def build_options_derived(options_glob: str,
						  underlying_csv: str,
						  rates_csv: str,
						  annualization_days: int = 252,
						  tenor_preference: str = "1M",
						  q_annual: float = 0.0) -> pd.DataFrame:
	files: List[str] = sorted(glob.glob(options_glob))
	if not files:
		raise FileNotFoundError(f"No option files matched: {options_glob}")

	underlying_df = load_underlying(underlying_csv)
	rates_df = load_rates(rates_csv)
	cal = pd.Index(pd.to_datetime(underlying_df["date"]))

	if len(files) == 1:
		# 单文件：行级并行
		frame = _process_file_single(files[0], underlying_df, rates_df, cal, tenor_preference, q_annual, annualization_days, chunk_jobs=max(1, NUM_WORKERS))
		out = frame
	else:
		# 多文件：文件级并行（每文件单进程），文件内顺序（避免嵌套并行）
		frames = Parallel(n_jobs=min(NUM_WORKERS, len(files)), backend="loky")(delayed(_process_file_single)(
			f, underlying_df, rates_df, cal, tenor_preference, q_annual, annualization_days, 1
		) for f in files)
		out = pd.concat(frames, ignore_index=True)

	out = out.sort_values(["trade_date", "expiration_date", "strike", "call_put"]).reset_index(drop=True)
	return out


def main():
	import yaml
	cfg = yaml.safe_load(open("Volatility_Prediction_Strategy/config/params.yaml", "r", encoding="utf-8"))
	paths = cfg["paths"]
	rates_cfg = cfg["rates"]
	iv_cfg = cfg["iv_inversion"]

	os.makedirs(paths["output_dir"], exist_ok=True)

	# 直接在 main 中用 joblib 管理并行，避免嵌套并行导致的资源浪费
	files: List[str] = sorted(glob.glob(paths["options_glob"]))
	if not files:
		raise FileNotFoundError(f"No option files matched: {paths['options_glob']}")
	underlying_df = load_underlying(paths["underlying_csv"])
	rates_df = load_rates(paths["rates_csv"])
	cal = pd.Index(pd.to_datetime(underlying_df["date"]))

	if len(files) == 1:
		# 单文件：行级并行
		df = pd.read_csv(files[0])
		df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d").dt.date
		df = _prepare_base(df, underlying_df, rates_df, cal, rates_cfg["tenor_preference"], iv_cfg["dividend_yield_annual"], rates_cfg["annualization_days"])
		parts = np.array_split(df.index.values, max(1, NUM_WORKERS))
		chunks = Parallel(n_jobs=max(1, NUM_WORKERS), backend="loky")(delayed(_compute_iv_greeks_chunk)(df.loc[idxs].copy()) for idxs in parts)
		options_derived = pd.concat(chunks).sort_index()
	else:
		# 多文件：文件级并行（每文件单进程），文件内顺序（避免嵌套）
		def _process_file_no_inner_parallel(fpath: str) -> pd.DataFrame:
			fdf = pd.read_csv(fpath)
			fdf["trade_date"] = pd.to_datetime(fdf["trade_date"], format="%Y%m%d").dt.date
			fdf = _prepare_base(fdf, underlying_df, rates_df, cal, rates_cfg["tenor_preference"], iv_cfg["dividend_yield_annual"], rates_cfg["annualization_days"])
			return _compute_iv_greeks_chunk(fdf)

		frames = Parallel(n_jobs=min(NUM_WORKERS, len(files)), backend="loky")(delayed(_process_file_no_inner_parallel)(f) for f in files)
		options_derived = pd.concat(frames, ignore_index=True)

	out_csv = os.path.join(paths["output_dir"], "options_derived.csv")
	options_derived.to_csv(out_csv, index=False)
	print(f"written: {out_csv}")


if __name__ == "__main__":
	main()
