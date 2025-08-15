import os
import pandas as pd
import yaml

from .engine import LongStraddleBacktester
from .metrics import compute_metrics


def main():
	cfg = yaml.safe_load(open("Volatility_Prediction_Strategy/config/params.yaml", "r", encoding="utf-8"))
	paths = cfg["paths"]
	bt_cfg = cfg["backtest"]
	exec_cfg = cfg["execution"]

	options_derived_csv = os.path.join(paths["output_dir"], "options_derived.csv")
	labels_csv = os.path.join(paths["output_dir"], "labels.csv")

	bt = LongStraddleBacktester(
		options_derived_csv=options_derived_csv,
		labels_csv=labels_csv,
		holding_period_days=bt_cfg["holding_period_days"],
		pre_expiry_buffer_days=bt_cfg["pre_expiry_buffer_days"],
		use_open_for_entry=exec_cfg["use_open_for_entry"],
		use_open_for_exit=exec_cfg["use_open_for_exit"],
		slippage_per_leg=exec_cfg["slippage_per_leg"],
		fee_per_lot=exec_cfg["fee_per_lot"],
	)
	equity = bt.run()
	os.makedirs(paths["output_dir"], exist_ok=True)
	out_csv = os.path.join(paths["output_dir"], "equity.csv")
	equity.to_csv(out_csv, index=False)

	metrics = compute_metrics(equity.set_index("date")["equity"].astype(float))
	print(metrics)


if __name__ == "__main__":
	main()
