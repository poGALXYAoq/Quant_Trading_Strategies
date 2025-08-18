import os
import pandas as pd
import yaml

try:
	from .engine import LongStraddleBacktester
	from .metrics import compute_metrics
except ImportError:
	import sys, os
	sys.path.append(os.path.dirname(os.path.dirname(__file__)))
	from backtest.engine import LongStraddleBacktester
	from backtest.metrics import compute_metrics


def main():
	cfg = yaml.safe_load(open("Volatility_Prediction_Strategy/config/params.yaml", "r", encoding="utf-8"))
	paths = cfg["paths"]
	bt_cfg = cfg["backtest"]
	exec_cfg = cfg["execution"]

	options_derived_csv = os.path.join(paths["output_dir"], "options_derived.csv")
	labels_csv = os.path.join(paths["output_dir"], "labels.csv")
	result_dir = paths.get("result_dir", os.path.join(paths["data_dir"], "result"))
	os.makedirs(result_dir, exist_ok=True)

	bt = LongStraddleBacktester(
		options_derived_csv=options_derived_csv,
		labels_csv=labels_csv,
		holding_period_days=bt_cfg["holding_period_days"],
		pre_expiry_buffer_days=bt_cfg["pre_expiry_buffer_days"],
		use_open_for_entry=exec_cfg["use_open_for_entry"],
		use_open_for_exit=exec_cfg["use_open_for_exit"],
		slippage_per_leg=exec_cfg["slippage_per_leg"],
		fee_per_lot=exec_cfg["fee_per_lot"],
		avoid_open_new_when_dte_leq=exec_cfg.get("avoid_open_new_when_dte_leq", 0),
		max_abs_moneyness_pct=cfg["filters"]["max_abs_moneyness_pct"],
		position_sizing=exec_cfg.get("position_sizing", {"mode": "fixed_qty", "fixed_qty": 1}),
		initial_cash=float(exec_cfg.get("initial_cash", 1_000_000.0)),
	)
	equity, trades, ledger = bt.run()

	# write results
	equity.to_csv(os.path.join(result_dir, "equity.csv"), index=False)
	trades.to_csv(os.path.join(result_dir, "trades.csv"), index=False)
	ledger.to_csv(os.path.join(result_dir, "daily_ledger.csv"), index=False)

	metrics = compute_metrics(equity.set_index("date")["equity"].astype(float))
	print(metrics)

	# Plotly visualization: multi-view
	try:
		import plotly.graph_objects as go
		from plotly.subplots import make_subplots

		# load underlying and labels for panels
		underlying = pd.read_csv(paths["underlying_csv"]) if os.path.exists(paths["underlying_csv"]) else None
		if underlying is not None and "date" in underlying.columns:
			underlying["date"] = pd.to_datetime(underlying["date"]).dt.date
		labels = pd.read_csv(labels_csv)
		labels["date"] = pd.to_datetime(labels["date"]).dt.date
		labels["neg_epsilon"] = -labels.get("epsilon", 0)

		fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03,
			subplot_titles=("Underlying Kline with Signals", "Equity Curve", "ATM IV vs Forward RV (+/- epsilon)", "Signal"), specs=[[{"secondary_y": False}],[{"secondary_y": False}],[{"secondary_y": False}],[{"secondary_y": False}]])

		# Row 1: Candlestick + signals
		if underlying is not None and set(["open","high","low","close"]).issubset(underlying.columns):
			fig.add_trace(go.Candlestick(x=underlying["date"], open=underlying["open"], high=underlying["high"], low=underlying["low"], close=underlying["close"], name="K"), row=1, col=1)
			if len(trades) > 0:
				ud = dict(zip(underlying["date"], underlying["close"]))
				entries = trades.dropna(subset=["open_date"]).copy()
				exits = trades.dropna(subset=["close_date"]).copy()
				fig.add_trace(go.Scatter(x=entries["open_date"], y=entries["open_date"].map(ud), mode="markers", name="Entry", marker=dict(color="#2ca02c", size=9, symbol="triangle-up")), row=1, col=1)
				fig.add_trace(go.Scatter(x=exits["close_date"], y=exits["close_date"].map(ud), mode="markers", name="Exit", marker=dict(color="#d62728", size=9, symbol="triangle-down")), row=1, col=1)

		# Row 2: Equity
		fig.add_trace(go.Scatter(x=equity["date"], y=equity["equity"], mode="lines", name="Equity", line=dict(color="#1f77b4")), row=2, col=1)

		# Row 3: IV vs RV with epsilon band
		if labels is not None:
			fig.add_trace(go.Scatter(x=labels["date"], y=labels.get("ATM_IV_30D", pd.Series(index=labels.index)), name="ATM_IV_30D", line=dict(color="#9467bd")), row=3, col=1)
			fig.add_trace(go.Scatter(x=labels["date"], y=labels.get("RV_30D", pd.Series(index=labels.index)), name="RV_30D_fwd", line=dict(color="#8c564b")), row=3, col=1)
			if "epsilon" in labels.columns:
				fig.add_trace(go.Scatter(x=labels["date"], y=labels["epsilon"], name="+epsilon", line=dict(color="#ff7f0e", dash="dash")), row=3, col=1)
				fig.add_trace(go.Scatter(x=labels["date"], y=labels["neg_epsilon"], name="-epsilon", line=dict(color="#ff7f0e", dash="dash")), row=3, col=1)

		# Row 4: Signal panel
		if labels is not None and "signal" in labels.columns:
			fig.add_trace(go.Scatter(x=labels["date"], y=labels["signal"], name="signal", line=dict(color="#2ca02c")), row=4, col=1)

		fig.update_layout(title="Volatility Strategy Backtest Overview", xaxis_rangeslider_visible=False, template="plotly_white", height=1200)
		overview_html = os.path.join(result_dir, "overview.html")
		fig.write_html(overview_html, include_plotlyjs="cdn")
		print(f"plot saved: {overview_html}")
	except Exception as e:
		print(f"plotly failed: {e}")


if __name__ == "__main__":
	main()
