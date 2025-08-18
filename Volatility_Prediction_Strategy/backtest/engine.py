from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import os
import pandas as pd
import numpy as np

try:
	from ..execution.selection import pick_near_dte_options
except ImportError:
	try:
		from execution.selection import pick_near_dte_options
	except ImportError:
		from Volatility_Prediction_Strategy.execution.selection import pick_near_dte_options


@dataclass
class Trade:
	open_date: pd.Timestamp
	close_date: Optional[pd.Timestamp]
	call_code: str
	put_code: str
	qty: int
	open_price: float
	close_price: Optional[float]
	dte_at_open: int
	# optional enriched fields
	entry_call_price: Optional[float] = None
	entry_put_price: Optional[float] = None
	entry_call_field: Optional[str] = None
	entry_put_field: Optional[str] = None
	exit_call_price: Optional[float] = None
	exit_put_price: Optional[float] = None
	exit_call_field: Optional[str] = None
	exit_put_field: Optional[str] = None
	open_cost: Optional[float] = None
	close_cost: Optional[float] = None


class LongStraddleBacktester:
	def __init__(self, options_derived_csv: str, labels_csv: str,
				holding_period_days: int = 10, pre_expiry_buffer_days: int = 3,
				use_open_for_entry: bool = True, use_open_for_exit: bool = False,
				slippage_per_leg: float = 0.5, fee_per_lot: float = 1.0,
				avoid_open_new_when_dte_leq: int = 0,
				max_abs_moneyness_pct: float = 0.1,
				position_sizing: Optional[Dict[str, Any]] = None,
				initial_cash: float = 1_000_000.0):
		self.opts = pd.read_csv(options_derived_csv)
		self.opts["trade_date"] = pd.to_datetime(self.opts["trade_date"]).dt.date
		self.lbl = pd.read_csv(labels_csv)
		self.lbl["date"] = pd.to_datetime(self.lbl["date"]).dt.date
		self.holding_period_days = holding_period_days
		self.pre_expiry_buffer_days = pre_expiry_buffer_days
		self.use_open_for_entry = use_open_for_entry
		self.use_open_for_exit = use_open_for_exit
		self.slippage_per_leg = slippage_per_leg
		self.fee_per_lot = fee_per_lot
		self.avoid_open_new_when_dte_leq = int(avoid_open_new_when_dte_leq or 0)
		self.max_abs_moneyness_pct = float(max_abs_moneyness_pct or 0.1)
		self.position_sizing = position_sizing or {"mode": "fixed_qty", "fixed_qty": 1}
		self.initial_cash = float(initial_cash)

	def price_on(self, row: pd.Series, use_open: bool) -> float:
		cand = ["open", "settle", "close"] if use_open else ["close", "settle", "open"]
		for c in cand:
			if c in row and not pd.isna(row[c]) and float(row[c]) > 0:
				return float(row[c])
		return float("nan")

	def price_on_with_field(self, row: pd.Series, use_open: bool) -> Tuple[float, Optional[str]]:
		cand = ["open", "settle", "close"] if use_open else ["close", "settle", "open"]
		for c in cand:
			if c in row and not pd.isna(row[c]) and float(row[c]) > 0:
				return float(row[c]), c
		return float("nan"), None

	def run(self) -> pd.DataFrame:
		dates = sorted(set(self.lbl["date"]).intersection(set(self.opts["trade_date"])) )
		trades: List[Trade] = []
		ledger: List[Dict[str, Any]] = []
		equity_rows: List[Tuple[pd.Timestamp, float]] = []
		cash = float(self.initial_cash)
		pos: Optional[Trade] = None
		contract_multiplier = float(self.position_sizing.get("contract_multiplier", 1.0)) if isinstance(self.position_sizing, dict) else 1.0
		label_map = {d: int(s) for d, s in zip(self.lbl["date"], self.lbl["signal"])}

		for i, d in enumerate(dates):
			day_action = "hold" if pos is not None else "flat"
			action_reason = None
			call_price_used = np.nan
			put_price_used = np.nan
			call_price_field = None
			put_price_field = None
			position_value = 0.0

			# exit conditions first
			if pos is not None:
				open_idx = dates.index(pos.open_date)
				hold_days = i - open_idx
				if hold_days >= self.holding_period_days or pos.dte_at_open - hold_days <= self.pre_expiry_buffer_days:
					# close today at exit price
					call_row = self.opts[(self.opts["trade_date"] == d) & (self.opts["ts_code"] == pos.call_code)]
					put_row = self.opts[(self.opts["trade_date"] == d) & (self.opts["ts_code"] == pos.put_code)]
					if not call_row.empty and not put_row.empty:
						cp, cpf = self.price_on_with_field(call_row.iloc[0], self.use_open_for_exit)
						pp, ppf = self.price_on_with_field(put_row.iloc[0], self.use_open_for_exit)
						close_val = (cp + pp) * pos.qty * contract_multiplier
						cost_fees = 2 * self.slippage_per_leg * pos.qty + 2 * self.fee_per_lot * pos.qty
						cash += close_val - cost_fees
						pos.close_date = d
						pos.close_price = close_val
						pos.exit_call_price = cp
						pos.exit_put_price = pp
						pos.exit_call_field = cpf
						pos.exit_put_field = ppf
						pos.close_cost = float(cost_fees)
						trades.append(pos)
						day_action = "close"
						action_reason = "hold_days_limit" if hold_days >= self.holding_period_days else "pre_expiry_buffer"
					pos = None
					# after closing, skip entry same day to avoid flip
					equity_rows.append((d, float(cash)))
					ledger.append({
						"date": d,
						"action": day_action,
						"reason": action_reason,
						"cash": float(cash),
						"equity": float(cash),
						"position_value": 0.0,
						"qty": 0,
						"call_code": None,
						"put_code": None,
						"price_field": None,
						"call_price": np.nan,
						"put_price": np.nan,
					})
					continue

			# entry: T+1 based on previous day's label
			if pos is None and i > 0 and label_map.get(dates[i - 1], 0) == 1:
				picked = pick_near_dte_options(self.opts, d, target_dte=30, max_abs_moneyness_pct=self.max_abs_moneyness_pct)
				if picked is not None:
					call, put = picked
					# avoid opening when DTE is too small
					try:
						min_dte = int(min(call.get("dte", np.inf), put.get("dte", np.inf)))
					except Exception:
						min_dte = np.inf
					if min_dte > self.avoid_open_new_when_dte_leq:
						cp, cpf = self.price_on_with_field(call, self.use_open_for_entry)
						pp, ppf = self.price_on_with_field(put, self.use_open_for_entry)
						if not (np.isnan(cp) or np.isnan(pp)):
							mode = (self.position_sizing.get("mode") if isinstance(self.position_sizing, dict) else "fixed_qty") or "fixed_qty"
							per_lot_cost = (cp + pp) * contract_multiplier + 2 * self.slippage_per_leg + 2 * self.fee_per_lot
							qty = 0
							if per_lot_cost > 0:
								if mode == "notional_fraction":
									notional_fraction = float(self.position_sizing.get("notional_fraction", 0.05))
									alloc = max(0.0, cash * notional_fraction)
									qty = int(np.floor(alloc / per_lot_cost))
								elif mode == "target_vega":
									try:
										per_lot_vega = float(call.get("vega", np.nan)) + float(put.get("vega", np.nan))
									except Exception:
										per_lot_vega = np.nan
									target_vega = float(self.position_sizing.get("target_vega", 0.0))
									if np.isfinite(per_lot_vega) and per_lot_vega > 0:
										qty_by_vega = int(np.floor(target_vega / (per_lot_vega * contract_multiplier)))
										qty_by_cash = int(np.floor(cash / per_lot_cost))
										qty = min(qty_by_vega, qty_by_cash)
									else:
										qty = 0
								else:
									fixed_qty = int(self.position_sizing.get("fixed_qty", 1))
									qty_by_cash = int(np.floor(cash / per_lot_cost))
									qty = min(fixed_qty, qty_by_cash)
							if qty > 0:
								cost = (cp + pp) * qty * contract_multiplier + 2 * self.slippage_per_leg * qty + 2 * self.fee_per_lot * qty
								cash -= cost
								pos = Trade(open_date=d, close_date=None, call_code=call["ts_code"], put_code=put["ts_code"], qty=qty,
										open_price=cost, close_price=None, dte_at_open=int(call["dte"]),
										entry_call_price=cp, entry_put_price=pp, entry_call_field=cpf, entry_put_field=ppf,
										open_cost=float(2 * self.slippage_per_leg * qty + 2 * self.fee_per_lot * qty))
								day_action = "open"
								action_reason = "signal_T+1"

			# compute end-of-day valuation (after actions)
			if pos is not None:
				call_row = self.opts[(self.opts["trade_date"] == d) & (self.opts["ts_code"] == pos.call_code)]
				put_row = self.opts[(self.opts["trade_date"] == d) & (self.opts["ts_code"] == pos.put_code)]
				if not call_row.empty and not put_row.empty:
					cp, cpf = self.price_on_with_field(call_row.iloc[0], self.use_open_for_exit)
					pp, ppf = self.price_on_with_field(put_row.iloc[0], self.use_open_for_exit)
					call_price_used, put_price_used = cp, pp
					call_price_field, put_price_field = cpf, ppf
					position_value = (cp + pp) * pos.qty * contract_multiplier
				else:
					position_value = 0.0
			equity_val = float(cash + position_value)
			equity_rows.append((d, equity_val))
			ledger.append({
				"date": d,
				"action": day_action,
				"reason": action_reason,
				"cash": float(cash),
				"equity": equity_val,
				"position_value": float(position_value),
				"qty": int(pos.qty) if pos is not None else 0,
				"call_code": pos.call_code if pos is not None else None,
				"put_code": pos.put_code if pos is not None else None,
				"price_field": {"call": call_price_field, "put": put_price_field} if pos is not None else None,
				"call_price": float(call_price_used) if pos is not None else np.nan,
				"put_price": float(put_price_used) if pos is not None else np.nan,
			})

		eq_df = pd.DataFrame(equity_rows, columns=["date", "equity"]).drop_duplicates("date").sort_values("date")
		trades_df = pd.DataFrame([
			{
				"open_date": t.open_date,
				"close_date": t.close_date,
				"call_code": t.call_code,
				"put_code": t.put_code,
				"qty": t.qty,
				"open_price": t.open_price,
				"close_price": t.close_price,
				"pnl": (t.close_price - t.open_price) if (t.close_price is not None and t.open_price is not None) else np.nan,
				"dte_at_open": t.dte_at_open,
				"entry_call_price": t.entry_call_price,
				"entry_put_price": t.entry_put_price,
				"entry_call_field": t.entry_call_field,
				"entry_put_field": t.entry_put_field,
				"exit_call_price": t.exit_call_price,
				"exit_put_price": t.exit_put_price,
				"exit_call_field": t.exit_call_field,
				"exit_put_field": t.exit_put_field,
				"open_cost": t.open_cost,
				"close_cost": t.close_cost,
			}
			for t in trades
		]) if trades else pd.DataFrame(columns=[
			"open_date","close_date","call_code","put_code","qty","open_price","close_price","pnl","dte_at_open",
			"entry_call_price","entry_put_price","entry_call_field","entry_put_field","exit_call_price","exit_put_price","exit_call_field","exit_put_field","open_cost","close_cost"
		])
		ledger_df = pd.DataFrame(ledger)
		return eq_df, trades_df, ledger_df
