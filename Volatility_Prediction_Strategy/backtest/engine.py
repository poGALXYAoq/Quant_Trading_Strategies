from dataclasses import dataclass
from typing import List, Optional, Tuple
import os
import pandas as pd
import numpy as np

from ..execution.selection import pick_near_dte_options


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


class LongStraddleBacktester:
	def __init__(self, options_derived_csv: str, labels_csv: str,
				holding_period_days: int = 10, pre_expiry_buffer_days: int = 3,
				use_open_for_entry: bool = True, use_open_for_exit: bool = False,
				slippage_per_leg: float = 0.5, fee_per_lot: float = 1.0):
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

	def price_on(self, row: pd.Series, use_open: bool) -> float:
		cand = ["open", "close"] if use_open else ["close", "open"]
		for c in cand + ["settle"]:
			if c in row and not pd.isna(row[c]) and float(row[c]) > 0:
				return float(row[c])
		return float("nan")

	def run(self) -> pd.DataFrame:
		dates = sorted(set(self.lbl["date"]).intersection(set(self.opts["trade_date"])) )
		trades: List[Trade] = []
		equity = []
		cash = 1_000_000.0
		pos: Optional[Trade] = None

		for i, d in enumerate(dates):
			# daily mark-to-market
			if pos is not None:
				call_row = self.opts[(self.opts["trade_date"] == d) & (self.opts["ts_code"] == pos.call_code)]
				put_row = self.opts[(self.opts["trade_date"] == d) & (self.opts["ts_code"] == pos.put_code)]
				if not call_row.empty and not put_row.empty:
					cp = self.price_on(call_row.iloc[0], self.use_open_for_exit)
					pp = self.price_on(put_row.iloc[0], self.use_open_for_exit)
					mtm = (cp + pp) * pos.qty
					equity.append((d, cash + mtm))
				else:
					equity.append((d, cash))
			else:
				equity.append((d, cash))

			# exit conditions
			if pos is not None:
				open_idx = dates.index(pos.open_date)
				hold_days = i - open_idx
				if hold_days >= self.holding_period_days or pos.dte_at_open - hold_days <= self.pre_expiry_buffer_days:
					# close today at exit price (use open or close per config)
					call_row = self.opts[(self.opts["trade_date"] == d) & (self.opts["ts_code"] == pos.call_code)]
					put_row = self.opts[(self.opts["trade_date"] == d) & (self.opts["ts_code"] == pos.put_code)]
					if not call_row.empty and not put_row.empty:
						cp = self.price_on(call_row.iloc[0], self.use_open_for_exit)
						pp = self.price_on(put_row.iloc[0], self.use_open_for_exit)
						close_price = (cp + pp) * pos.qty - 2 * self.slippage_per_leg * pos.qty - 2 * self.fee_per_lot * pos.qty
						cash += close_price
						pos.close_date = d
						pos.close_price = close_price
						trades.append(pos)
					pos = None
					continue

			# entry: only when no position and label==1
			sig_row = self.lbl[self.lbl["date"] == d]
			if pos is None and not sig_row.empty and sig_row.iloc[0]["signal"] == 1:
				picked = pick_near_dte_options(self.opts, d, target_dte=30, max_abs_moneyness_pct=0.1)
				if picked is None:
					continue
				call, put = picked
				cp = self.price_on(call, self.use_open_for_entry)
				pp = self.price_on(put, self.use_open_for_entry)
				if np.isnan(cp) or np.isnan(pp):
					continue
				qty = 1
				cost = (cp + pp) * qty + 2 * self.slippage_per_leg * qty + 2 * self.fee_per_lot * qty
				cash -= cost
				pos = Trade(open_date=d, close_date=None, call_code=call["ts_code"], put_code=put["ts_code"], qty=qty,
						  open_price=cost, close_price=None, dte_at_open=int(call["dte"]))

		eq_df = pd.DataFrame(equity, columns=["date", "equity"]).drop_duplicates("date").sort_values("date")
		return eq_df
