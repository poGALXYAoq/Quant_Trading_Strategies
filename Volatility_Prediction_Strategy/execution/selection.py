from typing import Optional, Tuple
import pandas as pd
import numpy as np


def pick_near_dte_options(options_derived: pd.DataFrame, trade_date, target_dte: int = 30,
                           max_abs_moneyness_pct: float = 0.1) -> Optional[Tuple[pd.Series, pd.Series]]:
    day_df = options_derived[options_derived["trade_date"] == pd.to_datetime(trade_date).date()].copy()
    if day_df.empty:
        return None
    # filter by moneyness window
    day_df["moneyness_pct"] = (day_df["spot"] - day_df["strike"]).abs() / np.where(day_df["spot"] != 0, day_df["spot"], np.nan)
    day_df = day_df[day_df["moneyness_pct"] <= max_abs_moneyness_pct]
    if day_df.empty:
        return None
    # pick closest DTE first
    day_df["dte_diff"] = (day_df["dte"] - target_dte).abs()
    # select ATM by minimizing |S-K|
    def _best_of(side):
        cands = day_df[day_df["call_put"] == side]
        if cands.empty:
            return None
        cands = cands.sort_values(["dte_diff", "moneyness_pct", "strike"]).head(1)
        return cands.iloc[0]
    call = _best_of("C")
    put = _best_of("P")
    if call is None or put is None:
        return None
    return call, put
