import numpy as np
import pandas as pd


def compute_metrics(equity: pd.Series, freq_per_year: int = 252) -> dict:
	ret = equity.pct_change().fillna(0.0)
	cumret = equity.iloc[-1] / equity.iloc[0] - 1.0 if len(equity) > 1 else 0.0
	ann_ret = (1.0 + cumret) ** (freq_per_year / max(len(ret), 1)) - 1.0 if len(ret) > 1 else 0.0
	ann_vol = ret.std() * np.sqrt(freq_per_year)
	sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
	peak = equity.cummax()
	dd = (equity / peak - 1.0).min()
	return {
		"cum_return": float(cumret),
		"ann_return": float(ann_ret),
		"ann_vol": float(ann_vol),
		"sharpe": float(sharpe),
		"max_drawdown": float(dd),
	}
