import os
import yaml
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List

# optional dependency flags
try:
	import sklearn as _sk
	_SKLEARN_AVAILABLE = True
except Exception:
	_SKLEARN_AVAILABLE = False


class SimpleLogistic:
	"""Minimal logistic regression using numpy, for environments without sklearn.
	Binary classification y in {0,1}. Provides fit() and predict_proba()."""

	def __init__(self, lr: float = 0.1, n_iter: int = 200, l2: float = 0.0):
		self.lr = float(lr)
		self.n_iter = int(n_iter)
		self.l2 = float(l2)
		self.w = None
		self.b = 0.0

	def _sigmoid(self, z):
		# clip to avoid overflow
		z = np.clip(z, -50, 50)
		return 1.0 / (1.0 + np.exp(-z))

	def fit(self, X: np.ndarray, y: np.ndarray):
		X = np.asarray(X, dtype=float)
		y = np.asarray(y, dtype=float)
		m, n = X.shape
		if self.w is None:
			self.w = np.zeros(n, dtype=float)
			self.b = 0.0
		for _ in range(self.n_iter):
			z = X.dot(self.w) + self.b
			p = self._sigmoid(z)
			error = p - y
			grad_w = X.T.dot(error) / max(m, 1) + self.l2 * self.w
			grad_b = float(np.sum(error) / max(m, 1))
			self.w -= self.lr * grad_w
			self.b -= self.lr * grad_b
		return self

	def predict_proba(self, X: np.ndarray):
		X = np.asarray(X, dtype=float)
		z = X.dot(self.w) + self.b
		p = self._sigmoid(z)
		# return 2D array (n_samples, 2) to mimic sklearn
		p = p.reshape(-1, 1)
		return np.hstack([1.0 - p, p])


def _load_data(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, str]:
	paths = cfg["paths"]
	output_dir = paths["output_dir"]
	result_dir = paths.get("result_dir", os.path.join(paths["data_dir"], "result"))
	factors_csv = os.path.join(output_dir, "factors.csv")
	labels_csv = os.path.join(output_dir, "labels.csv")
	if not os.path.exists(factors_csv):
		raise FileNotFoundError(f"factors.csv not found: {factors_csv}")
	if not os.path.exists(labels_csv):
		raise FileNotFoundError(f"labels.csv not found: {labels_csv}")
	fac = pd.read_csv(factors_csv)
	lbl = pd.read_csv(labels_csv)
	fac["date"] = pd.to_datetime(fac["date"]).dt.date
	lbl["date"] = pd.to_datetime(lbl["date"]).dt.date
	return fac, lbl, result_dir


def _select_features(df: pd.DataFrame) -> list[str]:
	candidates = [
		"HV_20D", "HV_60D",
		"ATM_IV_30D", "ATM_IV_60D", "TERM_STRUCTURE_60D_30D",
		"SKEW_30D_25DELTA",
		"PCR_OI", "PCR_VOL",
	]
	return [c for c in candidates if c in df.columns]


def generate_ml_labels(config_path: str = "Volatility_Prediction_Strategy/config/params.yaml") -> str:
	"""
	基于 factors.csv 与 labels.csv（使用其 signal 作为监督标签）训练模型并生成信号：
	- 默认使用步进向前（walk_forward）逐日训练与预测；
	- 当 ml.training_mode=split 时，按给定的 train/val/test 时间段：
	  * 在 train 训练一个固定模型；
	  * 在 val 上通过阈值搜索选择最优阈值；
	  * 在 test 上输出预测概率与最终信号（1=买跨，其余=不交易）。
	返回写出的 ml_labels.csv 路径。
	"""
	cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
	fac, lbl, result_dir = _load_data(cfg)
	ml_cfg = cfg.get("ml", {})
	threshold = float(ml_cfg.get("threshold", 0.5))
	min_train_days = int(ml_cfg.get("min_train_days", 252))
	training_mode = (ml_cfg.get("training_mode") or "walk_forward").lower()

	# merge
	df = fac.merge(lbl[["date", "signal"]], on="date", how="inner").sort_values("date").reset_index(drop=True)
	features = _select_features(df)
	if not features:
		raise ValueError("No available features found for ML training.")
	df = df.dropna(subset=features + ["signal"])  # drop rows with NA in features or label
	df["y"] = (df["signal"] == 1).astype(int)

	X = df[features].astype(float).values
	y = df["y"].astype(int).values
	dates = df["date"].values

	# model
	model_name = (ml_cfg.get("model") or "xgboost").lower()
	clf_factory = None
	if model_name == "xgboost":
		try:
			from xgboost import XGBClassifier
			# xgboost sklearn wrapper requires sklearn
			if _SKLEARN_AVAILABLE:
				clf_factory = lambda: XGBClassifier(
					n_estimators=200,
					max_depth=3,
					learning_rate=0.05,
					subsample=0.8,
					colsample_bytree=0.8,
					random_state=42,
					n_jobs=0,
					reg_alpha=0.0,
					reg_lambda=1.0,
				)
			else:
				clf_factory = None
		except Exception:
			clf_factory = None
	# sklearn fallback
	if clf_factory is None and _SKLEARN_AVAILABLE:
		try:
			from sklearn.ensemble import GradientBoostingClassifier
			clf_factory = lambda: GradientBoostingClassifier(random_state=42)
		except Exception:
			clf_factory = None
	# numpy-only fallback
	if clf_factory is None:
		clf_factory = lambda: SimpleLogistic(lr=0.1, n_iter=300, l2=0.0)

	def _predict_proba(model, X_part: np.ndarray) -> np.ndarray:
		if hasattr(model, "predict_proba"):
			pp = model.predict_proba(X_part)
			return np.asarray(pp)[:, 1].astype(float)
		else:
			# decision_function fallback
			score = np.asarray(model.decision_function(X_part)).astype(float).reshape(-1)
			return 1.0 / (1.0 + np.exp(-score))

	def _f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
		try:
			if _SKLEARN_AVAILABLE:
				from sklearn.metrics import f1_score as _f1
				return float(_f1(y_true, y_pred))
		except Exception:
			pass
		# minimal fallback
		tp = float(np.sum((y_true == 1) & (y_pred == 1)))
		fp = float(np.sum((y_true == 0) & (y_pred == 1)))
		fn = float(np.sum((y_true == 1) & (y_pred == 0)))
		precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
		recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
		return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

	def _mask_between(date_series: pd.Series, start: Optional[str], end: Optional[str]) -> np.ndarray:
		if start:
			sd = pd.to_datetime(start).date()
		else:
			sd = None
		if end:
			ed = pd.to_datetime(end).date()
		else:
			ed = None
		if sd and ed:
			return (date_series >= sd) & (date_series <= ed)
		elif sd:
			return (date_series >= sd)
		elif ed:
			return (date_series <= ed)
		else:
			return np.ones(len(date_series), dtype=bool)

	os.makedirs(result_dir, exist_ok=True)

	if training_mode == "split":
		# parse split
		split = ml_cfg.get("split", {}) or {}
		train_range = split.get("train") or [None, None]
		val_range = split.get("val") or [None, None]
		test_range = split.get("test") or [None, None]
		date_s = pd.Series(df["date"])  # dtype: date
		mask_train = _mask_between(date_s, train_range[0], train_range[1])
		mask_val = _mask_between(date_s, val_range[0], val_range[1])
		mask_test = _mask_between(date_s, test_range[0], test_range[1])

		# train one model on train
		X_tr, y_tr = X[mask_train], y[mask_train]
		if len(X_tr) == 0:
			raise ValueError("Empty training set under provided ml.split.train range.")
		try:
			model = clf_factory()
			model.fit(X_tr, y_tr)
		except Exception:
			model = SimpleLogistic(lr=0.1, n_iter=300, l2=0.0)
			model.fit(X_tr, y_tr)

		# choose threshold on validation
		thr = float(threshold)
		if np.any(mask_val):
			proba_val = _predict_proba(model, X[mask_val])
			y_val = y[mask_val]
			method = (ml_cfg.get("threshold_selection") or "f1").lower()
			grid = ml_cfg.get("threshold_grid") or [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
			best_score = -1.0
			best_thr = thr
			for t in grid:
				y_hat = (proba_val >= float(t)).astype(int)
				if method == "f1":
					s = _f1_score(y_val, y_hat)
				else:
					# default to f1
					s = _f1_score(y_val, y_hat)
				if s > best_score:
					best_score, best_thr = s, float(t)
			thr = best_thr

		# predict on test
		proba_test = _predict_proba(model, X[mask_test]) if np.any(mask_test) else np.array([], dtype=float)
		out = pd.DataFrame({
			"date": df.loc[mask_test, "date"].values,
			"proba": proba_test,
		})
		out["signal"] = (out["proba"] >= thr).astype(int)
		# write outputs (test-only as ml_labels.csv; and optional all for inspection)
		out_csv = os.path.join(result_dir, "ml_labels.csv")
		out[["date", "signal", "proba"]].to_csv(out_csv, index=False)
		# also dump full-set probabilities for audit (optional)
		try:
			proba_all = _predict_proba(model, X)
			out_all = pd.DataFrame({"date": df["date"].values, "proba": proba_all})
			out_all["signal"] = (out_all["proba"] >= thr).astype(int)
			out_all.to_csv(os.path.join(result_dir, "ml_labels_all.csv"), index=False)
		except Exception:
			pass
		return out_csv

	# walk-forward training: each day train on past data, predict current
	proba = np.full(shape=len(df), fill_value=np.nan, dtype=float)
	for i in range(len(df)):
		if i < min_train_days:
			continue
		X_train = X[:i]
		y_train = y[:i]
		X_test = X[i:i+1]
		try:
			model = clf_factory()
			model.fit(X_train, y_train)
		except Exception:
			# last-resort fallback: SimpleLogistic
			model = SimpleLogistic(lr=0.1, n_iter=300, l2=0.0)
			model.fit(X_train, y_train)
		proba[i] = float(_predict_proba(model, X_test)[0])

	# build output labels
	out = pd.DataFrame({
		"date": df["date"],
		"proba": proba,
	})
	out["signal"] = (out["proba"] >= threshold).astype(int)
	# 前期训练不足的样本，设定为不交易（0）
	out.loc[out["proba"].isna(), "signal"] = 0

	out_csv = os.path.join(result_dir, "ml_labels.csv")
	out[["date", "signal", "proba"]].to_csv(out_csv, index=False)
	return out_csv


if __name__ == "__main__":
	print(generate_ml_labels())


