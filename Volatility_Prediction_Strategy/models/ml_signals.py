import os
import yaml
import numpy as np
import pandas as pd

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
	基于 factors.csv 与 labels.csv（使用其 signal 作为监督标签）训练一个滚动 XGBoost 模型，
	输出按日的做多波动率概率，并按阈值生成信号（1=买跨，其余=不交易）。
	返回写出的 ml_labels.csv 路径。
	"""
	cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
	fac, lbl, result_dir = _load_data(cfg)
	ml_cfg = cfg.get("ml", {})
	threshold = float(ml_cfg.get("threshold", 0.5))
	min_train_days = int(ml_cfg.get("min_train_days", 252))

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
		if hasattr(model, "predict_proba"):
			proba[i] = float(model.predict_proba(X_test)[0, 1])
		else:
			# decision_function fallback
			score = float(model.decision_function(X_test)[0])
			# normalize to 0..1 via logistic
			proba[i] = 1.0 / (1.0 + np.exp(-score))

	# build output labels
	out = pd.DataFrame({
		"date": df["date"],
		"proba": proba,
	})
	out["signal"] = (out["proba"] >= threshold).astype(int)
	# 前期训练不足的样本，设定为不交易（0）
	out.loc[out["proba"].isna(), "signal"] = 0

	os.makedirs(result_dir, exist_ok=True)
	out_csv = os.path.join(result_dir, "ml_labels.csv")
	out[["date", "signal", "proba"]].to_csv(out_csv, index=False)
	return out_csv


if __name__ == "__main__":
	print(generate_ml_labels())


