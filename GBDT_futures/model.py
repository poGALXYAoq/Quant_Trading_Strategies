import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb


DATE_COL = "date"
PRICE_COL_DEFAULT = "期货收盘价(活跃合约):阴极铜"


# ===== 可在此处直接修改默认运行配置（无需命令行） =====
USER_CONFIG: Dict[str, object] = {
    "data_path": os.path.join(os.path.dirname(__file__), "data", "CU", "以收盘价为label_day_next.csv"),
    "output_dir": os.path.dirname(__file__),
    # 目标列（CSV 中已准备好的目标来源列）
    "y_col": PRICE_COL_DEFAULT,
    # 标签对齐：shift_y=1 表示 y=下一期该列的值；若已对齐则为 0
    "shift_y": 1,
    # 设备与调参
    "use_gpu": False,
    "tune": False,
    "n_trials": 200,
    # 模型与训练控制
    "booster": "gbtree",
    "objective": "reg:absoluteerror",
    "n_estimators": 20000,
    "early_stopping_rounds": 50,
    # 时间衰减样本权重（提升近期适配性）
    "use_time_decay": False,
    "half_life_days": 126,
}



@dataclass
class DateRange:
    start: str
    end: str

    def contains(self, series: pd.Series) -> pd.Series:
        return (series >= pd.to_datetime(self.start)) & (series <= pd.to_datetime(self.end))


SPLITS: Dict[str, DateRange] = {
    "train": DateRange("2018-01-01", "2022-06-30"),
    "valid": DateRange("2022-07-01", "2024-05-30"),
    "test": DateRange("2024-06-01", "2025-08-25"),
}


def ensure_dirs(output_dir: str) -> Dict[str, str]:
    paths = {
        "root": output_dir,
        "models": os.path.join(output_dir, "results/models"),
        "preds": os.path.join(output_dir, "results/predictions"),
        "artifacts": os.path.join(output_dir, "results/artifacts"),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if DATE_COL not in df.columns:
        raise ValueError(f"CSV 缺少日期列: {DATE_COL}")
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    return df


def build_label(df: pd.DataFrame, y_col: str, shift_y: int) -> pd.DataFrame:
    if y_col not in df.columns:
        available = ", ".join(df.columns)
        raise ValueError(f"找不到目标列: {y_col}. 可用列: {available}")
    df = df.copy()
    shift_n = int(shift_y)
    if shift_n < 0:
        shift_n = 0
    df["y"] = df[y_col].shift(-shift_n)
    if shift_n > 0:
        df = df.iloc[:-shift_n].reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
    return df


def select_features(df: pd.DataFrame, label_col: str = "y") -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    # 仅移除日期与标签列；不再出现 price_col 相关逻辑
    feature_cols = [c for c in df.columns if c not in [DATE_COL, label_col]]
    X = df[feature_cols]
    y = df[label_col]
    return X, y, feature_cols


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    s_true = np.sign(y_true)
    s_pred = np.sign(y_pred)
    return float(np.mean(s_true == s_pred))


def _directional_trade_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float | None]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.size == 0:
        return {
            "volatility_correct": 0.0,
            "volatility_incorrect": 0.0,
            "theoretical_profit": 0.0,
            "profit_loss_ratio": None,
            "long_signal_ratio": 0.0,
            "short_signal_ratio": 0.0,
        }
    s_true = np.sign(y_true)
    s_pred = np.sign(y_pred)
    correct_mask = s_true == s_pred
    incorrect_mask = ~correct_mask
    vol_correct = float(np.sum(np.abs(y_true[correct_mask])))
    vol_incorrect = float(np.sum(np.abs(y_true[incorrect_mask])))
    theoretical_profit = float(vol_correct - vol_incorrect)
    pl_ratio: float | None
    if vol_incorrect == 0.0:
        pl_ratio = None
    else:
        pl_ratio = float(vol_correct / vol_incorrect)
    long_ratio = float(np.mean(y_pred > 0))
    short_ratio = float(np.mean(y_pred < 0))
    return {
        "volatility_correct": vol_correct,
        "volatility_incorrect": vol_incorrect,
        "theoretical_profit": theoretical_profit,
        "profit_loss_ratio": pl_ratio,
        "long_signal_ratio": long_ratio,
        "short_signal_ratio": short_ratio,
    }


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float | None]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    da = directional_accuracy(y_true, y_pred)
    extra = _directional_trade_metrics(y_true, y_pred)
    out: Dict[str, float | None] = {"rmse": rmse, "mae": mae, "r2": r2, "directional_accuracy": da}
    out.update(extra)
    return out


def _sign_with_threshold(y_score: np.ndarray, tau: float = 0.0, band: float = 0.0) -> np.ndarray:
    z = np.asarray(y_score, dtype=float) - float(tau)
    s = np.zeros_like(z, dtype=float)
    if float(band) > 0.0:
        s[z > band] = 1.0
        s[z < -band] = -1.0
    else:
        s[z > 0] = 1.0
        s[z < 0] = -1.0
    return s


def calibrate_threshold_from_predictions(y_score: np.ndarray, neutral_band_q: float = 0.2) -> Tuple[float, float]:
    y_score = np.asarray(y_score, dtype=float)
    tau = float(np.median(y_score))
    if 0.0 < float(neutral_band_q) < 1.0:
        band = float(np.quantile(np.abs(y_score - tau), float(neutral_band_q)))
    else:
        band = 0.0
    return tau, band


def evaluate_directional_only(y_true: np.ndarray, y_score: np.ndarray, tau: float = 0.0, band: float = 0.0) -> Dict[str, float | None]:
    y_true = np.asarray(y_true, dtype=float)
    s_pred = _sign_with_threshold(np.asarray(y_score, dtype=float), tau, band)
    s_true = np.sign(y_true)
    da = float(np.mean(s_true == s_pred))
    extra = _directional_trade_metrics(y_true, s_pred)
    out: Dict[str, float | None] = {"directional_accuracy": da}
    out.update(extra)
    return out


def make_default_xgb_params(use_gpu: bool = False) -> Dict[str, object]:
    # 与 xgboost.train 一致的参数命名
    params: Dict[str, object] = dict(
        learning_rate=0.01,
        max_depth=3,
        subsample=0.7,
        colsample_bytree=0.6,
        min_child_weight=50.0,
        gamma=5.0,
        reg_alpha=0.1,
        reg_lambda=10.0,
        # 使用 y 的中位数作为初始预测水平更稳健（训练前将被覆盖）
        base_score=0.0,
        objective=str(USER_CONFIG.get("objective", "reg:squarederror")),
        booster=str(USER_CONFIG.get("booster", "gbtree")),
        tree_method="hist",
        device=("cuda" if use_gpu else "cpu"),
        verbosity=0,
        seed=42,
    )
    obj = params["objective"]
    params["eval_metric"] = "mae" if obj == "reg:absoluteerror" else "rmse"
    if params["booster"] == "dart":
        params.setdefault("rate_drop", 0.1)
        params.setdefault("skip_drop", 0.0)
        params.setdefault("sample_type", "uniform")
        params.setdefault("normalize_type", "tree")
    return params


def train_xgb_with_early_stopping(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    params: Dict[str, object],
    sample_weight: Optional[np.ndarray] = None,
    log_training: bool = False,
) -> xgb.Booster:
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weight)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)
    num_boost_round = int(params.get("n_estimators", USER_CONFIG.get("n_estimators", 2000)))
    patience = int(USER_CONFIG.get("early_stopping_rounds", 100))
    evals = [(dtrain, "train"), (dvalid, "valid")]
    callbacks = [xgb.callback.EarlyStopping(rounds=patience, save_best=True)]
    booster: xgb.Booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=evals,
        callbacks=callbacks,
        verbose_eval=log_training,
    )
    if not hasattr(booster, "best_iteration") or booster.best_iteration is None:
        raise RuntimeError("未检测到早停的 best_iteration。")
    return booster


def infer_best_n_estimators(booster: xgb.Booster) -> int:
    best_it = getattr(booster, "best_iteration", None)
    if best_it is not None:
        return int(best_it) + 1
    try:
        return int(booster.num_boosted_rounds())
    except Exception:
        return int(USER_CONFIG.get("n_estimators", 0))


def extract_best_validation_metric(booster: xgb.Booster, eval_metric_name: str) -> tuple[str, float] | None:
    best_score = getattr(booster, "best_score", None)
    if best_score is not None:
        try:
            return (str(eval_metric_name), float(best_score))
        except Exception:
            return (str(eval_metric_name), float(best_score))
    return None


def refit_final_model(
    best_booster: xgb.Booster,
    X_train_valid: np.ndarray,
    y_train_valid: np.ndarray,
    params: Dict[str, object],
    sample_weight: Optional[np.ndarray] = None,
) -> xgb.Booster:
    best_n_estimators = infer_best_n_estimators(best_booster)
    dtrval = xgb.DMatrix(X_train_valid, label=y_train_valid, weight=sample_weight)
    booster = xgb.train(
        params=params,
        dtrain=dtrval,
        num_boost_round=int(best_n_estimators),
        evals=[(dtrval, "trval")],
        verbose_eval=False,
    )
    return booster


def _time_decay_weights(length: int, half_life: float) -> np.ndarray:
    idx = np.arange(length)
    # 最近样本权重大（指数递增），使用相对时间，从最早到最新递增
    decay = np.exp(-np.log(2) * (length - 1 - idx) / half_life)
    return decay


def maybe_import_optuna():
    try:
        import optuna  # type: ignore

        return optuna
    except Exception:
        return None


def _build_year_folds(train_dates: pd.Series, cv_folds: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    years = sorted(train_dates.dt.year.unique())
    if cv_folds <= 0:
        return []
    cv_folds = min(cv_folds, len(years))
    selected_years = years[-cv_folds:]
    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    for y in selected_years:
        train_mask = train_dates.dt.year < y
        val_mask = train_dates.dt.year == y
        tr_idx = np.where(train_mask.values)[0]
        va_idx = np.where(val_mask.values)[0]
        if len(tr_idx) == 0 or len(va_idx) == 0:
            continue
        folds.append((tr_idx, va_idx))
    return folds


def tune_hyperparams_optuna(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    use_gpu: bool = False,
    n_trials: int = 30,
    train_dates: Optional[pd.Series] = None,
    cv_folds: int = 0,
) -> Dict[str, object]:
    optuna = maybe_import_optuna()
    if optuna is None:
        print("未检测到 optuna，跳过调参。可通过 pip install optuna 启用。")
        return make_default_xgb_params(use_gpu)

    def objective(trial):
        params = make_default_xgb_params(use_gpu).copy()
        # 扩大搜索空间，并纳入 n_estimators 与 booster（含 dart）
        booster = trial.suggest_categorical("booster", ["gbtree"])
        params.update(
            dict(
                booster=booster,
                n_estimators=trial.suggest_int("n_estimators", 1000, 6000, step=250),
                max_depth=trial.suggest_int("max_depth", 2, 10),
                min_child_weight=trial.suggest_float("min_child_weight", 1.0, 100.0, log=True),
                gamma=trial.suggest_float("gamma", 0.0, 12.0),
                subsample=trial.suggest_float("subsample", 0.5, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
                reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 20.0, log=True),
                reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 100.0, log=True),
                learning_rate=trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            )
        )
        if booster == "dart":
            params.update(
                dict(
                    rate_drop=trial.suggest_float("rate_drop", 0.0, 0.3),
                    skip_drop=trial.suggest_float("skip_drop", 0.0, 0.5),
                )
            )
        tune_target = str(USER_CONFIG.get("tune_target", "rmse"))
        # 若配置了时间序列 CV，则在训练集内部做滚动评估
        folds = _build_year_folds(train_dates, cv_folds) if train_dates is not None else []
        def _score_rmse(y_va: np.ndarray, p_va: np.ndarray) -> float:
            return float(np.sqrt(mean_squared_error(y_va, p_va)))
        def _score_da_or_profit(X_tr: np.ndarray, y_tr: np.ndarray, X_va: np.ndarray, y_va: np.ndarray, booster: xgb.Booster) -> float:
            p_tr = booster.predict(xgb.DMatrix(X_tr))
            p_va = booster.predict(xgb.DMatrix(X_va))
            q = 0.2
            tau, band = calibrate_threshold_from_predictions(p_tr, q)
            m = evaluate_directional_only(y_va, p_va, tau, band)
            if tune_target == "da":
                return 1.0 - float(m["directional_accuracy"])  # 最小化(1-DA)
            else:
                return -float(m["theoretical_profit"])          # 最小化(-profit)
        if folds:
            scores: List[float] = []
            for tr_idx, va_idx in folds:
                X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
                X_va, y_va = X_train[va_idx], y_train[va_idx]
                if bool(USER_CONFIG.get("use_time_decay", False)):
                    half_life = float(USER_CONFIG.get("half_life_days", 252))
                    sw = _time_decay_weights(len(y_tr), half_life)
                else:
                    sw = None
                booster_cv = train_xgb_with_early_stopping(
                    X_tr, y_tr, X_va, y_va, params, sample_weight=sw
                )
                if tune_target == "rmse":
                    scores.append(_score_rmse(y_va, booster_cv.predict(xgb.DMatrix(X_va))))
                else:
                    scores.append(_score_da_or_profit(X_tr, y_tr, X_va, y_va, booster_cv))
            return float(np.mean(scores)) if scores else 1e9
        else:
            # 否则退化为使用外部 valid
            if bool(USER_CONFIG.get("use_time_decay", False)):
                half_life = float(USER_CONFIG.get("half_life_days", 252))
                sw = _time_decay_weights(len(y_train), half_life)
            else:
                sw = None
            booster = train_xgb_with_early_stopping(
                X_train, y_train, X_valid, y_valid, params, sample_weight=sw
            )
            if tune_target == "rmse":
                pred_valid = booster.predict(xgb.DMatrix(X_valid))
                return _score_rmse(y_valid, pred_valid)
            else:
                return _score_da_or_profit(X_train, y_train, X_valid, y_valid, booster)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_params = make_default_xgb_params(use_gpu)
    best_params.update(study.best_params)
    return best_params


def save_predictions(
    path: str,
    dates: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    split: str,
) -> None:
    out = pd.DataFrame({
        DATE_COL: pd.to_datetime(dates).dt.strftime("%Y-%m-%d"),
        "y_true": y_true,
        "y_pred": y_pred,
        "split": split,
    })
    out.to_csv(path, index=False, encoding="utf-8-sig")


def save_feature_importance(path: str, booster: xgb.Booster, feature_names: List[str]) -> None:
    score = booster.get_score(importance_type="gain")
    importance = np.zeros(len(feature_names), dtype=float)
    for k, v in score.items():
        if k.startswith("f") and k[1:].isdigit():
            idx = int(k[1:])
            if 0 <= idx < len(feature_names):
                importance[idx] = float(v)
    df_imp = pd.DataFrame({"feature": feature_names, "importance_gain": importance})
    df_imp.sort_values(df_imp.columns[-1], ascending=False, inplace=True)
    df_imp.to_csv(path, index=False, encoding="utf-8-sig")


def run(
    data_path: str,
    output_dir: str,
    y_col: str,
    shift_y: int = 0,
    use_gpu: bool = False,
    tune: bool = False,
    n_trials: int = 30,
) -> None:
    paths = ensure_dirs(output_dir)

    # 1) 读取数据，构造标签（不再依赖 price_col/label_type）
    df_raw = load_data(data_path)
    df = build_label(df_raw, y_col, shift_y)

    # 2) 日期切分
    mask_train = SPLITS["train"].contains(df[DATE_COL])
    mask_valid = SPLITS["valid"].contains(df[DATE_COL])
    mask_test = SPLITS["test"].contains(df[DATE_COL])

    df_train, df_valid, df_test = df[mask_train], df[mask_valid], df[mask_test]
    if len(df_train) == 0 or len(df_valid) == 0 or len(df_test) == 0:
        raise ValueError("切分后某个数据集为空，请确认日期区间与数据覆盖范围。")

    # 3) 特征与标签
    X_train, y_train, feature_cols = select_features(df_train)
    X_valid, y_valid, _ = select_features(df_valid)
    X_test, y_test, _ = select_features(df_test)

    # 4) 缺失值填充（仅基于训练集拟合）
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_valid_imp = imputer.transform(X_valid)
    X_test_imp = imputer.transform(X_test)

    # 5) 获取参数（可选调参）并训练（早停基于验证集）
    if tune:
        params = tune_hyperparams_optuna(
            X_train_imp,
            y_train.values,
            X_valid_imp,
            y_valid.values,
            use_gpu=use_gpu,
            n_trials=n_trials,
            train_dates=df_train[DATE_COL],
            cv_folds=int(USER_CONFIG.get("cv_folds", 0)),
        )
    else:
        params = make_default_xgb_params(use_gpu)

    # 将 base_score 设置为训练标签的中位数，可缓解第一轮偏置带来的度量不稳
    try:
        params["base_score"] = float(np.median(y_train.values.astype(float)))
    except Exception:
        pass

    # 可选时间衰减权重
    if bool(USER_CONFIG.get("use_time_decay", False)):
        half_life = float(USER_CONFIG.get("half_life_days", 252))
        sw_train = _time_decay_weights(len(y_train), half_life)
        sw_trval = _time_decay_weights(len(y_train) + len(y_valid), half_life)
    else:
        sw_train = None
        sw_trval = None

    booster_es = train_xgb_with_early_stopping(
        X_train_imp, y_train.values, X_valid_imp, y_valid.values, params,
        sample_weight=sw_train,
        log_training=True,
    )

    # 6) 复训（训练+验证）并在测试集上推断
    X_trval_imp = np.vstack([X_train_imp, X_valid_imp])
    y_trval = np.concatenate([y_train.values, y_valid.values])
    booster_final = refit_final_model(booster_es, X_trval_imp, y_trval, params, sample_weight=sw_trval)

    # 7) 预测
    pred_train = booster_es.predict(xgb.DMatrix(X_train_imp))
    pred_valid = booster_es.predict(xgb.DMatrix(X_valid_imp))
    pred_test = booster_final.predict(xgb.DMatrix(X_test_imp))

    # 8) 评估
    # 附加早停与验证最优度量信息，便于确认是否触发
    best_n = infer_best_n_estimators(booster_es)
    best_metric = extract_best_validation_metric(booster_es, params.get("eval_metric", "mae"))
    # 训练细节：用于判断是否触发早停与原因
    metric_name = params.get("eval_metric", "mae")
    n_rounds_set = int(params.get("n_estimators", USER_CONFIG.get("n_estimators", 0)))
    patience_used = int(USER_CONFIG.get("early_stopping_rounds", 100))
    stopped_early = int(best_n) < int(n_rounds_set if n_rounds_set > 0 else best_n)
    no_improve_rounds_after_best = (n_rounds_set - 1 - (int(best_n) - 1)) if n_rounds_set > 0 else None
    metrics = {
        "train": evaluate(y_train.values, pred_train),
        "valid": evaluate(y_valid.values, pred_valid),
        "test": evaluate(y_test.values, pred_test),
        "best_n_estimators": int(best_n),
        "final_n_estimators": int(best_n),
        "best_valid_metric": {best_metric[0]: best_metric[1]} if best_metric else None,
        "used_params": params,
        "training_info": {
            "eval_metric": metric_name,
            "patience": patience_used,
            "n_estimators_set": int(n_rounds_set if n_rounds_set > 0 else best_n),
            "best_iteration": int(best_n) - 1,
            "num_rounds_recorded": int(n_rounds_set),
            "stopped_early": bool(stopped_early),
            "no_improve_rounds_after_best": int(no_improve_rounds_after_best) if no_improve_rounds_after_best is not None else None,
        },
    }

    # 8.1) 保留基础指标即可

    # 9) 保存预测结果
    save_predictions(
        os.path.join(paths["preds"], "train.csv"), df_train[DATE_COL], y_train.values, pred_train, "train"
    )
    save_predictions(
        os.path.join(paths["preds"], "valid.csv"), df_valid[DATE_COL], y_valid.values, pred_valid, "valid"
    )
    save_predictions(
        os.path.join(paths["preds"], "test.csv"), df_test[DATE_COL], y_test.values, pred_test, "test"
    )

    # 10) 保存模型与要素
    booster_es.save_model(os.path.join(paths["models"], "xgb_model_es.json"))
    booster_final.save_model(os.path.join(paths["models"], "xgb_model_final.json"))
    # 保存填充器与特征名
    pd.Series(feature_cols).to_csv(
        os.path.join(paths["artifacts"], "feature_names.csv"), index=False, header=["feature"], encoding="utf-8-sig"
    )
    # 简易保存 imputer（以训练集中位数形式）
    medians = pd.Series(imputer.statistics_, index=feature_cols)
    medians.to_csv(os.path.join(paths["artifacts"], "feature_medians.csv"), encoding="utf-8-sig")

    # 11) 特征重要性
    save_feature_importance(
        os.path.join(paths["artifacts"], "feature_importance.csv"), booster_final, feature_cols
    )

    # 12) 保存度量
    with open(os.path.join(paths["artifacts"], "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # 控制台输出简要指标
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    # 13) 精简：不再输出价格明细日志，避免与 y 源列发生依赖


if __name__ == "__main__":
    cfg = USER_CONFIG
    run(
        data_path=str(cfg["data_path"]),
        output_dir=str(cfg["output_dir"]),
        y_col=str(cfg["y_col"]),
        shift_y=int(cfg.get("shift_y", 0)),
        use_gpu=bool(cfg.get("use_gpu", False)),
        tune=bool(cfg.get("tune", False)),
        n_trials=int(cfg.get("n_trials", 30)),
       )


