import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


DATE_COL = "date"
PRICE_COL_DEFAULT = "期货收盘价(活跃合约):精对苯二甲酸(PTA)"


# ===== 可在此处直接修改默认运行配置（无需命令行） =====
USER_CONFIG: Dict[str, object] = {
    "data_path": os.path.join(os.path.dirname(__file__), "data", "PTA/PTA收盘价label.csv"),
    "output_dir": os.path.dirname(__file__),
    "price_col": PRICE_COL_DEFAULT,
    # 设备与调参
    "use_gpu": True,
    "tune": False,
    "n_trials": 200,
    # 模型与训练控制（不改数据，仅从模型侧优化）
    "booster": "gbtree",               # 可先用 gbtree；若仍过拟合可试 "dart"
    "objective": "reg:absoluteerror",  # 稳健损失，降低极端值影响
    "n_estimators": 1500,               # 限制上限，让早停更容易触发
    "early_stopping_rounds": 50,        # 更积极的早停
    # 训练期内的时间序列交叉验证
    "cv_folds": 0,                      # 以 2019/2020/2021 为滚动验证
    # 时间衰减样本权重（提升近期适配性）
    "use_time_decay": True,
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


def build_label(df: pd.DataFrame, price_col: str) -> pd.DataFrame:
    if price_col not in df.columns:
        available = ", ".join(df.columns)
        raise ValueError(f"找不到价格列: {price_col}. 可用列: {available}")
    df = df.copy()
    df["y"] = df[price_col].shift(-1) - df[price_col]
    df = df.iloc[:-1].reset_index(drop=True)
    return df


def select_features(df: pd.DataFrame, label_col: str = "y") -> Tuple[pd.DataFrame, pd.Series, List[str]]:
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


def make_default_xgb_params(use_gpu: bool = False) -> Dict[str, object]:
    # 更强正则 + 更浅的树，降低过拟合风险
    params: Dict[str, object] = dict(
        n_estimators=int(USER_CONFIG.get("n_estimators", 2000)),
        learning_rate=0.03,
        max_depth=4,
        subsample=0.7,
        colsample_bytree=0.6,
        min_child_weight=10.0,
        gamma=1.0,
        reg_alpha=0.1,
        reg_lambda=2.0,
        objective=str(USER_CONFIG.get("objective", "reg:squarederror")),
        booster=str(USER_CONFIG.get("booster", "gbtree")),
        n_jobs=-1,
        random_state=42,
        tree_method="hist",   # 统一 hist，GPU 由 device 控制（xgboost>=2 推荐）
        verbosity=0,
    )
    # 评估指标随目标自动选择
    obj = params["objective"]
    if obj == "reg:absoluteerror":
        params["eval_metric"] = "mae"
    else:
        params["eval_metric"] = "rmse"
    # dart 轻量默认
    if params["booster"] == "dart":
        params.setdefault("rate_drop", 0.1)
        params.setdefault("skip_drop", 0.0)
        params.setdefault("sample_type", "uniform")
        params.setdefault("normalize_type", "tree")
    if use_gpu:
        params["device"] = "cuda"  # xgboost>=2.0 推荐写法
    return params


def train_xgb_with_early_stopping(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    params: Dict[str, object],
    sample_weight: Optional[np.ndarray] = None,
    log_training: bool = False,
) -> XGBRegressor:
    # 兼容不同 xgboost 版本：优先使用 device=cuda + tree_method=hist
    # 若旧版不支持 device 参数，则回退为 gpu_hist + gpu_predictor
    try:
        model = XGBRegressor(**params)
    except TypeError:
        fallback = params.copy()
        fallback.pop("device", None)
        fallback["tree_method"] = "gpu_hist"
        fallback.setdefault("predictor", "gpu_predictor")
        model = XGBRegressor(**fallback)
    patience = int(USER_CONFIG.get("early_stopping_rounds", 100))
    try:
        # 新版 API：fit 支持 early_stopping_rounds（常见于 1.x 与 2.x）
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            early_stopping_rounds=patience,
            sample_weight=sample_weight,
            verbose=log_training,
        )
    except TypeError:
        # 兼容路径：使用 callbacks（2.x 支持），或最终退化为无早停训练
        try:
            import xgboost as xgb  # 延迟导入，避免版本差异

            early_stop_cb = xgb.callback.EarlyStopping(rounds=patience, save_best=True)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_valid, y_valid)],
                callbacks=[early_stop_cb],
                sample_weight=sample_weight,
                verbose=log_training,
            )
        except Exception:
            # 最保守回退：不使用早停
            if sample_weight is not None:
                model.fit(X_train, y_train, sample_weight=sample_weight)
            else:
                model.fit(X_train, y_train)
    return model


def infer_best_n_estimators(model: XGBRegressor) -> int:
    # 1) 直接属性
    best_it = getattr(model, "best_iteration", None)
    if best_it is not None:
        try:
            return int(best_it) + 1
        except Exception:
            pass
    best_limit = getattr(model, "best_ntree_limit", None)
    if best_limit is not None:
        try:
            return int(best_limit)
        except Exception:
            pass
    # 2) Booster 属性
    try:
        booster = model.get_booster()
        if hasattr(booster, "best_iteration") and booster.best_iteration is not None:
            return int(booster.best_iteration) + 1
        if hasattr(booster, "best_ntree_limit") and booster.best_ntree_limit is not None:
            return int(booster.best_ntree_limit)
    except Exception:
        pass
    # 3) 从 evals_result 里解析（兼容 evals_result 或 evals_result_；validation_0 的 rmse/mae 最小点）
    try:
        er_m = getattr(model, "evals_result", None)
        er = er_m() if callable(er_m) else getattr(model, "evals_result_", None)
        # 常见 key：'validation_0' 或 'eval'
        if er:
            keys = list(er.keys())
            for key in ["validation_0", "eval", "valid"]:
                if key in er:
                    metrics = er[key]
                    metric_name = "rmse" if "rmse" in metrics else ("mae" if "mae" in metrics else list(metrics.keys())[0])
                    values = metrics[metric_name]
                    if isinstance(values, list) and len(values) > 0:
                        best_idx = int(np.argmin(values))
                        return best_idx + 1
            if keys:
                first_key = keys[0]
                metrics = er[first_key]
                metric_name = list(metrics.keys())[0]
                values = metrics[metric_name]
                best_idx = int(np.argmin(values))
                return best_idx + 1
    except Exception:
        pass
    # 4) 回退：使用设定的 n_estimators
    return int(model.get_params().get("n_estimators", 100))


def extract_best_validation_metric(model: XGBRegressor) -> tuple[str, float] | None:
    try:
        er_m = getattr(model, "evals_result", None)
        er = er_m() if callable(er_m) else getattr(model, "evals_result_", None)
        if not er:
            return None
        for key in ["validation_0", "eval", "valid"]:
            if key in er:
                metrics = er[key]
                if "rmse" in metrics:
                    values = metrics["rmse"];  return ("rmse", float(np.min(values)))
                if "mae" in metrics:
                    values = metrics["mae"];   return ("mae", float(np.min(values)))
                mname = list(metrics.keys())[0]
                values = metrics[mname];       return (mname, float(np.min(values)))
        first_key = list(er.keys())[0]
        metrics = er[first_key]
        mname = list(metrics.keys())[0]
        values = metrics[mname]
        return (mname, float(np.min(values)))
    except Exception:
        return None


def refit_final_model(
    best_model: XGBRegressor,
    X_train_valid: np.ndarray,
    y_train_valid: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> XGBRegressor:
    # 依据早停得到的最佳迭代数进行复训
    best_n_estimators = infer_best_n_estimators(best_model)
    base_params = best_model.get_params()
    base_params["n_estimators"] = int(best_n_estimators)
    # 去掉与训练时冲突的回调/早停状态
    final_model = XGBRegressor(**base_params)
    if sample_weight is not None:
        final_model.fit(X_train_valid, y_train_valid, sample_weight=sample_weight)
    else:
        final_model.fit(X_train_valid, y_train_valid)
    return final_model


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
        params.update(
            dict(
                max_depth=trial.suggest_int("max_depth", 3, 6),
                min_child_weight=trial.suggest_float("min_child_weight", 3.0, 50.0, log=True),
                gamma=trial.suggest_float("gamma", 0.0, 8.0),
                subsample=trial.suggest_float("subsample", 0.5, 0.85),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 0.85),
                reg_alpha=trial.suggest_float("reg_alpha", 1e-2, 10.0, log=True),
                reg_lambda=trial.suggest_float("reg_lambda", 0.5, 10.0, log=True),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            )
        )
        # 若配置了时间序列 CV，则在训练集内部做滚动评估
        folds = _build_year_folds(train_dates, cv_folds) if train_dates is not None else []
        if folds:
            rmses: List[float] = []
            for tr_idx, va_idx in folds:
                X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
                X_va, y_va = X_train[va_idx], y_train[va_idx]
                if bool(USER_CONFIG.get("use_time_decay", False)):
                    half_life = float(USER_CONFIG.get("half_life_days", 252))
                    sw = _time_decay_weights(len(y_tr), half_life)
                else:
                    sw = None
                mdl = train_xgb_with_early_stopping(
                    X_tr, y_tr, X_va, y_va, params, sample_weight=sw
                )
                p = mdl.predict(X_va)
                rmses.append(float(np.sqrt(mean_squared_error(y_va, p))))
            return float(np.mean(rmses)) if rmses else 1e9
        else:
            # 否则退化为使用外部 valid
            if bool(USER_CONFIG.get("use_time_decay", False)):
                half_life = float(USER_CONFIG.get("half_life_days", 252))
                sw = _time_decay_weights(len(y_train), half_life)
            else:
                sw = None
            model = train_xgb_with_early_stopping(
                X_train, y_train, X_valid, y_valid, params, sample_weight=sw
            )
            pred_valid = model.predict(X_valid)
            rmse = float(np.sqrt(mean_squared_error(y_valid, pred_valid)))
            return rmse

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


def save_feature_importance(path: str, model: XGBRegressor, feature_names: List[str]) -> None:
    try:
        # 基于 gain 的重要性更具解释性
        booster = model.get_booster()
        score = booster.get_score(importance_type="gain")
        # get_score 返回 dict，key 是 f{index}
        importance = np.zeros(len(feature_names), dtype=float)
        for k, v in score.items():
            if k.startswith("f") and k[1:].isdigit():
                idx = int(k[1:])
                if 0 <= idx < len(feature_names):
                    importance[idx] = float(v)
        df_imp = pd.DataFrame({"feature": feature_names, "importance_gain": importance})
    except Exception:
        # 回退到 sklearn 的属性
        importance = getattr(model, "feature_importances_", None)
        if importance is None:
            return
        df_imp = pd.DataFrame({"feature": feature_names, "importance_weight": importance})
    df_imp.sort_values(df_imp.columns[-1], ascending=False, inplace=True)
    df_imp.to_csv(path, index=False, encoding="utf-8-sig")


def run(
    data_path: str,
    output_dir: str,
    price_col: str,
    use_gpu: bool = False,
    tune: bool = False,
    n_trials: int = 30,
) -> None:
    paths = ensure_dirs(output_dir)

    # 1) 读取数据，构造标签
    df_raw = load_data(data_path)
    df = build_label(df_raw, price_col)

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

    # 可选时间衰减权重
    if bool(USER_CONFIG.get("use_time_decay", False)):
        half_life = float(USER_CONFIG.get("half_life_days", 252))
        sw_train = _time_decay_weights(len(y_train), half_life)
        sw_trval = _time_decay_weights(len(y_train) + len(y_valid), half_life)
    else:
        sw_train = None
        sw_trval = None

    model_es = train_xgb_with_early_stopping(
        X_train_imp, y_train.values, X_valid_imp, y_valid.values, params,
        sample_weight=sw_train,
        log_training=True,
    )

    # 6) 复训（训练+验证）并在测试集上推断
    X_trval_imp = np.vstack([X_train_imp, X_valid_imp])
    y_trval = np.concatenate([y_train.values, y_valid.values])
    model_final = refit_final_model(model_es, X_trval_imp, y_trval, sample_weight=sw_trval)

    # 7) 预测
    pred_train = model_es.predict(X_train_imp)
    pred_valid = model_es.predict(X_valid_imp)
    pred_test = model_final.predict(X_test_imp)

    # 8) 评估
    # 附加早停与验证最优度量信息，便于确认是否触发
    best_n = infer_best_n_estimators(model_es)
    best_metric = extract_best_validation_metric(model_es)
    metrics = {
        "train": evaluate(y_train.values, pred_train),
        "valid": evaluate(y_valid.values, pred_valid),
        "test": evaluate(y_test.values, pred_test),
        "best_n_estimators": int(best_n),
        "best_valid_metric": {best_metric[0]: best_metric[1]} if best_metric else None,
        "used_params": params,
    }

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
    model_es.save_model(os.path.join(paths["models"], "xgb_model_es.json"))
    model_final.save_model(os.path.join(paths["models"], "xgb_model_final.json"))
    # 保存填充器与特征名
    pd.Series(feature_cols).to_csv(
        os.path.join(paths["artifacts"], "feature_names.csv"), index=False, header=["feature"], encoding="utf-8-sig"
    )
    # 简易保存 imputer（以训练集中位数形式）
    medians = pd.Series(imputer.statistics_, index=feature_cols)
    medians.to_csv(os.path.join(paths["artifacts"], "feature_medians.csv"), encoding="utf-8-sig")

    # 11) 特征重要性
    save_feature_importance(
        os.path.join(paths["artifacts"], "feature_importance.csv"), model_final, feature_cols
    )

    # 12) 保存度量
    with open(os.path.join(paths["artifacts"], "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # 控制台输出简要指标
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    cfg = USER_CONFIG
    run(
        data_path=str(cfg["data_path"]),
        output_dir=str(cfg["output_dir"]),
        price_col=str(cfg["price_col"]),
        use_gpu=bool(cfg.get("use_gpu", False)),
        tune=bool(cfg.get("tune", False)),
        n_trials=int(cfg.get("n_trials", 30)),
       )


