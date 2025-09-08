import os
import json
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor


DATE_COL = "date"
PRICE_COL_DEFAULT = "期货收盘价(活跃合约):阴极铜"


# ===== 可在此处直接修改默认运行配置（无需命令行） =====
USER_CONFIG: Dict[str, object] = {
    "data_path": os.path.join(os.path.dirname(__file__), "data", "CU", "以收盘价为label_day_next.csv"),
    "output_dir": os.path.dirname(__file__),
    "y_col": PRICE_COL_DEFAULT,
    "shift_y": 1,
    # 起始使用数据的最早日期（含）。为空表示不限制，从最早可用数据开始。
    "start_date": "2018-01-01",
    # 训练贴近 cut_date（当日为验证窗右端），并用于明日预测
    "cut_date": "2025-08-25",
    "valid_days": 90,
    "embargo_days": 1,
    # 训练细节
    "use_time_decay": False,
    "half_life_days": 126.0,
    "use_gpu": True,
    "early_stopping_rounds": 50,
    # 可选：从调参脚本导出的最终参数 JSON（如无则使用内置默认）
    "params_json": "",
}


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
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    da = directional_accuracy(y_true, y_pred)
    extra = _directional_trade_metrics(y_true, y_pred)
    out: Dict[str, float | None] = {"rmse": rmse, "mae": mae, "r2": r2, "directional_accuracy": da}
    out.update(extra)
    return out


def make_default_xgb_params(use_gpu: bool = False) -> Dict[str, object]:
    params: Dict[str, object] = dict(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.7,
        colsample_bytree=0.6,
        min_child_weight=10.0,
        gamma=1.0,
        reg_alpha=0.1,
        reg_lambda=2.0,
        objective="reg:absoluteerror",
        booster="gbtree",
        n_jobs=-1,
        random_state=42,
        tree_method="hist",
        verbosity=0,
        eval_metric="mae",
    )
    if use_gpu:
        params["device"] = "cuda"
    return params


def train_xgb_with_early_stopping(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    params: Dict[str, object],
    sample_weight: Optional[np.ndarray] = None,
    early_stopping_rounds: int = 50,
    log_training: bool = False,
) -> XGBRegressor:
    try:
        model = XGBRegressor(**params)
    except TypeError:
        fallback = params.copy()
        fallback.pop("device", None)
        fallback["tree_method"] = "gpu_hist"
        fallback.setdefault("predictor", "gpu_predictor")
        model = XGBRegressor(**fallback)
    try:
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            early_stopping_rounds=early_stopping_rounds,
            sample_weight=sample_weight,
            verbose=log_training,
        )
    except TypeError:
        try:
            import xgboost as xgb

            early_stop_cb = xgb.callback.EarlyStopping(rounds=early_stopping_rounds, save_best=True)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_valid, y_valid)],
                callbacks=[early_stop_cb],
                sample_weight=sample_weight,
                verbose=log_training,
            )
        except Exception:
            if sample_weight is not None:
                model.fit(X_train, y_train, sample_weight=sample_weight)
            else:
                model.fit(X_train, y_train)
    return model


def infer_best_n_estimators(model: XGBRegressor) -> int:
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
    try:
        booster = model.get_booster()
        if hasattr(booster, "best_iteration") and booster.best_iteration is not None:
            return int(booster.best_iteration) + 1
        if hasattr(booster, "best_ntree_limit") and booster.best_ntree_limit is not None:
            return int(booster.best_ntree_limit)
    except Exception:
        pass
    try:
        er = model.evals_result()
        keys = list(er.keys())
        for key in ["validation_0", "eval", "valid"]:
            if key in er:
                metrics = er[key]
                if "rmse" in metrics:
                    values = metrics["rmse"]
                else:
                    mname = list(metrics.keys())[0]
                    values = metrics[mname]
                if isinstance(values, list) and len(values) > 0:
                    best_idx = int(np.argmin(values))
                    return best_idx + 1
        if keys:
            first_key = keys[0]
            metrics = er[first_key]
            mname = list(metrics.keys())[0]
            values = metrics[mname]
            best_idx = int(np.argmin(values))
            return best_idx + 1
    except Exception:
        pass
    return int(model.get_params().get("n_estimators", 100))


def refit_final_model(
    best_model: XGBRegressor,
    X_train_valid: np.ndarray,
    y_train_valid: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> XGBRegressor:
    best_n_estimators = infer_best_n_estimators(best_model)
    base_params = best_model.get_params()
    base_params["n_estimators"] = int(best_n_estimators)
    final_model = XGBRegressor(**base_params)
    if sample_weight is not None:
        final_model.fit(X_train_valid, y_train_valid, sample_weight=sample_weight)
    else:
        final_model.fit(X_train_valid, y_train_valid)
    return final_model


def _time_decay_weights(length: int, half_life: float) -> np.ndarray:
    idx = np.arange(length)
    decay = np.exp(-np.log(2) * (length - 1 - idx) / half_life)
    return decay


def build_dynamic_masks(dates: pd.Series, cut_date: str, valid_days: int, embargo_days: int = 1):
    cut = pd.to_datetime(cut_date)
    valid_start = cut - pd.Timedelta(days=valid_days)
    train_end = valid_start - pd.Timedelta(days=embargo_days)
    mask_train = dates < train_end
    mask_valid = (dates >= valid_start) & (dates < cut)
    return mask_train, mask_valid


def run_refit(
    data_path: str,
    output_dir: str,
    y_col: str,
    shift_y: int,
    start_date: str,
    cut_date: str,
    valid_days: int,
    embargo_days: int,
    use_time_decay: bool,
    half_life_days: float,
    use_gpu: bool,
    early_stopping_rounds: int,
    params_json: str = "",
) -> None:
    df_raw = load_data(data_path)
    # 可选：按 start_date 过滤数据
    if start_date:
        sd = pd.to_datetime(start_date)
        df_raw = df_raw[df_raw[DATE_COL] >= sd].reset_index(drop=True)

    df = build_label(df_raw, y_col, shift_y)

    m_tr, m_va = build_dynamic_masks(df[DATE_COL], cut_date, valid_days, embargo_days=embargo_days)
    df_tr, df_va = df[m_tr], df[m_va]
    if len(df_tr) == 0 or len(df_va) == 0:
        raise ValueError("训练/验证集为空，请检查 cut_date, valid_days 与 embargo_days。")

    X_tr, y_tr, feature_cols = select_features(df_tr)
    X_va, y_va, _ = select_features(df_va)

    imputer = SimpleImputer(strategy="median")
    X_tr_imp = imputer.fit_transform(X_tr)
    X_va_imp = imputer.transform(X_va)

    if params_json and os.path.isfile(params_json):
        with open(params_json, "r", encoding="utf-8") as f:
            params = json.load(f)
    else:
        params = make_default_xgb_params(use_gpu=bool(use_gpu))

    sw_tr = _time_decay_weights(len(y_tr), half_life_days) if use_time_decay else None
    sw_trva = _time_decay_weights(len(y_tr) + len(y_va), half_life_days) if use_time_decay else None

    model_es = train_xgb_with_early_stopping(
        X_tr_imp, y_tr.values, X_va_imp, y_va.values, params,
        sample_weight=sw_tr, early_stopping_rounds=int(early_stopping_rounds), log_training=True,
    )

    X_trva_imp = np.vstack([X_tr_imp, X_va_imp])
    y_trva = np.concatenate([y_tr.values, y_va.values])
    model_final = refit_final_model(model_es, X_trva_imp, y_trva, sample_weight=sw_trva)

    # 将训练输出与回测输出分离：写入 results_deploy/cut_<cut_date>/...
    deploy_root = os.path.join(output_dir, "results_deploy", f"cut_{cut_date}")
    models_dir = os.path.join(deploy_root, "models")
    preds_dir = os.path.join(deploy_root, "predictions")
    arts_dir = os.path.join(deploy_root, "artifacts")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(preds_dir, exist_ok=True)
    os.makedirs(arts_dir, exist_ok=True)

    # 预测：训练/验证
    pred_train = model_es.predict(X_tr_imp)
    pred_valid = model_es.predict(X_va_imp)

    # 评估：训练/验证
    metrics = {
        "train": evaluate(y_tr.values, pred_train),
        "valid": evaluate(y_va.values, pred_valid),
        "used_params": params,
    }

    # 保存预测 CSV
    out_tr = pd.DataFrame({
        DATE_COL: pd.to_datetime(df_tr[DATE_COL]).dt.strftime("%Y-%m-%d"),
        "y_true": y_tr.values,
        "y_pred": pred_train,
        "split": "train",
    })
    out_tr.to_csv(os.path.join(preds_dir, "train.csv"), index=False, encoding="utf-8-sig")

    out_va = pd.DataFrame({
        DATE_COL: pd.to_datetime(df_va[DATE_COL]).dt.strftime("%Y-%m-%d"),
        "y_true": y_va.values,
        "y_pred": pred_valid,
        "split": "valid",
    })
    out_va.to_csv(os.path.join(preds_dir, "valid.csv"), index=False, encoding="utf-8-sig")

    model_es.save_model(os.path.join(models_dir, "xgb_model_es.json"))
    model_final.save_model(os.path.join(models_dir, "xgb_model_final.json"))
    pd.Series(feature_cols).to_csv(
        os.path.join(arts_dir, "feature_names.csv"), index=False, header=["feature"], encoding="utf-8-sig"
    )
    medians = pd.Series(data=imputer.statistics_, index=feature_cols)
    medians.to_csv(os.path.join(arts_dir, "feature_medians.csv"), encoding="utf-8-sig")

    # 保存度量
    with open(os.path.join(arts_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    meta = dict(
        cut_date=cut_date,
        valid_days=int(valid_days),
        embargo_days=int(embargo_days),
        use_time_decay=bool(use_time_decay),
        half_life_days=float(half_life_days),
        used_params=params,
        train_size=int(len(df_tr)),
        valid_size=int(len(df_va)),
        output_root=deploy_root,
    )
    with open(os.path.join(arts_dir, "train_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(json.dumps({
        "status": "ok",
        "train_size": int(len(df_tr)),
        "valid_size": int(len(df_va)),
        "train_metrics": metrics["train"],
        "valid_metrics": metrics["valid"],
        "deploy_root": deploy_root,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    cfg = USER_CONFIG
    run_refit(
        data_path=str(cfg["data_path"]),
        output_dir=str(cfg["output_dir"]),
        y_col=str(cfg.get("y_col", PRICE_COL_DEFAULT)),
        shift_y=int(cfg.get("shift_y", 1)),
        start_date=str(cfg.get("start_date", "")),
        cut_date=str(cfg["cut_date"]),
        valid_days=int(cfg.get("valid_days", 63)),
        embargo_days=int(cfg.get("embargo_days", 1)),
        use_time_decay=bool(cfg.get("use_time_decay", True)),
        half_life_days=float(cfg.get("half_life_days", 126.0)),
        use_gpu=bool(cfg.get("use_gpu", False)),
        early_stopping_rounds=int(cfg.get("early_stopping_rounds", 50)),
        params_json=str(cfg.get("params_json", "")),
    )


