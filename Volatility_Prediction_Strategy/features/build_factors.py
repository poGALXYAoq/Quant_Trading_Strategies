import os
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from joblib import Parallel, delayed

# 并行设置
NUM_WORKERS = int(os.getenv("VP_NUM_WORKERS", "30"))


def realized_vol(close: pd.Series, window: int, annualization_days: int = 252) -> pd.Series:
    ret = np.log(close).diff()
    rv = ret.rolling(window).std() * np.sqrt(annualization_days)
    return rv


def choose_atm_rows(df: pd.DataFrame, max_abs_moneyness_pct: float = 0.1) -> pd.DataFrame:
    moneyness = (df["spot"] - df["strike"]).abs() / df["spot"].replace(0, np.nan)
    mask = moneyness <= max_abs_moneyness_pct
    return df.loc[mask]


def interpolate_to_target_iv(df_on_date: pd.DataFrame, target_dte: int) -> float:
    if df_on_date.empty:
        return np.nan
    # group by expiration to compute mid IV around target; use nearest two expiries for interpolation
    dtes = df_on_date["dte"].dropna().values
    if len(dtes) == 0:
        return np.nan
    # pick closest and second closest
    idx_sorted = np.argsort(np.abs(dtes - target_dte))
    nearest = df_on_date.iloc[idx_sorted[0]]
    iv1 = nearest["iv"]
    dte1 = nearest["dte"]
    if len(idx_sorted) == 1:
        return iv1
    second = df_on_date.iloc[idx_sorted[1]]
    iv2 = second["iv"]
    dte2 = second["dte"]
    if pd.isna(iv1) and pd.isna(iv2):
        return np.nan
    if dte1 == dte2 or pd.isna(iv2):
        return iv1
    # linear interpolation by DTE
    w = (target_dte - dte2) / (dte1 - dte2)
    return float(w * iv1 + (1 - w) * iv2)


def build_factors(underlying_csv: str, options_derived_csv: str, output_csv: str,
                  target_dte: int = 30, secondary_dte: int = 60,
                  annualization_days: int = 252, max_abs_moneyness_pct: float = 0.1) -> None:
    spot = pd.read_csv(underlying_csv)
    spot["date"] = pd.to_datetime(spot["date"]).dt.date
    spot = spot.sort_values("date").reset_index(drop=True)

    opts = pd.read_csv(options_derived_csv)
    opts["trade_date"] = pd.to_datetime(opts["trade_date"]).dt.date

    # HV factors
    hv20 = realized_vol(spot["close"], 20, annualization_days)
    hv60 = realized_vol(spot["close"], 60, annualization_days)
    spot_factors = spot[["date"]].copy()
    spot_factors["HV_20D"] = hv20
    spot_factors["HV_60D"] = hv60

    # ATM IV (interpolated to target_dte)
    atm_df = choose_atm_rows(opts, max_abs_moneyness_pct)
    atm_group = list(atm_df.groupby("trade_date"))

    def _calc_for_date(item):
        date, g = item
        iv30 = interpolate_to_target_iv(g, target_dte)
        iv60 = interpolate_to_target_iv(g, secondary_dte)
        return date, iv30, iv60

    if NUM_WORKERS > 1:
        results = Parallel(n_jobs=NUM_WORKERS, backend="loky")(delayed(_calc_for_date)(it) for it in atm_group)
    else:
        results = [_calc_for_date(it) for it in atm_group]

    atm_merged = pd.DataFrame(results, columns=["date", "ATM_IV_30D", "ATM_IV_60D"])

    # TERM STRUCTURE
    atm_merged["TERM_STRUCTURE_60D_30D"] = atm_merged["ATM_IV_60D"] - atm_merged["ATM_IV_30D"]

    # SKEW 25-delta approx: choose quantile-based OTM as proxy if delta unavailable
    # Approximation: use lower 25% strike (put) and upper 75% strike (call) around ATM rows to get IVs
    def _skew_proxy(g: pd.DataFrame) -> float:
        if g.empty:
            return np.nan
        # separate calls and puts near-ATM set
        c = g[g["call_put"] == "C"].sort_values("strike")
        p = g[g["call_put"] == "P"].sort_values("strike")
        if c.empty or p.empty:
            return np.nan
        # use quartiles as rough 25-delta proxies
        c_iv = c["iv"].quantile(0.75)
        p_iv = p["iv"].quantile(0.25)
        return float(p_iv - c_iv)

    # 复用分组并行
    def _skew_for_date(item):
        date, g = item
        return date, _skew_proxy(g)

    if NUM_WORKERS > 1:
        skew_results = Parallel(n_jobs=NUM_WORKERS, backend="loky")(delayed(_skew_for_date)(it) for it in atm_group)
    else:
        skew_results = [_skew_for_date(it) for it in atm_group]

    skew_merged = pd.DataFrame(skew_results, columns=["date", "SKEW_30D_25DELTA"])

    # PCR by OI and VOL
    opts["is_call"] = (opts["call_put"] == "C").astype(int)
    daily = opts.groupby(["trade_date", "call_put"]).agg({"oi": "sum", "vol": "sum"}).reset_index()
    piv = daily.pivot(index="trade_date", columns="call_put", values=["oi", "vol"]).fillna(0)
    piv.columns = [f"{a}_{b}" for a, b in piv.columns]
    piv.reset_index(inplace=True)
    piv.rename(columns={"trade_date": "date"}, inplace=True)
    piv["PCR_OI"] = piv.get("oi_P", 0) / (piv.get("oi_C", 0) + 1e-9)
    piv["PCR_VOL"] = piv.get("vol_P", 0) / (piv.get("vol_C", 0) + 1e-9)

    # merge all
    factors = spot_factors.merge(atm_merged, on="date", how="left").merge(skew_merged, on="date", how="left").merge(piv[["date", "PCR_OI", "PCR_VOL" ]], on="date", how="left")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    factors.to_csv(output_csv, index=False)


def main():
    import yaml
    cfg = yaml.safe_load(open("Volatility_Prediction_Strategy/config/params.yaml", "r", encoding="utf-8"))
    paths = cfg["paths"]
    rates_cfg = cfg["rates"]
    filters = cfg["filters"]
    build_factors(
        underlying_csv=paths["underlying_csv"],
        options_derived_csv=os.path.join(paths["output_dir"], "options_derived.csv"),
        output_csv=os.path.join(paths["output_dir"], "factors.csv"),
        target_dte=cfg["vol_targets"]["target_dte_days"],
        secondary_dte=cfg["vol_targets"]["secondary_target_dte_days"],
        annualization_days=rates_cfg["annualization_days"],
        max_abs_moneyness_pct=filters["max_abs_moneyness_pct"],
    )
    print("written factors.csv")


if __name__ == "__main__":
    main()
