import math
import datetime as dt
from typing import Tuple, Optional

import pandas as pd


def parse_ts_code(ts_code: str) -> Tuple[str, float, str]:
    """
    Parse CFFEX IO ts_code like 'IO2002-C-3950.CFX' into (call_put, strike, contract_month).
    contract_month is YYMM string, e.g. '2002'.
    """
    # Expected format: <symbol><yymm>-<C/P>-<strike>.<ex>
    base, _ex = ts_code.split(".") if "." in ts_code else (ts_code, "")
    parts = base.split("-")
    if len(parts) < 3:
        raise ValueError(f"Unexpected ts_code format: {ts_code}")
    month = parts[0][-4:]
    call_put = parts[1].upper()
    strike = float(parts[2])
    return call_put, strike, month


def third_friday(year: int, month: int) -> dt.date:
    """
    Approximate CFFEX equity index option expiration: the 3rd Friday of the month.
    If non-trading day, upstream alignment should move to previous trading day using calendar later.
    """
    first_day = dt.date(year, month, 1)
    weekday = first_day.weekday()  # Monday=0 ... Sunday=6
    # Friday=4; days to first Friday
    days_to_first_friday = (4 - weekday) % 7
    first_friday = first_day + dt.timedelta(days=days_to_first_friday)
    third = first_friday + dt.timedelta(days=14)
    return third


def infer_expiration_date(contract_month: str) -> dt.date:
    """
    contract_month: YYMM string, e.g. '2002' -> 2020-02  (assumes 2000-2099)
    """
    if len(contract_month) != 4:
        raise ValueError(f"Unexpected contract_month: {contract_month}")
    yy = int(contract_month[:2])
    mm = int(contract_month[2:])
    year_full = 2000 + yy
    return third_friday(year_full, mm)


def compute_dte(trade_date: dt.date, expiration_date: dt.date, trading_calendar: pd.Index) -> int:
    """
    Compute remaining trading days between trade_date (exclusive) and expiration_date (inclusive),
    using trading_calendar (Index of pd.Timestamp normalized to date).
    """
    if isinstance(trading_calendar, pd.DatetimeIndex):
        cal = trading_calendar.normalize().date
    else:
        cal = pd.to_datetime(trading_calendar).normalize().date
    cal = pd.Index(cal)
    mask = (cal > trade_date) & (cal <= expiration_date)
    return int(mask.sum())


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_pdf(x: float) -> float:
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)


def bsm_price(is_call: bool, s: float, k: float, t: float, r: float, q: float, sigma: float) -> float:
    if t <= 0 or sigma <= 0 or s <= 0 or k <= 0:
        return max(0.0, (s - k) if is_call else (k - s))
    d1 = (math.log(s / k) + (r - q + 0.5 * sigma * sigma) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    if is_call:
        return math.exp(-q * t) * s * norm_cdf(d1) - math.exp(-r * t) * k * norm_cdf(d2)
    else:
        return math.exp(-r * t) * k * norm_cdf(-d2) - math.exp(-q * t) * s * norm_cdf(-d1)


def bsm_greeks(is_call: bool, s: float, k: float, t: float, r: float, q: float, sigma: float) -> Tuple[float, float, float, float]:
    if t <= 0 or sigma <= 0 or s <= 0 or k <= 0:
        return 0.0, 0.0, 0.0, 0.0
    sqrt_t = math.sqrt(t)
    d1 = (math.log(s / k) + (r - q + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    pdf = norm_pdf(d1)
    delta = math.exp(-q * t) * (norm_cdf(d1) if is_call else norm_cdf(d1) - 1.0)
    gamma = math.exp(-q * t) * pdf / (s * sigma * sqrt_t)
    vega = math.exp(-q * t) * s * pdf * sqrt_t
    theta = (
        -math.exp(-q * t) * s * pdf * sigma / (2 * sqrt_t)
        - (r * k * math.exp(-r * t) * norm_cdf(d2) - q * s * math.exp(-q * t) * norm_cdf(d1))
        if is_call
        else -math.exp(-q * t) * s * pdf * sigma / (2 * sqrt_t)
        + (r * k * math.exp(-r * t) * norm_cdf(-d2) - q * s * math.exp(-q * t) * norm_cdf(-d1))
    )
    return delta, gamma, vega, theta


def implied_vol(price: float, is_call: bool, s: float, k: float, t: float, r: float, q: float,
                 initial_iv: float = 0.2, tol: float = 1e-6, max_iter: int = 100,
                 min_iv: float = 1e-4, max_iv: float = 5.0) -> Optional[float]:
    # Newton-Raphson with safeguards
    sigma = max(initial_iv, min_iv)
    for _ in range(max_iter):
        model = bsm_price(is_call, s, k, t, r, q, sigma)
        diff = model - price
        if abs(diff) < tol:
            return max(min(sigma, max_iv), min_iv)
        # vega as derivative wrt sigma
        _, _, vega, _ = bsm_greeks(is_call, s, k, t, r, q, sigma)
        if vega <= 1e-8:
            break
        sigma -= diff / vega
        if sigma <= min_iv or sigma >= max_iv or not math.isfinite(sigma):
            # fallback to bisection
            low, high = min_iv, max_iv
            for _ in range(60):
                mid = 0.5 * (low + high)
                val = bsm_price(is_call, s, k, t, r, q, mid) - price
                if abs(val) < tol:
                    return mid
                if val > 0:
                    high = mid
                else:
                    low = mid
            return mid
    return None


def to_date_ymd(s: str) -> dt.date:
    return pd.to_datetime(s).date()


def rate_lookup_1m(rates: pd.DataFrame, date: dt.date, tenor_col: str = "1M") -> float:
    """Return annualized risk-free rate (decimal) for the given date using 1M, ffilled."""
    rates = rates.copy()
    rates["date"] = pd.to_datetime(rates["date"]).dt.date
    rates = rates.sort_values("date").set_index("date")
    if rates.empty or tenor_col not in rates.columns:
        return 0.0
    if date in rates.index:
        v = rates.loc[date, tenor_col]
    else:
        v = rates[tenor_col].reindex(rates.index.union(pd.Index([date]))).sort_index().ffill().loc[date]
    try:
        return float(v) / 100.0
    except Exception:
        return 0.0


def choose_option_price_for_iv(row: pd.Series) -> Optional[float]:
    for col in ("settle", "close", "pre_settle", "pre_close"):
        if col in row and pd.notna(row[col]) and float(row[col]) > 0:
            return float(row[col])
    return None
