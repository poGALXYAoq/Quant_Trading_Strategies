"""蝶式套利监测脚本。

该脚本周期性抓取指定商品期权的行情数据，寻找满足
2 * Bid(B) > Ask(A) + Ask(C)、且行权价等距的蝶式组合，
并在发现潜在错误定价时输出提示。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List

import akshare as ak
import pandas as pd


@dataclass(frozen=True)
class OptionTarget:
    symbol: str
    contract: str


@dataclass(frozen=True)
class MonitorSettings:
    interval: float = 30.0
    min_open_interest: int = 100  # 最小持仓量


@dataclass(frozen=True)
class MonitorTask:
    target: OptionTarget
    settings: MonitorSettings


def split_option_chains(
    raw_df: pd.DataFrame,
    *,
    min_open_interest: int,
) -> Dict[str, pd.DataFrame]:
    """抓取并整理看涨期权行情数据。"""

    result: Dict[str, pd.DataFrame] = {}

    for option_type in ("C", "P"):
        if option_type == "C":
            prefix = "看涨合约"
            code_col = "看涨合约-看涨期权合约"
        else:
            prefix = "看跌合约"
            code_col = "看跌合约-看跌期权合约"

        rename_map = {
            "行权价": "strike",
            f"{prefix}-买价": "bid_price",
            f"{prefix}-卖价": "ask_price",
            f"{prefix}-持仓量": "open_interest",
            f"{prefix}-买量": "bid_volume",
            f"{prefix}-卖量": "ask_volume",
            code_col: "option_code",
        }

        required_cols = list(rename_map.keys())
        missing_cols = [col for col in required_cols if col not in raw_df.columns]
        if missing_cols:
            raise KeyError(f"缺失必要列: {missing_cols}")

        df = raw_df[required_cols].rename(columns=rename_map).copy()

        for col in ("strike", "bid_price", "ask_price", "open_interest", "bid_volume", "ask_volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["strike", "bid_price", "ask_price"])

        df = df[df["open_interest"].fillna(0) >= min_open_interest]
        df = df[df["bid_price"] > 0]
        df = df[df["ask_price"] > 0]

        df = (
            df.sort_values(["strike", "open_interest"], ascending=[True, False])
            .drop_duplicates(subset=["strike"], keep="first")
            .reset_index(drop=True)
        )

        result[option_type] = df

    return result


def find_butterfly_arbitrage(
    option_df: pd.DataFrame,
    *,
    min_open_interest: int,
    symbol: str,
    contract: str,
    option_type: str,
) -> List[Dict[str, float]]:
    """在单个期权链内查找蝶式错误定价。"""

    if option_df.empty:
        return []

    strikes = option_df["strike"].tolist()
    strike_to_row = option_df.set_index("strike").to_dict("index")

    signals: List[Dict[str, float]] = []

    for center_strike in strikes:
        center_row = strike_to_row.get(center_strike)
        if center_row is None:
            continue

        steps = [abs(center_strike - other) for other in strikes if other != center_strike]
        for step in sorted(set(steps)):
            if step == 0:
                continue

            lower_strike = center_strike - step
            upper_strike = center_strike + step

            lower_row = strike_to_row.get(lower_strike)
            upper_row = strike_to_row.get(upper_strike)

            if lower_row is None or upper_row is None:
                continue

            if (
                lower_row.get("open_interest", 0) < min_open_interest
                or center_row.get("open_interest", 0) < min_open_interest
                or upper_row.get("open_interest", 0) < min_open_interest
            ):
                continue

            mid_bid = center_row.get("bid_price")
            lower_ask = lower_row.get("ask_price")
            upper_ask = upper_row.get("ask_price")

            if None in (mid_bid, lower_ask, upper_ask):
                continue

            parity_gap = 2 * mid_bid - (lower_ask + upper_ask)
            if parity_gap <= 0:
                continue

            signals.append(
                {
                    "symbol": symbol,
                    "contract": contract,
                    "option_type": option_type,
                    "strike_lower": lower_strike,
                    "strike_middle": center_strike,
                    "strike_upper": upper_strike,
                    "mid_bid": mid_bid,
                    "lower_ask": lower_ask,
                    "upper_ask": upper_ask,
                    "parity_gap": parity_gap,
                }
            )

    return signals


def run_monitor(tasks: Iterable[MonitorTask]) -> None:
    """循环执行监测任务，按各自设定的间隔独立调度。"""

    task_list = list(tasks)
    if not task_list:
        logging.warning("未提供监测任务")
        return

    summary = ", ".join(
        f"{task.target.symbol}:{task.target.contract}(间隔={task.settings.interval}s, 持仓阈值={task.settings.min_open_interest})"
        for task in task_list
    )
    logging.info("监测目标: %s", summary)

    task_states = [
        {"task": task, "next_run": 0.0}
        for task in task_list
    ]

    while True:
        now = time.time()
        due_states = [state for state in task_states if now >= state["next_run"]]

        if not due_states:
            next_run = min(state["next_run"] for state in task_states)
            sleep_time = max(0.0, next_run - now)
            logging.info("无任务到期，等待 %.2f 秒进入下一轮", sleep_time)
            try:
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    time.sleep(0.5)
            except KeyboardInterrupt:
                logging.info("检测终止")
                return
            continue

        for state in due_states:
            task = state["task"]
            target = task.target
            run_start = time.time()
            try:
                logging.info("开始查询 %s-%s 行情", target.symbol, target.contract)
                raw_df = ak.option_commodity_contract_table_sina(
                    symbol=target.symbol,
                    contract=target.contract,
                )
            except Exception as exc:  # pylint: disable=broad-except
                logging.exception("抓取 %s-%s 行情失败: %s", target.symbol, target.contract, exc)
                state["next_run"] = time.time() + task.settings.interval
                continue

            if raw_df is None or raw_df.empty:
                logging.warning("%s-%s 行情为空", target.symbol, target.contract)
                state["next_run"] = time.time() + task.settings.interval
                continue

            option_chains = split_option_chains(
                raw_df,
                min_open_interest=task.settings.min_open_interest,
            )

            for option_type, option_df in option_chains.items():
                signals = find_butterfly_arbitrage(
                    option_df,
                    min_open_interest=task.settings.min_open_interest,
                    symbol=target.symbol,
                    contract=target.contract,
                    option_type=option_type,
                )

                if not signals:
                    logging.info("%s-%s-%s 本轮未发现蝶式机会", target.symbol, target.contract, option_type)
                else:
                    for sig in signals:
                        logging.info(
                            "发现蝶式机会 %s-%s-%s | %s-%s-%s | 2*Bid=%.2f, Ask合计=%.2f, 差值=%.2f",
                            sig["symbol"],
                            sig["contract"],
                            sig["option_type"],
                            sig["strike_lower"],
                            sig["strike_middle"],
                            sig["strike_upper"],
                            2 * sig["mid_bid"],
                            sig["lower_ask"] + sig["upper_ask"],
                            sig["parity_gap"],
                        )

            duration = time.time() - run_start
            logging.info("完成 %s-%s 查询，耗时 %.2f 秒", target.symbol, target.contract, duration)
            state["next_run"] = time.time() + task.settings.interval


def configure_logging(level: str) -> None:
    if logging.getLogger().handlers:
        logging.getLogger().setLevel(getattr(logging, level.upper()))
        return
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )


def start_monitor(
    tasks: Iterable[MonitorTask],
    *,
    log_level: str = "INFO",
) -> None:
    configure_logging(log_level)
    run_monitor(tasks)

