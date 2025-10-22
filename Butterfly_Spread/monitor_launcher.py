"""监测配置入口。

该模块定义需要监测的期权列表以及监测参数，
执行时会调用 `butterfly_detector` 中的检测逻辑。
"""

from __future__ import annotations

from butterfly_detector import MonitorSettings, MonitorTask, OptionTarget, start_monitor


def main() -> None:
    tasks = [
        MonitorTask(
            target=OptionTarget(symbol="豆粕期权", contract="m2601"),
            settings=MonitorSettings(interval=30.0, min_open_interest=100),
        ),
        MonitorTask(
            target=OptionTarget(symbol="沪铜期权", contract="cu2512"),
            settings=MonitorSettings(interval=45.0, min_open_interest=200),
        )
        # MonitorTask(
        #     target=OptionTarget(symbol="豆粕期权", contract="m2603"),
        #     settings=MonitorSettings(interval=45.0, min_open_interest=200),
        # ),
    ]

    start_monitor(tasks, log_level="INFO")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass

