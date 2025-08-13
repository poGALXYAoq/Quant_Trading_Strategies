# 策略文档：受保护的期货逆势网格策略 (Protected Futures Reverse Grid Strategy)

本策略以期货多头为核心，买入虚值看跌期权作为保险，在持仓期间按价格网格对期货头寸进行逆势加减仓。期权仓位持有至到期，在到期日统一平仓。

## 快速开始
```bash
python Protected_Futures_Reverse_Grid_Strategy/backtest.py
```
默认读取的数据路径（如需更改，请在脚本顶部参数区修改）：
- 期货：`Protected_Futures_Reverse_Grid_Strategy/data/PTA2501/futures_data.csv`
- 期权：`Protected_Futures_Reverse_Grid_Strategy/data/PTA2501/options_data.csv`
输出：
- `Protected_Futures_Reverse_Grid_Strategy/backtest_results/PTA2501/daily_log.csv`
- 交互式报告：`Protected_Futures_Reverse_Grid_Strategy/backtest_results/PTA2501/PTA2501分析报告.html`

## 数据准备
可使用 `ETL/PFRG` 中的脚本将原始数据转换为回测所需格式：
- `process_futures_data.py` 将原始期货文件转换为：
  - 格式：`datetime,open,high,low,close,volume`
  - 路径：`Protected_Futures_Reverse_Grid_Strategy/data/<symbol>/futures_data.csv`
- `process_options_data.py` 将原始期权文件与期货文件合并标的价，筛选目标合约前缀，输出：
  - 格式：`datetime,underlying_price,strike_price,option_type,close`
  - 路径：`Protected_Futures_Reverse_Grid_Strategy/data/<symbol>/options_data.csv`

提示：脚本中示例路径包含反斜杠（Windows 风格）。在 Linux/MacOS 下请改用正斜杠或 `os.path.join`。

## 核心参数（在 `backtest.py` 顶部修改）
- 入场与价格口径：
  - `START_DATE_STR`: 启动日期
  - `START_PRICE_MODE`: `open` 或 `close`（建仓价口径）
- 头寸规模：
  - `FUTURES_INITIAL_LOTS`: 初始期货手数
  - `PUT_OPTION_LOTS`: 保护性看跌手数（全程不变）
- 期权选择：
  - `STRIKE_PRICE_OFFSET`: 相对于标的参考价的行权价偏移（负值偏虚值）
  - `SELECT_STRIKE_METHOD`: `atm` | `nearest` | `floor` | `ceiling`
- 网格与限制：
  - `PRICE_GRID_INTERVAL`: 网格价格步长
  - `LOTS_ADJUSTMENT_STEP`: 每档增减手数
  - `FUTURES_MAX_LOTS` / `FUTURES_MIN_LOTS`: 仓位上下限
- 成本参数：
  - `COMMISSION_FUTURES`, `COMMISSION_OPTIONS`, `SLIPPAGE`
- 可视化：
  - `PLOT_TITLE`, `PLOT_NAME`
- 文件路径：
  - `FUTURES_DATA_PATH`, `OPTIONS_DATA_PATH`, `RESULTS_DIR`

## 逻辑摘要
1) 首日在 `START_DATE_STR`：
- 建立期货多头 `FUTURES_INITIAL_LOTS` 手（`START_PRICE_MODE` 决定价格口径）。
- 按当日期权数据中的 `underlying_price` 与 `STRIKE_PRICE_OFFSET`，结合 `SELECT_STRIKE_METHOD` 选择看跌期权并买入 `PUT_OPTION_LOTS` 手（保险仓位全程不动）。

2) 持有期间每日：
- 以当日收盘价与入场价计算价差，按 `PRICE_GRID_INTERVAL` 计算目标手数并对期货进行逆势加减仓，受 `FUTURES_MAX_LOTS/MIN_LOTS` 约束。

3) 到期日：
- 平掉所有期货与期权持仓，计算最终结果。

4) 输出：
- `daily_log.csv`（含净值、期货/期权市值、每日盈亏等）
- 交互式 HTML 报告（K线、交易标记、指标表）

## 依赖
```bash
pip install -U pandas numpy plotly
```

## 注意事项
- `process_options_data.py` 需指定目标合约前缀（如 `TA501`），并依赖已转换的期货 `futures_data.csv` 以合并 `underlying_price`。
- 若同日有多行数据，代码会选择首行作为参考以保证稳定性。
- 若当日缺数据，期权估值会尽量回退使用最近可得数据或入场价，保证运行连贯。