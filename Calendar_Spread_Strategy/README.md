# Calendar_Spread_Strategy

期权跨期价差（Calendar Spread）策略回测实现：近月卖出、远月买入，同步选择行权价，参数网格遍历虚值程度 k 与目标 DTE 值 N，输出每日净值、交易明细与性能热力图。

## 数据输入
- 目标输入文件：`data/RB/master_option_data_RB.csv`
- 可通过 `ETL/CSS/prepare_data.py` 将多源数据合并成“大宽表”。关键字段：
  - `交易日期`（index）
  - `到期日`
  - `合约名称`（例如 RBxxxxCyyyy / RBxxxxPyyyy）
  - `剩余到期日`（DTE）
  - `标的开盘价`、`标的收盘价`
  - `今结算`（用于回测核心价格）

提示：在 Linux 环境下，将脚本中的反斜杠路径改为正斜杠，或在脚本内改为 `os.path.join`。

## 快速开始
- 全参数回测（`backtest.py`）：
```bash
python Calendar_Spread_Strategy/backtest.py
```
- 限定仅交易指定近月月份（`backtest_with_contract_month_filter.py`）：
```bash
python Calendar_Spread_Strategy/backtest_with_contract_month_filter.py
```

## 主要参数（在脚本顶部修改）
- `K_VALUES`: 虚值程度 k 列表，例如 `[0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08]`
- `N_RANGE`: 近月目标 DTE 值范围，例如 `range(5, 21)`
- `DTE_TOLERANCE`: 实际 DTE 与目标 N 的容忍差
- `INITIAL_CAPITAL`: 初始资金
- `CAPITAL_USAGE_RATIO`: 单组交易可动用资金比例
- `COMMISSION_PER_CONTRACT`: 每手单边手续费
- `MARGIN_RATE_FUTURES`、`MIN_MARGIN_FACTOR`: 简化保证金估算参数
- `N_JOBS`: 并行计算核数
- `DATA_PATH`: 输入数据路径（默认 `'data\RB\master_option_data_RB.csv'`，跨平台请改正斜杠）
- `RESULTS_DIR`: 输出目录

在 `backtest_with_contract_month_filter.py` 中额外提供：
- `TRADE_CONTRACT_MONTHS`: 仅交易这些近月月份（如 `[1, 5, 10]`）

## 回测逻辑摘要
- 依据到期日序列，生成开平仓计划：开仓在到期日前 N 天附近，平仓在近月到期日收盘。
- 开仓日：
  - 按当日标的开盘价计算目标行权价：
    - 看涨目标行权价: `Strike_Call = underlying_open * (1 + k)`
    - 看跌目标行权价: `Strike_Put = underlying_open * (1 - k)`
  - 在近月中选择行权价最接近目标的看涨、看跌合约进行卖出；在更远月中买入相同行权价的看涨、看跌合约。
- 资金/保证金：
  - 卖出近月所需保证金按简化规则估算；买入远月按权利金支付；综合动用资金按 `CAPITAL_USAGE_RATIO` 限制进行手数计算。
- 平仓日：对本组合约以结算价进行平仓结算，记录盈亏与费用。

## 输出
- 脚本 `backtest.py` 的默认输出目录：`backtest_results/RB`
  - 每组参数 `(k, N)`：
    - `daily_equity_log_k{k}_N{N}.csv`
    - `trade_log_k{k}_N{N}.csv`
  - 汇总：`all_results.csv`
  - 可视化：`cagr_heatmap.png`、`sharpe_heatmap.png`、`max_drawdown_heatmap.png`、`win_rate_heatmap.png`
- 脚本 `backtest_with_contract_month_filter.py` 的默认输出目录：`backtest_results_159/RB`

## 依赖
```bash
pip install -U pandas numpy matplotlib seaborn joblib tqdm
```

## 备注
- 若数据中结算价为 0，代码会在开仓时将其替换为 0.5 以避免无法成交的异常情况。
- 若部分交易日缺少当日标的或个别合约报价，代码会使用可得的信息进行稳健回退处理。
