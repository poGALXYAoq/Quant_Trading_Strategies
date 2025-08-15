# 波动率预测与期权交易策略建模(Volatility Prediction and Options Trading Strategy Modeling)

## 项目简介 (Project Overview)

这是一个旨在通过机器学习方法，对**金融市场中的各类标的物（如股指、ETF、个股等）**的未来波动率进行预测，并基于预测结果构建自动化期权交易策略的通用量化研究框架。

项目的核心是量化**未来已实现波动率 (Realized Volatility, RV)** 与**当前期权市场隐含波动率 (Implied Volatility, IV)** 之间的价差，以指导买入或卖出波动率的期权组合（如跨式、宽跨式等），从而系统性地捕捉不同标的物上的波动率风险溢价。

## 核心目标 (Core Objective)

本项目旨在回答以下核心问题：
> **我们能否通过历史数据和期权衍生数据，建立一个能有效预测未来一段时间内，特定标的物的已实现波动率将显著高于还是低于当前市场隐含波动率的通用模型？**

模型的输出将是一个分类信号，指导具体的交易决策：
* **信号 1 (做多波动率):** 预测未来RV将显著高于当前IV。
* **信号 -1 (做空波动率):** 预测未来RV将显著低于当前IV。
* **信号 0 (无操作):** 预测未来RV与当前IV无显著差异，不存在交易机会。

## 策略逻辑 (Strategy Logic)

模型的预测信号将直接映射到具体的期权策略：
* **接收到信号 1:** 执行 **买入跨式/宽跨式组合 (Long Straddle/Strangle)**，押注未来市场波动加剧。
* **接收到信号 -1:** 执行 **卖出跨式/宽跨式组合 (Short Straddle/Strangle)**，押注未来市场风平浪静，以赚取时间价值和波动率下降的利润。

### 目标变量与时间对齐 (Target & Time Alignment)

- 决策时点 `t` 使用当日收盘后可得的 `IV_30D`（标准化至30个交易日）。
- 未来实现波动率 `RV_{t→t+30D}` 基于标的在 [t, t+30个交易日] 的收盘收益率计算（用交易日而非自然日）。
- 标签定义（示例）：
  - 计算差值 `diff = RV_{t→t+30D} - IV_{t,30D}`。
  - 设定阈值 `ε`（如历史滚动中位数或60分位），得到分类信号：
    - `diff > ε` → 信号 `1`（做多波动率）
    - `diff < -ε` → 信号 `-1`（做空波动率）
    - 其余 → 信号 `0`（不交易）
- 标签窗口重叠会导致样本相关性：评估阶段需使用滚动窗口（或周频/月频触发）减少高相关性偏误。

## 技术栈 (Tech Stack)

* **数据处理:** `Python`, `Pandas`, `NumPy`
* **机器学习:** `Scikit-learn`, `XGBoost`, `LightGBM`
* **期权定价:** 自定义或使用 `QuantLib` 等库实现Black-Scholes-Merton模型
* **数据可视化:** `Matplotlib`, `Seaborn`,`Plotly`
* **数据源 (初期):** 本地CSV文件或数据库 (SQLite)

## 项目工作流 (Project Workflow)

本项目将遵循标准的量化研究流程，以迭代方式进行开发：

1.  **数据采集与清洗:** 从数据源获取所需数据，进行缺失值处理、异常值检测和数据对齐。
2.  **特征工程 (因子库构建):** 基于原始数据，计算并合成一系列用于预测波动率的因子。
3.  **模型训练:** 使用**步进向前验证 (Walk-Forward Validation)** 的方式训练机器学习模型，以避免前视偏差。
4.  **回测引擎:** 开发一个事件驱动的回测系统，模拟真实的交易执行、手续费和滑点。
5.  **策略执行逻辑:** 将模型信号转化为具体的开平仓指令，包括选券逻辑（选择到期日、行权价）。
6.  **绩效分析:** 对回测结果进行全面的绩效评估，包括夏普比率、最大回撤、年化收益率等。

notebook的大致流程将会是：load data, train model, predict, backtest, analysis.
考虑各个结构之间松解耦模块化，模型部分预设多种模型以便于调整，策略执行层面也考虑多种多样的逻辑可选例如Long Straddle 买入跨式组合，Short Collar 空头领式策略等等。

## 模块化架构 (Modular Architecture)

- 数据层 `data/`：原始数据与衍生数据（清洗后、标准化后的因子表、IV曲面快照）。
- 特征工程 `features/`：从原始行情与期权链计算 `HV/ATR/RSI/ATM_IV_30D/TERM_STRUCTURE/SKEW/PCR` 等因子。
- 模型层 `models/`：训练、验证、保存模型与阈值；支持 `sklearn/XGBoost/LightGBM` 的统一接口。
- 交易执行 `execution/`：选券（到期、行权价）、下单价、成交假设、手续费与滑点、保证金估算。
- 回测引擎 `backtest/`：事件驱动（每日/每周触发），仓位管理，PnL 归因（theta/gamma/vega）。
- 评估与可视化 `report/`：绩效指标、曲线、因子贡献与敏感度分析。
- 配置 `config/`：标的、交易日历、费率、阈值、风控参数等等。

### 目录结构（当前）

```
Volatility_Prediction_Strategy/
  ├─ data/
  │   ├─ CSI300.csv
  │   ├─ option_CSI300_2020-24.csv
  │   ├─ interbank_offered_rate.csv
  │   └─ derived/
  ├─ features/
  │   ├─ iv_utils.py                # ts_code解析、到期日/交易日、BSM/IV/Greeks
  │   ├─ build_derived_tables.py    # 读取三表→派生字段→反推IV→Greeks→导出
  │   ├─ build_factors.py           # 计算HV/ATM_IV/TERM/SKEW/PCR等因子
  │   └─ build_labels.py            # 用RV_30D与ATM_IV_30D构造分类标签与ε阈值
  ├─ execution/
  │   └─ selection.py               # 近30D平值跨式/宽跨式选券
  ├─ backtest/
  │   ├─ engine.py                  # 回测引擎（T+1成交、持有期/到期前N天平仓）
  │   ├─ metrics.py                 # 绩效指标计算
  │   └─ run_backtest.py            # 载入参数、运行回测、输出净值与指标
  ├─ report/
  ├─ config/
  │   └─ params.yaml                # 路径、利率、过滤、IV反推、回测参数
  └─ README.md
```

## 数据需求 (Data Requirements)

* **标的物行情:** **目标标的物**的日线K线数据 (例如：沪深300指数, 上证50ETF, 特定股票等)。
* **期权链数据:** **目标标的物对应**的每日全量期权链数据（包括所有行权价和多个到期月份）。
* **无风险利率:** 市场无风险利率的历史时间序列 (例如: SHIBOR)。

当前阶段选用沪深300指数为标的物，对应的期权为沪深300股指期权（欧式，CFFEX）。

### 数据字段规范（需准备的数据表）

#### 1) 股指日频行情 `data/CSI300.csv`

| 字段 | 类型 | 描述 | 示例 |
| --- | --- | --- | --- |
| date | date | 交易日（YYYY-MM-DD） | 2020-01-02 |
| prev_close | float | 前收 | 4096.5821 |
| high | float | 最高 | 4172.6555 |
| low | float | 最低 | 4121.3487 |
| open | float | 开盘 | 4121.3487 |
| volume | float | 成交量 | 18211677200.0 |
| close | float | 收盘 | 4152.2408 |
| total_turnover | float | 成交额 | 270105532027.6 |

备注：后续以 `close` 计算收益与波动，信号在收盘生成，默认 T+1 用 `open` 成交（可切换为 `settle` 方案，见回测设置）。

#### 2) 期权链日频快照 `data/option_CSI300_YYYY-YY.csv`

| 字段 | 类型 | 描述 | 示例 |
| --- | --- | --- | --- |
| ts_code | string | 合约代码（含品种、到期、C/P、行权价、交易所） | IO2002-C-3950.CFX |
| trade_date | date | 交易日（YYYYMMDD） | 20200123 |
| exchange | string | 交易所 | CFFEX |
| pre_settle | float | 前结算价 | 208.2 |
| pre_close | float | 前收 | 216.4 |
| open | float | 开盘价 | 169.0 |
| high | float | 最高价 | 169.0 |
| low | float | 最低价 | 98.6 |
| close | float | 收盘价 | 116.2 |
| settle | float | 结算价 | 116.2 |
| vol | float/int | 成交量 | 287.0 |
| amount | float | 成交额 | 350.014 |
| oi | float/int | 持仓量 | 201.0 |

派生字段（由 `ts_code` 与交易规则解析/计算）：

| 字段 | 类型 | 描述 | 示例 |
| --- | --- | --- | --- |
| call_put | string | C/P 标识 | C |
| strike | float | 行权价（从代码解析） | 3950 |
| contract_month | string | 合约到期年月（YYMM） | 2002 |
| expiration_date | date | 合约到期日（近似：每月第3个周五，若休市则前一交易日） | 2020-02-21 |
| dte | int | 剩余到期交易日数（用股指交易日历近似） | 20 |
| option_price_for_iv | float | 反推IV使用价格（优先 `settle`，否则 `close`） | 116.2 |
| iv | float | 隐含波动率（BSM 欧式，q≈0 近似） | 0.215 |
| delta,gamma,vega,theta | float | 希腊字母（由 BSM + iv 计算） | 见计算 |

说明：
- `expiration_date` 与 `dte` 使用“每月第3个周五”规则近似，并以股指CSV中的交易日作为交易日历。
- 由于缺少股息率，这里先取 `q≈0` 近似；后续可在 `config/` 中设定常数或时间序列进行敏感度测试。
- IV 反推：用股指 `close`、利率 `1M`、`T=dte/252`、以及期权 `option_price_for_iv` 进行 BSM 欧式求解。

#### 3) 银行间拆借利率（无风险利率）`data/interbank_offered_rate.csv`

| 字段 | 类型 | 描述 | 示例 |
| --- | --- | --- | --- |
| date | date | 日期（YYYY-MM-DD） | 2020-01-02 |
| ON,1W,2W,1M,3M,6M,9M,1Y | float | 各期限年化利率（%） | 1.4436, 2.433, ..., 3.086 |

说明：
- 以 `1M` 作为 30D 标准化期限的无风险利率近似；换算为小数形式 `rf_annual_1M = 1M/100`。
- 若需其它期限，按相邻期限线性插值近似。

### 数据对齐与清洗规则（基于现有字段）

- 交易日历：以股指CSV的 `date` 序列为主，期权 `trade_date` 转换为日期并对齐。
- 价格过滤：移除非正价格、异常跳变（可用分位数/倍数阈值），优先使用 `settle` 作为定价参考。
- 流动性过滤：按 `vol`、`oi` 设下限（如 `oi >= 100`），剔除流动性过低的合约。
- 合约接近到期：当 `dte` 过小（如 ≤ 3 交易日）时避免新开仓，减少行权结算噪声。

## 特征工程（因子库） (Feature Engineering)

因子库是本项目的核心资产，将包含但不限于以下可适用于多种标的物的因子：

### A. 基于标的物历史数据
* `HV_20D`, `HV_60D`: 基于日线收盘价计算的20日、60日历史波动率。
* `ATR_14D`: 14日平均真实波幅。
* `RSI_14D`: 14日相对强弱指数。

### B. 基于期权合成数据
* `ATM_IV_30D`: 通过插值计算的标准化30天期**平价隐含波动率**。
* `TERM_STRUCTURE_60D_30D`: 60天与30天ATM IV的价差，反映波动率期限结构。
* `SKEW_30D_25DELTA`: 30天期25-delta看跌与看涨期权IV的价差，反映市场偏斜与恐慌情绪。
* `PCR_OI`, `PCR_VOL`: 基于持仓量和成交量的看跌/看涨比率。

### 选券与执行细则（Straddle/Strangle）

- 期限选择：优先选择 DTE 最接近 30 个交易日的到期；若不存在，使用相邻两期按时间线性加权标准化到 30D（用于因子/目标）。
- ATM 定义：基于远期价格的平值（非现货）。选取行权价最接近远期-ATM 的看涨与看跌构成交叉。
- 宽跨式（Strangle）宽度：按标的价格百分比或以 delta 目标（如 |delta|≈0.25）选取 OTM 看涨/看跌。
- 卖方结构：为控制尾部风险，优先采用有界结构（如铁秃鹰/价差）替代裸卖跨式。
- 对冲选项：
  - 非对冲版本（更接近多数实际交易者）；
  - 日终 delta 对冲版本（更纯粹表达“买/卖波动率”的观点）。
- 平仓规则：持有至到期前 N 日或达到止盈/止损阈值；出现信号翻转可择时换仓。

### 波动率曲面构建与标准化

- 使用全链条 IV 点（到期×行权价）构建日内快照；先基于远期价格计算 moneyness。
- 期限维度：对目标期限（30D、60D）做时间插值；形状维度可采用在 moneyness 上的样条插值（SABR/SVI 作为后续升级）。
- 由标准化曲面派生 `ATM_IV_30D`、`TERM_STRUCTURE_60D_30D`、`SKEW_30D_25DELTA` 等因子；25-delta 的定义需与远期/BSM 口径一致。

### 模型与验证（Walk-Forward）

- 训练-验证-测试滚动：严格时间因果，参数在验证集固定后，应用于后推测试集。
- 类别不均衡：波动率风险溢价通常导致 `RV < IV` 偏多，采用阈值与成本敏感学习；评估以交易收益为主。
- 指标：除分类指标外，关注 PnL/夏普/最大回撤/卡玛/尾损分位/回撤恢复时间及保证金占用口径收益。

### 回测设置与成交假设（无 bid/ask 情况）

- 信号在 `t` 收盘生成，默认 `t+1` 用期权 `open` 成交；若 `open` 缺失，则用 `settle`。平仓同理。
- 滑点与手续费：设定固定滑点（如每腿 0.5 元）与每手固定手续费（配置于 `config/`）。
- 规模：按目标名义 Vega 或固定名义资金占比建仓，跨期限保持风险可比。
- 卖方结构需保证金数据；在数据受限下优先回测多波动率（买跨/买宽跨）。

### 风险管理与头寸规模

- 尾部保护：优先采用价差/铁秃鹰；设置极端波动下的熔断式减仓。
- 事件日：财报/重磅政策/节假日前后限仓或不交易。
- 止盈止损：基于组合 PnL 或 IV 收敛/扩散幅度的动态调节。

## MVP 落地清单（基于现有三类数据）

- 数据：上述 3 张表（股指、期权、拆借利率）覆盖 3-5 年；期权需尽量全链条（多到期、多行权）。
- 因子：实现 `HV_20/60`、`ATM_IV_30D`（由IV反推+期限插值）、`TERM_STRUCTURE_60D_30D`、`SKEW_30D_25DELTA`（由delta选取）、`PCR_{OI,VOL}`（按C/P聚合）。
- 目标：`y = sign(RV_{30D} - IV_{30D} - ε)`；`ε` 为历史滚动分位阈值，降低噪声交易。
- 执行：
  - 多波动率：买入最接近 30D 的平值跨式（现货ATM近似），可选日终 delta 对冲；
  - 空波动率：因无保证金数据，先不纳入MVP主线，或仅以有限风险的价差结构做小规模实验。
- 评估：以策略净值与夏普/最大回撤/尾损分位为核心；记录建仓的 DTE、ATM 偏差与IV 估计稳定性以做健壮性检验。

### 运行步骤（示例）

1. 生成期权衍生表（IV/Greeks）：
   - 运行：`python Volatility_Prediction_Strategy/features/build_derived_tables.py`
   - 输出：`data/derived/options_derived.csv`
2. 生成因子：
   - 运行：`python Volatility_Prediction_Strategy/features/build_factors.py`
   - 输出：`data/derived/factors.csv`
3. 生成标签：
   - 运行：`python Volatility_Prediction_Strategy/features/build_labels.py`
   - 输出：`data/derived/labels.csv`
4. 回测：
   - 运行：`python Volatility_Prediction_Strategy/backtest/run_backtest.py`
   - 输出：`data/derived/equity.csv` 与控制台打印的指标。

### 并行与性能

- 可通过环境变量或脚本常量设置并行度：
  - 环境变量：`VP_NUM_WORKERS=8`（线程数，Windows 上推荐线程并行；计算密集场景可换为进程并行在后续版本中开启）。
  - 位置：`features/build_derived_tables.py`、`features/build_factors.py` 均支持 `NUM_WORKERS` 设置；`build_labels.py` 暂不需要并行。
- Windows 下建议使用线程池并行（已默认），避免进程开销。

## 未来工作 (Future Work)

* [ ] **多品种扩展:** 将框架设计得更具扩展性，使其能够并行处理和回测多个不同标的物。
* [ ] **升级目标变量:** 待获取分钟数据后，将模型的预测目标 `y` 从基于HV的粗糙指标升级为基于RV的精确指标。
* [ ] **扩充因子库:** 引入更多另类数据和宏观经济因子。
* [ ] **模型优化:** 尝试不同的机器学习模型（如深度学习LSTM）和超参数调优。
* [ ] **动态头寸管理:** 开发更精细化的出场策略（如动态止盈止损）和资金管理模块。
* [ ] **迁移学习:** 探索将在全市场数据上预训练的通用模型，微调至**特定目标标的物**上的可能性。

