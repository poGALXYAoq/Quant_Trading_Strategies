# 策略文档：受保护的期货逆势网格策略 (Protected Futures Reverse Grid Strategy)

## 1\. 策略概述 (Strategy Overview)

本策略旨在通过构建一个受保护的期货多头头寸来进行交易。策略的核心由两部分组成：

1.  **基础头寸**：买入期货多头，并同时买入大量对应的虚值看跌期权，为期货头寸提供下行风险保护。
2.  **动态调整**：以初始入场价为基准，对期货头寸进行逆势网格操作。当价格上涨时，分批止盈减仓；当价格下跌时，分批补仓拉低成本。

该策略试图在控制最大回撤的前提下，捕捉市场的震荡或温和上涨行情。期权头寸在整个生命周期内保持不变，仅作为保险。

## 2\. 交易标的 (Trading Instruments)

  * **期货 (Futures)**：特定品种的期货合约（例如：螺纹钢 2509，豆粕 2509）。
  * **期权 (Options)**：与上述期货合约挂钩、到期日相同的欧式看跌期权（Put Option）。

## 3\. 核心参数 (Core Parameters)

以下为本策略中所有可配置的全局参数，便于在代码中进行调整和优化。

```python
# ================= 策略核心参数 =================
# 入场相关
START_DATE = '2025-05-01'      # 策略起始日期
CONTRACT_MONTH = '2509'       # 目标合约月份

# 头寸相关
FUTURES_INITIAL_LOTS = 100    # 期货初始买入手数
PUT_OPTION_LOTS = 300         # 看跌期权买入手数

# 期权选择相关
STRIKE_PRICE_OFFSET = -200    # 看跌期权行权价与期货入场价的价差

# 网格交易相关
PRICE_GRID_INTERVAL = 10      # 触发调仓的价格区间步长
LOTS_ADJUSTMENT_STEP = 10     # 每次调仓的手数变化量

# 仓位限制
FUTURES_MAX_LOTS = 300        # 期货持仓手数下限（即最多加仓到300手）
FUTURES_MIN_LOTS = 0          # 期货持仓手数上限（即最多减仓到0手）
# ============================================
```

## 4\. 策略逻辑详解 (Detailed Strategy Logic)

### 步骤一：入场条件 (Entry Conditions)

1.  **交易日确定**：在预设的 `START_DATE` 当天开始执行。若当天为非交易日，则顺延至下一个交易日。
2.  **基准价确定**：在执行当天的开盘时，获取 `CONTRACT_MONTH` 期货合约的开盘价，此价格定义为 `ENTRY_PRICE`。
3.  **建立期货头寸**：以 `ENTRY_PRICE` 附近的价格，买入 `FUTURES_INITIAL_LOTS` 手期货多单。
4.  **建立期权保护头寸**：
      * 计算目标行权价：`TARGET_STRIKE = ENTRY_PRICE + STRIKE_PRICE_OFFSET` (即 `ENTRY_PRICE - 200`)。
      * 在市场上寻找与 `TARGET_STRIKE` 最接近的可用看跌期权行权价。
      * 买入 `PUT_OPTION_LOTS` 手该看跌期权。

### 步骤二：持仓期动态调整 (Dynamic Adjustments During Holding Period)

1.  **触发条件**：从入场后的下一个交易日开始，每日执行一次，以当日收盘价为判断依据。
2.  **期权持仓**：**全程不动**。已买入的 `PUT_OPTION_LOTS` 手看跌期权一直持有，不进行任何操作。
3.  **期货持仓调整**：
      * 获取当日 `CONTRACT_MONTH` 期货合约的收盘价，定义为 `CLOSE_PRICE`。
      * 计算当前价格与初始入场价的偏离：`PRICE_DIFF = CLOSE_PRICE - ENTRY_PRICE`。
      * **情况A：价格上涨 (`PRICE_DIFF > 0`)**
          * 计算下跌档位：`steps = floor(PRICE_DIFF / PRICE_GRID_INTERVAL)` (向下取整)。
          * 计算目标持仓手数：`TARGET_LOTS = FUTURES_INITIAL_LOTS - steps * LOTS_ADJUSTMENT_STEP`。
          * 将当前期货持仓调整至 `max(TARGET_LOTS, FUTURES_MIN_LOTS)` 手。
          * *示例：若 `ENTRY_PRICE`=3000，`CLOSE_PRICE`=3025，则 `PRICE_DIFF`=25。`steps = floor(25/10) = 2`。目标手数为 `100 - 2 * 10 = 80` 手。*
      * **情况B：价格下跌 (`PRICE_DIFF < 0`)**
          * 计算上涨档位：`steps = floor(abs(PRICE_DIFF) / PRICE_GRID_INTERVAL)` (向下取整)。
          * 计算目标持仓手数：`TARGET_LOTS = FUTURES_INITIAL_LOTS + steps * LOTS_ADJUSTMENT_STEP`。
          * 将当前期货持仓调整至 `min(TARGET_LOTS, FUTURES_MAX_LOTS)` 手。
          * *示例：若 `ENTRY_PRICE`=3000，`CLOSE_PRICE`=2975，则 `PRICE_DIFF`=-25。`steps = floor(25/10) = 2`。目标手数为 `100 + 2 * 10 = 120` 手。*
      * **情况C：价格在初始区间内 (`0 <= PRICE_DIFF <= PRICE_GRID_INTERVAL`)**
          * 持仓保持 `FUTURES_INITIAL_LOTS` 手不变。

### 步骤三：出场条件 (Exit Conditions)

1.  **最终平仓日**：策略持续运行，直到所持有的看跌期权合约到期日为止。
2.  **清仓操作**：在期权到期日当天，将所有剩余的期货持仓和期权持仓，以收盘价全部平仓，策略结束。

## 5\. 数据要求 (Data Requirements)

本策略回测需要两种数据文件，均以 CSV 格式存储。

### 5.1 期货日线数据

  * **文件名**: `futures_data.csv` (建议)
  * **内容**: 包含目标合约从策略开始日期前的某段时间，至合约到期日的全部日线行情数据。

**表头和格式规范:**

| 表头 (Column) | 数据类型 (Type) | 说明 (Description) |
| :--- | :--- | :--- |
| `datetime` | `string` | 日期，格式为 `YYYY-MM-DD` |
| `open` | `float` | 开盘价 |
| `high` | `float` | 最高价 |
| `low` | `float` | 最低价 |
| `close` | `float` | 收盘价 |
| `volume` | `integer` | 成交量 |

**数据示例 (`futures_data.csv`):**

```csv
datetime,open,high,low,close,volume
2025-05-01,3005.0,3025.0,2998.0,3010.0,1500000
2025-05-02,3011.0,3030.0,3008.0,3028.0,1650000
...
```

### 5.2 期权日线数据

  * **文件名**: `options_data.csv` (建议)
  * **内容**: 包含与目标期货合约对应的**所有看跌期权**的日线行情数据。数据量会较大，但这是为了能在任意一天寻找到我们需要的特定行权价期权。

**表头和格式规范:**

| 表头 (Column) | 数据类型 (Type) | 说明 (Description) |
| :--- | :--- | :--- |
| `datetime` | `string` | 日期，格式为 `YYYY-MM-DD` |
| `underlying_price` | `float` | 当天标的期货的收盘价 |
| `strike_price` | `float` | 该期权的行权价 |
| `option_type` | `string` | 期权类型，此处应为 'put' |
| `close` | `float` | 该期权的收盘价 (即权利金) |

**数据示例 (`options_data.csv`):**

```csv
datetime,underlying_price,strike_price,option_type,close
2025-05-01,3010.0,2800.0,put,15.5
2025-05-01,3010.0,2850.0,put,20.2
2025-05-01,3010.0,2900.0,put,28.0
2025-05-02,3028.0,2800.0,put,12.1
2025-05-02,3028.0,2850.0,put,16.8
...
```

## 6\. 算法流程详解

本策略的算法流程严格按照时间顺序执行，无未来函数。

1.  **初始化 (在 `START_DATE`)**

    1.  读取 `futures_data.csv`，找到 `START_DATE` 的开盘价 `ENTRY_PRICE`。
    2.  计算目标期权行权价 `TARGET_STRIKE = ENTRY_PRICE + STRIKE_PRICE_OFFSET`。
    3.  在 `options_data.csv` 中筛选出 `datetime` 为 `START_DATE` 且 `option_type` 为 `put` 的所有期权。
    4.  从筛选结果中，找到 `strike_price` 与 `TARGET_STRIKE` 最接近的期权，记录其价格并“买入”`PUT_OPTION_LOTS` 手。
    5.  “买入” `FUTURES_INITIAL_LOTS` 手期货，成本为 `ENTRY_PRICE`。

2.  **每日循环 (从 `START_DATE`+1天 到期权到期日)**

    1.  获取当天期货收盘价 `CLOSE_PRICE`。
    2.  计算价格偏离 `PRICE_DIFF = CLOSE_PRICE - ENTRY_PRICE`。
    3.  根据 `PRICE_DIFF` 和 `PRICE_GRID_INTERVAL` 计算需要调整的档位 `steps`。
    4.  根据文档第三节的公式计算目标期货手数 `TARGET_LOTS`。
    5.  对比当前持仓与 `TARGET_LOTS`，产生交易信号（买入/卖出/不动），调整期货仓位。
    6.  期权仓位始终保持不变。

3.  **最终清算 (在期权到期日)**

    1.  获取到期日当天期货和期权的收盘价。
    2.  将所有剩余的期货和期权头寸全部平仓。
    3.  计算总盈亏，策略结束。

## 7\. 回测注意事项

  * **交易成本**: 回测时必须考虑期货和期权的交易手续费及滑点，特别是期货仓位可能频繁调整，成本累计不容忽视。
  * **保证金计算**: 需监控整个策略的保证金占用，尤其是在下跌加仓导致期货头寸增加时，确保不会触发爆仓。
  * **期权流动性**: 虚值期权的流动性可能较差。在回测中，可以假设能以收盘价成交，但在实盘中需要考虑买卖价差。
  * **Theta 损耗**: 策略的固定成本来自期权的时间价值流逝。应在回测结果中单独分析期权部分的P/L，以评估其“保险成本”。