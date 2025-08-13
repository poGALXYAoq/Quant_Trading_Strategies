# 量化交易策略实验仓库

## 关于本仓库
- 用于存放与回测多类量化交易策略的代码与文档。
- 每个策略在独立目录内维护，互不依赖，便于单独运行与扩展。
- 提供基础的数据准备脚本（ETL）与可视化输出。

## 仓库结构
```
.
├── Calendar_Spread_Strategy/              # 期权跨期价差（Calendar Spread）回测
│   ├── README.md
│   ├── backtest.py                         # 标准回测（k,N 全参数格点）
│   └── backtest_with_contract_month_filter.py  # 带合约月份过滤的回测
│
├── Protected_Futures_Reverse_Grid_Strategy/   # 受保护的期货逆势网格策略
│   ├── README.MD
│   └── backtest.py
│
├── ETL/
│   ├── CSS/                               # Calendar Spread 所需的数据整理脚本
│   │   ├── prepare_data.py                # 生成 master_option_data_RB.csv（“大宽表”）
│   │   ├── concatenate_csv.py / clean_csv.py / excel_to_csv.py / ...
│   └── PFRG/                              # 受保护网格策略的数据整理脚本
│       ├── process_futures_data.py        # 期货数据 -> futures_data.csv
│       └── process_options_data.py        # 期权数据 -> options_data.csv（含标的价）
│
├── LICENSE
└── README.md                              # 本文件
```

## 运行环境
- Python 3.9+（建议 3.10/3.11）
- 依赖（按需安装）：
```bash
pip install -U pandas numpy matplotlib seaborn joblib tqdm plotly
```

## 数据准备（概要）
- Calendar Spread（期权跨期价差）：
  - 目标输入文件：`data/RB/master_option_data_RB.csv`
  - 可使用 `ETL/CSS/prepare_data.py` 将多源数据（期权日行情、期权静态信息、标的期货日线）合并为“大宽表”。
  - 详细字段与流程见 `Calendar_Spread_Strategy/README.md`。
- 受保护的期货逆势网格策略：
  - 目标输入文件：
    - `Protected_Futures_Reverse_Grid_Strategy/data/<symbol>/futures_data.csv`
    - `Protected_Futures_Reverse_Grid_Strategy/data/<symbol>/options_data.csv`
  - 可使用 `ETL/PFRG/process_futures_data.py` 与 `ETL/PFRG/process_options_data.py` 生成上述文件。
  - 详细字段与流程见 `Protected_Futures_Reverse_Grid_Strategy/README.MD`。

提示：部分脚本默认使用 Windows 风格的路径分隔符（例如 `data\RB\...`）。在 Linux/MacOS 环境下请改为正斜杠（例如 `data/RB/...`）或在脚本内调整为 `os.path.join`。

## 快速开始
- 运行 Calendar Spread 回测：
```bash
python Calendar_Spread_Strategy/backtest.py
# 或（仅交易特定合约月份，如 1/5/10 月）：
python Calendar_Spread_Strategy/backtest_with_contract_month_filter.py
```
- 运行受保护网格策略回测：
```bash
python Protected_Futures_Reverse_Grid_Strategy/backtest.py
```

## 策略清单
| 策略名称 | 简要描述 | 路径 |
| :-- | :-- | :-- |
| 受保护的期货逆势网格策略 | 以期货多头为基，买入虚值看跌保护；按网格对期货头寸逆势加减仓；期权持仓持有至到期 | `Protected_Futures_Reverse_Grid_Strategy/` |
| 期权跨期价差（Calendar Spread） | 近月卖出、远月买入，同步选择行权价，参数网格遍历 k 与 N | `Calendar_Spread_Strategy/` |

## 免责声明
- 本仓库所有内容仅供学习与研究，不构成任何投资建议。
- 金融市场风险较高，历史回测不代表未来表现。任何由此产生的实际交易风险自负。
