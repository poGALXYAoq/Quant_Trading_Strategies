## 使用说明（GBDT_futures）

本目录提供三类脚本，按“研究/调参 → 贴近预测日训练 → 线上推断”分层：

- `model.py`：研究/调参与回测（固定日期段）。输出在 `results/`（用于研究，不用于部署）。
- `train_refit.py`：给定 `cut_date` 的临近实盘训练脚本。输出在 `results_deploy/cut_<cut_date>/`（用于部署）。
- `predict_next.py`：加载部署目录中的模型与工件，输出最近 N 天的“次日价差/次日收盘价”预测。

### 1) 数据与标签

- 日期列：`date`
- 默认价格列：`加权平均价(主力合约):沪铜(9:00-15:00)`
- 标签：`y_t = P_{t+1} - P_t`（预测“明日涨跌额”）。若要得到“明日收盘价”，为 `P_t + y_hat`。

注意：线下评估某日 `t` 的预测需要数据包含 `t+1` 的真实价；实盘推断不需要。

### 2) 研究/调参（可选）

运行 `model.py` 会在 `results/` 下产生研究用的指标与模型等工件。调参稳定后，可将最终参数写成 JSON 文件（例如从 `results/artifacts/metrics.json` 的 `used_params` 提取），供生产训练使用。

### 3) 贴近预测日训练（生产训练）

- 打开 `train_refit.py`，在顶部 `USER_CONFIG` 设置 `data_path/cut_date` 等参数，确保 `USE_CLI_ARGS = False`（默认）。
- 直接运行。输出目录：
    - `results_deploy/cut_<cut_date>/models/xgb_model_final.json`
    - `results_deploy/cut_<cut_date>/artifacts/feature_names.csv`
    - `results_deploy/cut_<cut_date>/artifacts/feature_medians.csv`
    - `results_deploy/cut_<cut_date>/artifacts/train_meta.json`



### 4) 线上推断（每日出信号）
  - 打开 `predict_next.py`，在顶部 `USER_CONFIG` 设置 `data_path`。`deploy_root` 留空时，脚本会在 `results_deploy/` 下自动选择最新的 `cut_*` 目录作为加载路径；也可显式填入某个 `results_deploy/cut_YYYY-MM-DD` 路径。
  - `days` 控制对最近 N 天做次日预测（默认 1）。


### 5) 常见问题

- Q: 部署与研究的输出会不会混在一起？
  - A: 不会。回测/研究输出在 `results/`；用于部署的训练输出在 `results_deploy/cut_<cut_date>/`，与回测分离。

- Q: 只有到 `T` 的数据，能预测 `T+1` 吗？
  - A: 可以。推断只需 `T` 的特征；线下评估 `T` 的预测需数据里有 `T+1` 的真实价。

- Q: 为什么要保留验证窗？
  - A: 用于早停与时间衰减强度的校准，避免用测试集做决策而产生泄漏。

- Q: 我需要每天重训吗？
  - A: 可按周/月重训，日内只推断；若市场漂移明显，可提高重训频率。


