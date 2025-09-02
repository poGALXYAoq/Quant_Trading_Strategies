## 使用说明（GBDT_futures）

本目录提供三类脚本，按“研究/调参 → 贴近预测日训练 → 线上推断”分层：

- `model.py`：研究/调参与回测（固定日期段）。输出在 `results/`（用于研究，不用于部署）。
- `train_refit.py`：给定 `cut_date` 的贴近实盘训练脚本。输出在 `results_deploy/cut_<cut_date>/`（用于部署）。
- `predict_next.py`：加载部署目录中的模型与工件，输出最近 N 天或指定日期范围的“次日价差/次日收盘价”预测。

### 1) 数据与标签

- 必备日期列：`date`
- 默认价格列（与代码一致）：`期货收盘价(活跃合约):阴极铜`
- 标签：`y_t = P_{t+1} - P_t`（预测“明日涨跌额”）。若要得到“明日收盘价”，为 `P_t + y_hat`。

注意：线下评估某日 `t` 的预测需要数据包含 `t+1` 的真实价；实盘推断不需要。

### 2) 研究/调参（可选）

运行 `model.py` 会在 `results/` 下产生研究用的指标与模型等工件：
- `results/models/xgb_model_es.json`、`results/models/xgb_model_final.json`
- `results/predictions/{train,valid,test}.csv`
- `results/artifacts/{feature_names.csv,feature_medians.csv,feature_importance.csv,metrics.json}`

当调参稳定后，可将最终参数写成 JSON 文件并在生产训练中复用（也可直接使用默认参数）。

### 3) 贴近预测日训练（生产训练）

- 打开 `train_refit.py`，在顶部 `USER_CONFIG` 设置 `data_path/cut_date` 等参数。
- 直接运行。输出目录（已在 `.gitignore` 忽略）：
    - `results_deploy/cut_<cut_date>/models/xgb_model_final.json`
    - `results_deploy/cut_<cut_date>/artifacts/feature_names.csv`
    - `results_deploy/cut_<cut_date>/artifacts/feature_medians.csv`
    - `results_deploy/cut_<cut_date>/artifacts/{metrics.json,train_meta.json}`
    - `results_deploy/cut_<cut_date>/predictions/{train.csv,valid.csv}`

关键配置说明：
- `PRICE_COL_DEFAULT`：默认价格列（如需更换请在脚本顶部修改）。
- `start_date`：训练起始日期（可为空）。
- `cut_date`：验证窗右端，模型贴近该日训练，并用于预测 `cut_date+1`。
- `valid_days`/`embargo_days`：验证窗长度与隔离天数。
- `use_time_decay`/`half_life_days`：时间衰减样本权重设置。

### 4) 线上推断（每日/区间出信号）

- 打开 `predict_next.py`，在顶部 `USER_CONFIG` 设置 `data_path`。
- `deploy_root` 留空时，脚本会在 `results_deploy/` 下自动选择最新的 `cut_*` 目录作为加载路径；也可显式填入某个 `results_deploy/cut_YYYY-MM-DD` 路径。
- 两种选择方式：
  - 方式A：设置 `days > 0`，对最近 N 天做预测；
  - 方式B：设置 `start_date/end_date`，对闭区间内的有效交易日做预测（端点若非交易日将自动向内收敛）。

输出示例（控制台 JSON）：
```json
{
  "deploy_root": ".../results_deploy/cut_2025-08-25",
  "effective_start": "2025-08-15",
  "effective_end": "2025-09-01",
  "predictions": [
    {"date": "2025-08-29", "close": 70000.0, "predicted_delta": 120.5, "predicted_next_close": 70120.5}
  ]
}
```

### 5) 依赖与环境

建议 Python 3.10/3.11，安装：
```bash
pip install -U pandas numpy scikit-learn xgboost optuna
```
说明：optuna 仅在 `model.py` 开启调参时需要。

### 6) .gitignore 与输出

- `GBDT_futures/results/` 与 `GBDT_futures/results_deploy/` 已在仓库根 `.gitignore` 中忽略。
- 若需分享结果，请手动挑选关键 JSON/CSV 片段粘贴到文档，或放置到 `samples/` 之类的小体量目录中。

### 7) 常见问题（FAQ）

- Q: 部署与研究的输出会不会混在一起？
  - A: 不会。回测/研究输出在 `results/`；用于部署的训练输出在 `results_deploy/cut_<cut_date>/`，与回测分离。

- Q: 只有到 `T` 的数据，能预测 `T+1` 吗？
  - A: 可以。推断只需 `T` 的特征；线下评估 `T` 的预测需数据里有 `T+1` 的真实价。

- Q: 为什么要保留验证窗？
  - A: 用于早停与时间衰减强度的校准，避免用测试集做决策而产生泄漏。

- Q: 我需要每天重训吗？
  - A: 可按周/月重训，日内只推断；若市场漂移明显，可提高重训频率。
