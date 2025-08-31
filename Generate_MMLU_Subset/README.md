## 项目简介

该项目从大型 `anchor_points.ipynb` 中抽取核心逻辑，重构为可维护的 Python 模块（位于 `scripts/`），并提供一键脚本 `run_pipeline.py` 用于生成多种方法的 MMLU 中文子集、评估并导出结果。这样能更好地管理与复用代码，避免在体积很大的笔记本中修改困难。

## 环境与安装

- Python 版本：建议 3.10+（如 3.12.11）
- 安装依赖：
```bash
pip install -r requirements.txt
```

## 数据准备

放置在项目根目录或按需通过参数指定：
- 评测结果数据：`data/lb.pickle`
- 中文题库：`mmlu_ZH-CN.csv`
- IRT 参数：`data/irt_model/best_parameters.json`

注意：`mmlu_ZH-CN.csv` 行数需与英文侧 MMLU 项目数一致，否则脚本会报错提示。

## 快速运行

在项目根目录运行：
```bash
python run_pipeline.py
```

默认会：
- 生成 3 种方法（Bucket/IRT/Matrix）各 2 个子集（共 6 个），每个子集默认 300 题；
- 评估每种方法的代表性与一致性；
- 将子集导出到 `mmlu_subsets_csv/` 目录；
- 输出评估摘要到根目录 `mmlu_ZH-CN_subset_summary.json`。

### 命令行参数

```bash
python run_pipeline.py \
  --lb_pickle data/lb.pickle \
  --mmlu_csv mmlu_ZH-CN.csv \
  --irt_model_dir data/irt_model/ \
  --subset_size 300 \
  --out_dir_csv mmlu_subsets_csv \
  --save_summary mmlu_ZH-CN_subset_summary.json \
  --random_state 42
```

- `--lb_pickle`：Leaderboard pickle 路径（默认 `data/lb.pickle`）
- `--mmlu_csv`：中文题库 CSV 路径（默认 `mmlu_ZH-CN.csv`）
- `--irt_model_dir`：IRT 参数目录（需包含 `best_parameters.json`）
- `--subset_size`：每个子集的题目数（默认 300）
- `--out_dir_csv`：子集 CSV 输出目录（默认 `mmlu_subsets_csv`）
- `--save_summary`：评估摘要 JSON 文件名（默认根目录 `mmlu_ZH-CN_subset_summary.json`）
- `--random_state`：随机种子（默认 42）

## 输出说明

### 子集 CSV（英文列名）
路径：`mmlu_subsets_csv/`
- `mmlu_ZH-CN_subset1_bucket.csv`
- `mmlu_ZH-CN_subset2_bucket.csv`
- `mmlu_ZH-CN_subset1_irt.csv`
- `mmlu_ZH-CN_subset2_irt.csv`
- `mmlu_ZH-CN_subset1_matrix.csv`
- `mmlu_ZH-CN_subset2_matrix.csv`

列结构与源 `mmlu_ZH-CN.csv` 保持一致：`ID, Question, A, B, C, D, Answer, Subject`

排序策略：按 `Subject` 升序，同一 `Subject` 内按 `ID` 升序（稳定排序）。

### 评估摘要 JSON（英文键）
路径：根目录 `mmlu_ZH-CN_subset_summary.json`
- `best_method`：综合评分最高的方法（`bucket`/`irt`/`matrix`）
- `method_scores`：各方法的一致性评分、代表性评分、综合评分
- `method_consistency`：各方法两子集之间的差异、相关等指标
- `method_representativeness`：各方法与完整有效 MMLU 的相关、均值差、标准差差等指标
- `all_subsets`：六个子集的题目索引（相对 `mmlu_ZH-CN.csv` 的行号）
- `export_paths`：六个子集 CSV 的输出路径

## 顶层脚本：run_pipeline.py

`run_pipeline.py` 将完整流程自动化：
1) 加载评测数据与中文题库；
2) 过滤指定排除科目；
3) 用三种方法分别生成两个不重叠子集；
4) 评估与完整 MMLU 的代表性、一致性；
5) 导出 6 个子集 CSV 和摘要 JSON。

你可通过修改命令行参数控制子集大小、随机种子、输入/输出路径等。

## 模块说明（scripts/）

### scripts/utils.py
- `scenarios`：各评测场景与子场景定义
- `prepare_data(scenarios, data)`：构建 `scenarios_position` 与 `subscenarios_position`
- `create_responses(scenarios, data)`：将各场景拼接为统一 0/1 响应矩阵 `Y`

### scripts/data_io.py
- `load_leaderboard_pickle(path)`：加载 `lb.pickle`
- `load_mmlu_zh_cn_csv(path)`：加载中文题库 CSV
- `export_subset_sorted(df, indices, path, cols)`：导出子集为 CSV（排序规则见上）
- `save_json(path, payload)`：保存 JSON（UTF-8，`ensure_ascii=False`）

### scripts/subset_methods.py
- `create_difficulty_bucket_subsets(item_difficulties, valid_indices, subset_size, num_buckets=5, random_state=42)`：
  基于题目平均正确率分位桶的分层采样，生成两个不重叠子集
- `create_irt_clustering_subsets(valid_indices, subset_size, df_cn, excluded_subjects, irt_model_dir='data/irt_model/', random_state=42)`：
  基于 IRT 参数（区分度/难度）K-Means 聚类；每簇取最近的 2 个有效题目，均分到两个子集
- `create_response_matrix_clustering_subsets(Y_train_mmlu, valid_indices, subset_size, df_cn, excluded_subjects, random_state=42)`：
  基于模型-题目正误矩阵 K-Means 聚类；每簇取 2 个有效题目，均分到两个子集

### scripts/evaluation.py
- `compute_full_accuracy(Y, valid_indices)`：完整（过滤后的有效题目）MMLU 准确率（逐模型）
- `compute_subset_accuracies(Y, subsets)`：各子集准确率（逐模型）
- `compute_method_consistency(subset_accuracies)`：方法内两个子集之间一致性指标
- `compute_method_representativeness(subset_accuracies, full_acc)`：相对完整 MMLU 的代表性指标
- `score_methods(method_consistency, method_repr, weights=None)`：综合打分

### scripts/anchors.py（可选）
- `compute_anchor_points_and_weights(...)`：
  - clustering=`'correct.'`：按题目正误向量聚类
  - clustering=`'irt'`：按 IRT 特征聚类
  - 计算锚点题与其权重
- `evaluate_anchor_estimation(...)`：基于锚点估计的误差（平均绝对误差）

### scripts/multi_seed.py（可选）
- `run_multi_seed(seeds, Y, Y_train_mmlu, valid_indices, item_acc_valid, df_cn, excluded_subjects, irt_model_dir='data/irt_model/', subset_size=300)`：
  - 依次对多个随机种子重复 3 种方法的子集生成与评估
  - 返回 `per_seed_df` 与 `agg_df`，便于进一步可视化/统计

### scripts/plots.py（可选）
- `generate_charts(per_seed_csv, aggregate_csv, summary_json, out_dir='subset_analysis_charts')`：
  - 输入多随机种子评估的 CSV/JSON，输出对比图表（PNG）

### scripts/weights.py（可选）
- `compute_balance_weights(Y, scenarios_position, subscenarios_position, scenarios)`：
  - 计算平衡权重，使多子场景按“科目均衡”聚合（通用，但当前主要用于 MMLU）

## 常见修改点

- 更改排除科目：编辑 `run_pipeline.py` 中的 `excluded_subjects` 集合
- 更改子集规模或随机种子：改 `--subset_size` 或 `--random_state`
- 仅使用“训练模型集合”进行 Matrix 聚类：将传入 `create_response_matrix_clustering_subsets` 的矩阵从 `Y_mmlu` 换为你的训练切片（如 `Y_train_mmlu`）
- 增加/替换评估指标：在 `scripts/evaluation.py` 扩展对应函数

## 注意事项

- 子集 CSV/JSON 中的字段均使用英文；图表由 `scripts/plots.py` 生成时也使用英文注释/图例。
- 若你在 Jupyter 笔记本中使用本项目，建议仅做可视化与结果检查，将逻辑调用切换为 `scripts.*` 里的函数，便于维护。

## 许可证与致谢

- 本仓库在整理自内部分析与 `anchor_points.ipynb` 的基础上模块化重构，感谢相关项目与数据来源。
