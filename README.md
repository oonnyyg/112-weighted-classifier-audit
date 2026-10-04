# weighted-classifier-audit

带权二分类模型质量评估工具（仅 Python 标准库）。不训练模型：在验证集上按业务成本选择阈值，冻结规则后独立评估测试集。

## 输入

验证集与测试集各为一个 CSV，必需列：

| 列名 | 含义 |
| --- | --- |
| `id` | 记录唯一标识（两集之间不得重复） |
| `label` | 真实标签，0 或 1 |
| `probability` | 预测概率，须为 [0,1] 内有限值 |
| `weight` | 样本正权重，非负有限值 |

两集均不能为空；标识重复、概率越界或权重非法会直接报错退出（退出码 2）。

## 用法

```bash
python -m weighted_audit \
  --validation val.csv --test test.csv \
  --fp-cost 1.0 --fn-cost 5.0 \
  --summary-json summary.json --predictions-csv predictions.csv
```

`--fp-cost` / `--fn-cost` 为误报、漏报成本，均须非负且不能同时为零。

## 阈值选择口径

- 候选阈值：`0`、验证集中所有不同概率值、以及特殊候选「全部判负」（视为高于一切数值阈值）。
- 判定规则：概率 `>=` 阈值判为正。
- 目标：最小化验证集 `fp_cost * 误报权重和 + fn_cost * 漏报权重和`。
- 并列时选择更高阈值；「全部判负」高于所有数值候选。
- 阈值在验证集上选定后冻结，测试集不参与任何调参。

## 指标口径（测试集，全部按权重加权）

- 混淆矩阵：TP/FP/TN/FN 均为权重和。
- 精确率 `TP / (TP + FP)`，召回率 `TP / (TP + FN)`；分母为零时输出 `null`。
- F1：`2TP / (2TP + FP + FN)`；分母为零时输出 `null` 并在 `f1_note` 说明原因。
- ROC AUC：逐正负样本对按权重乘积计分，正样本概率更高计满分，同分计半分；测试集只含一种真实类别（或样本对总权重为零）时输出 `null` 并在 `roc_auc_note` 说明。
- Brier 均方误差：`sum(w * (p - y)^2) / sum(w)`，使用原始概率，不使用二值化预测。
- `total_cost`：测试集逐条成本之和，与 CSV 明细 `cost` 列合计一致。

## 输出

- JSON 汇总：成本配置、选中阈值（全部判负时 `threshold` 为 `null` 且 `threshold_rule` 为 `all_negative`）、验证集成本、测试集全部指标。
- CSV 明细：`id,label,probability,weight,predicted,cost` 逐条预测与成本。

## 测试

```bash
python -m unittest discover -s tests
```

覆盖：输入校验、标识冲突、并列阈值（含全部判负最高）、概率同分 AUC 计半分、单类别 AUC 为空、F1 分母为零、端到端汇总与明细对账。
