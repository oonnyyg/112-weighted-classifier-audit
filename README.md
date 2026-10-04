# weighted-classifier-audit

二分类模型的带权质量评估工具（纯 Python 标准库，无需训练模型）。
在验证集上按业务成本选择阈值，冻结后独立评估测试集。

## 输入

验证集与测试集各为一个 CSV，列：`id,label,probability,weight`。

- `id`：记录唯一标识；两份数据不得共享任何 id。
- `label`：真实零一标签（0 或 1）。
- `probability`：预测概率，必须有限且位于 [0, 1]。
- `weight`：正权重（有限且大于 0）。
- 两份数据均不能为空，各自 id 不得重复。

## 用法

```bash
python -m weighted_audit.cli \
  --validation val.csv --test test.csv \
  --fp-cost 1.0 --fn-cost 5.0 \
  --summary summary.json --details details.csv
```

安装后也可使用 `weighted-audit` 命令。`--fp-cost` / `--fn-cost` 非负且不能同时为零。

## 阈值选择口径

- 候选阈值：`0.0`、验证集中所有不同概率、以及特殊候选“全部判负”（视为 +inf）。
- 判定规则：概率 >= 阈值判为正。
- 目标：最小化验证集加权成本 `fp_cost * FP权重 + fn_cost * FN权重`。
- 并列时选更高阈值；“全部判负”高于一切其他候选。
- 阈值在验证集上选定后冻结，直接应用于测试集，不再使用测试标签调参。

## 指标口径

以下指标均在测试集上按冻结阈值计算，所有混淆矩阵元素为权重和：

- 精确率 precision = TP / (TP + FP)，分母为零时为 null。
- 召回率 recall = TP / (TP + FN)，分母为零时为 null。
- F1 = 2TP / (2TP + FP + FN)；分母为零时输出 null 并在 `f1_note` 说明原因。
- 加权 ROC AUC：遍历正负样本对，按权重乘积计分，正样本概率更高计 1、
  同分计 0.5，归一化到 [0, 1]；测试集只有一种真实类别时输出 null 并说明。
- 加权 Brier 均方误差：`sum(w * (p - y)^2) / sum(w)`，使用原始概率，
  不以二值化预测代替。

## 输出

- JSON 汇总：阈值（全部判负时 `threshold` 为 null 且 `threshold_all_negative` 为 true）、
  成本配置、验证集成本、测试集混淆矩阵与各指标、测试集总成本。
- CSV 明细：逐条 `id,label,probability,weight,predicted,outcome,cost`；
  明细 `cost` 之和等于汇总中的 `total_cost`，`outcome` 计数与混淆矩阵一致。

## 测试

```bash
python -m unittest discover -s tests
```

覆盖并列阈值、概率同分 AUC、单类别测试集、全部判负、输入校验等场景。
