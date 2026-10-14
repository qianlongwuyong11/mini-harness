# 最后阶段：基准、重复运行与验收

## 生成与冻结

在项目目录 PowerShell 执行，paper-id 从 UI 论文库或目录查询获得：

```powershell
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe -m mini_harness.evaluation.benchmark prepare --paper-id <实际论文ID> --output .mini-harness/evals/benchmarks/v1
```

生成 regression.json、heldout.json 和 review-manifest.json。已有输出目录会拒绝覆盖，更新语料须创建新版本。manifest 记录数据集摘要和未完成审定项。源论文更新后必须重新绑定，不能修改旧报告。

人工逐条审阅：预期行为是否合理、参考答案是否能由原文支撑、gold chunk/page 是否准确、拒答是否符合题意。补全 reference_answer/key_points/gold_evidence 后重新记录摘要并冻结。缺少 gold 的 Recall 不计零。heldout 草稿必须由人工独立审定或替换，且不可用于调参。

## 校准与执行

先选开发集由人工给出通过/失败样例，对照 judge 的分数和理由，选择阈值并记录误判。把确认后的阈值写入 judge.json，冻结模型、rubric、阈值、语料与数据集版本。没有人工校准时保持 incomplete，不能默认使用某个分数宣称质量达标。

进入 UI 评测中心：分别运行相同 regression.json 至少两次，再运行独立 heldout.json。每次选择对应 API 数据授权；不会自动外发或自动消费队列。每个回归运行是新 run，不是 retry。每例最多 6 step、120 秒、20000 token；同步请求只能协作超时，可能发生最后一次响应越过 token 预算。Judge 另有 max_calls 和输出预算。

记录环境（Git commit、依赖版本、模型/ES配置、冷/热启动、是否同时导入或聊天）。暂停竞争任务后比较。对不一致 case 检查 step 输入、工具错误、实际注入证据和引用；在 UI 保存人工复核，不覆盖自动结果。

```powershell
.\.venv\Scripts\python.exe -m mini_harness.evaluation.benchmark report --regression <run1> <run2> --heldout <run3>
```

报告检查重复 run、数据集/config、split、问题重叠、完成状态、自动质量与人工复核。缺少运行、阈值或人工复核都不会通过。中位耗时显示有效样本数；小样本不作显著性判断。文字不重复不足以证明 heldout 独立性，仍需人工审核。

## 当前交付与未完成验收

已提供可复现草稿生成、66f807 场景覆盖、重复运行报告、复核门槛与操作手册。自动测试验证这些机制，不证明真实 Agent 质量。

尚需人工审定标准答案与阈值，并授权选定数据集的真实模型运行。当前没有生成虚假的真实分数或“人工已通过”标记。工程交付与业务质量验收分别记录。
