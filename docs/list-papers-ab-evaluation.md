# list_papers 修复前后真实 Agent 对照评测

## 当前状态

2026-10-14：用户明确授权后已完成 40 次真实 Agent 用例执行，110 次模型调用；逐条复核后，总通过率由 4/20（20%）变为 14/20（70%）。
空范围提示专项由 0/10 变为 10/10。没有 60% → 90% 的证据。语义部分为 Codex 助手按冻结标准复核，尚未经独立人工校准。
完整结果和面试表达见 [list-papers-ab-results.md](list-papers-ab-results.md)。

## 固定的实验设计

- 数据源：原 stage8-v1/regression.json，原样冻结四条用例、论文版本、预算。
- 两版各重复五轮，每轮四条用例，共 40 次用例执行；每条可包含多次模型调用。
- 基线：准备时 Git HEAD 中的 rag/plugin.py；候选：准备时工作区的该文件。其余应用源码哈希一致。
- baseline 与 candidate 每轮交替先后顺序。每轮新进程、每用例新 Session；多轮用例内部保留对话。
- 两版均使用现有 deepseek-chat 客户端和服务端默认参数。服务端模型修订/默认参数不可完全锁定，报告必须保留此限制。
- 保留原 6 steps/turn、120 秒/用例、20,000 累计 token 预算。预算按安全边界检查，最后一次请求可能超额；不承诺硬费用上限。
- 不为提高通过率删除失败、替换失败运行或更换判定标准。

## 用例判定

| 用例 | 检查重点 |
|---|---|
| scope-empty-66f807 | 提示在 UI 勾选论文；不误说库空/论文不存在；不以上传或提供 ID 替代勾选 |
| stale-id-66f807 | 用户提供 ID 仍不能绕过选择范围，提示勾选；同样不能误判库空 |
| contributions | 正常选中后能检索并提供有引用的方法总结；主要论断须由实际返回证据支持 |
| scope-cleared | 第一轮正常检索；第二轮取消选择后应重新提示勾选，不继续使用旧范围和旧证据 |

所有用例还必须正常完成、符合工具约束、无范围违规和无无效引用。
语义复核需读取答案及相关证据，并在 reviews.json 中写入 run_id、case_id、verdict、reason、reviewer。
如果由助手复核，标注助手复核，不能冒充用户人工验收或已校准的独立 Judge。
缺少复核时为 pending_review，不产生最终通过率。未执行也不当成质量失败。
执行失败或硬规则失败不能被语义复核覆盖成通过。

## 数据位置与运行

实验目录：`.mini-harness/evals/experiments/list-papers-ab-20261014/`。
其中 manifest.json 保存哈希和运行槽位；before.py/after.py 为代码快照；dataset.json/rubric.json 为冻结输入；runs/ 为独立 SQLite 队列、会话轨迹和证据；results.json 为汇总。

在项目根目录 PowerShell 中运行（须先取得本次外发授权）：

```powershell
$env:PYTHONPATH='src'
$env:PYTHONIOENCODING='utf-8'
.\.venv\Scripts\python.exe scripts/evaluate_list_papers_ab.py run --root .mini-harness/evals/experiments/list-papers-ab-20261014 --allow-model-api
```

复核后重新汇总：

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_list_papers_ab.py export --root .mini-harness/evals/experiments/list-papers-ab-20261014
```

脚本会跳过已经分配 run_id 的槽位，防止挑选较好的重跑结果。若进程被打断，应检查该槽位；不要删除 ID 重跑冒充首次执行。
不要同时启动两份实验脚本。执行器仅处理本实验明确的 run_id，不消费 UI 的其他队列。

## 最终报告口径

分别给出每条用例通过次数 / 5，以及两版通过次数 / 20；单独展示空范围提示问题和正常检索控制用例。
执行异常、预算耗尽与回答错误分别记录；报告 token 和耗时，但冷启动和缓存会影响性能比较。
这只是四条小规模回归用例的重复执行结果，不能表述为所有论文问答准确率；仍需独立能力集和人工复核才能扩展结论。
