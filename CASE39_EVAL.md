# case39 统一内核评测（case39-v2）

本协议在正式实验前冻结。不能根据正式实验结果修改提示词或阈值；修改需递增 `CASE39_PROMPT_VERSION`，使用新输出文件。CSV 同时保存 HEAD 和源代码 SHA256，未提交修改也纳入续跑一致性检查。旧版本 CSV 不可与本版本合并。本次只运行脚本化 LLM 的离线测试，不产生正式实验结论。

## 已核实的旧实现问题

1. 旧 `Case39Trial` 没有实例化 S1–S4 方法类，不能代表项目的 hierarchical。
2. 区域2先执行、区域1失败后保留区域2是评测脚本硬编码。
3. 旧目标解析只能识别单母线；本指令落到默认内部 Bus 13。
4. 旧评测的 d0/H 只用于 CSV；没有驱动权限与智能体层级。
5. 旧评测 duplicate 实际比较整份 actions JSON，而真实 RootAgent 比较重规划后的单工具调用；二者口径不一致。缺少逐来源调用与 token、限幅指标。
6. 旧脚本每轮重写文件，无 resume，方法顺序固定。已有预检和 retryable 异常，但批次没有暂停/续跑协议；预检 client 与实际 trial client 也不同。

另外：旧真实链路的内部联合仿真重试、格式修复和 S4 重试会相乘；不同来源 temperature/max_tokens 不同；客户端截断后自动增大 max_tokens。v2 执行路径统一上述配置，legacy 路径保留兼容性。

## 方法开关

| 方法 | S1 Planner / S2 Root | 编排深度 | 权限 | 偏差恢复 |
|---|---|---|---|---|
| hierarchical | Planner → RootAgent | `regional_tree_depth` → `max_depth` | `permission_shrink=True`，任务区域与 regional_scope 相交 | `replan_mode="local"`，Replanner 失效路径重建 |
| two_layer | 无 Planner、Root | 单 OrchestrationAgent，`max_depth=2` | 根权限，`permission_shrink=False` | 保持物理状态，重新请求整体方案 |
| two_layer_full_restart | 无 Planner、Root | 同上 | 同上 | `replan_mode="full"`，恢复初始物理断面，动作日志与预算累计 |
| hierarchical_no_shrink（可选） | Planner → RootAgent | 同 hierarchical | 根权限，关闭收缩 | 局部重规划 |
| hierarchical_full_restart（可选） | Planner → RootAgent | 同 hierarchical | 收缩 | 整体回滚，重新请求全部任务方案 |

开关在 `agents/regional.py:MethodConfig/METHOD_CONFIGS/build_method`。默认三方法；`--ablation` 才增加最后两项。重启复用原始 LLM DAG，不重新运行 Planner；清空方案缓存、任务状态，重新询问所有分支。

Planner 依据当前故障断面，按允许区域选择最低电压目标（包含并列最低），通过 `regional_scope` 调用 `compute_regional_d0`；D0 的扰动衰减范围驱动权限裁剪，区域拓扑深度驱动 H。不是将无量纲 D0 直接当层数。LLM 自己生成 tasks、设备和依赖，严格验证 ID、设备类型、依赖存在性与无环性；不补写任务，不修正设备编号，不代替 LLM 选动作。`dag_task_count` 与 `partitions` 原样反映其分解质量，单任务/全依赖链不会强制拆成两个独立区域。

`partition_dag` 合并同区域任务与所有依赖连通任务：有依赖的任务不会分到不同区域子树。分支按模型输出顺序稳定排列，分支内拓扑排序；不指定哪个区域先执行。每个分区对应真实 OrchestrationAgent；H 大于3时沿编排层继续委派，到叶子调用 LLM。Root 先收集所有未完成任务方案，经过同一联合门后按依赖顺序提交。后继动作只有前驱完成才提交。Replanner 只清除失效节点及未完成后继的方案和实例；其他已完成任务不会重新询问或执行。联合预演失败无法唯一归因时，重询所有未完成任务，保留已完成任务。一次任务内部在偏差前已执行的部分动作不会被代码改写成“剩余动作”，由该任务的 LLM 依据当前断面重新决定。

`run_case39.py` 只负责场景装配、调用工厂、采样与 CSV，不承担区域规则或动作决策。真实运行唯一使用 `RealLLMClient(fixed_limits=True)`，一个批次共用同一实例，固定 `CASE39_TEMPERATURE=0.2`、`CASE39_MAX_TOKENS=2048`，不自动扩大输出上限。两种架构最多3轮候选尝试（含初次）；不存在额外叶子修复循环。无有效初始 DAG 的规划失败立即记录，不由代码补造 DAG。预检调用不计入 trial 指标。

## 统一物理内核与信息边界

`grid/dispatch_kernel.py:DispatchKernel` 无方法名称和方法分支。统一负责目录、严格参数/设备/区域校验、动作预算、联合仿真、真实提交、不恶化检查与全网验收。

- 预演仍使用原 `grid.tools.simulate_action`，不应用隐藏限幅。每个动作前缀都预演，比较**相邻步骤**是否新增/恶化越限，最后完整联合方案必须满足全网 Goal，才可真实提交。
- ExecutionAgent 通过内核调用同一个 `PowerNetwork.set_gen_voltage`；真实设备接口应用限幅。每次真实请求计一次动作，失败、限幅和无效操作也占预算；回滚不占动作且不清空累计日志。真实操作后再次逐步检查；预算耗尽不能提交。
- success 仍要求全网电压和线路约束均满足、真实请求数不超预算、未触碰区域3。初始故障、隐藏设备参数及离线见证保持原样。生产入口不调用 `validate_scenario()`；见证仅供离线测试。
- 三方法叶子使用同一系统提示与同一 JSON 结构：`instruction`（mission/task/devices/device_type）、`current_goal`、`used_actions`、`budget`、`feedback`、`catalog`。任务指派来自 Planner；baseline 的任务是整体指令。目录裁剪是权限造成的信息差异，所有方法均可见全网越限观测，禁止区域设备不作为 hierarchical 可调用目录项。
- **严格数值保密的代价**：初始公开断面本来存在与隐藏上限相同的正常电压读数。为避免这个数值进入模型，所有方法统一把目录中处于 Goal 合格区间的电压读数标为 `within_goal_range`，越限测量保留数值；不访问隐藏参数来决定屏蔽哪些设备。重试同样采用此规则。反馈说明实测不符但不回传限幅数值。此规则牺牲了合格电压的精确观测，各方法一致；原始实测限幅仅写在离线 CSV `clipped_actions` 中。模型自己输出的数值不是先验泄露，不将其解释成隐藏上限。
- baseline 目录含区域3设备，但候选仍由相同禁止区域校验拒绝，计入 illegal/scope。权限收缩只改变可见/可调用目录，不改变物理规则。

## 运行命令

项目根目录，已安装 requirements.txt；API key 从环境读取，不能写进命令行参数或结果文件。

```sh
export LLM_MODEL='deepseek-flash'
export LLM_BASE_URL='https://api.deepseek.com'
# 在当前 shell 安全设置 LLM_API_KEY。
.venv/bin/python -m pytest -q
.venv/bin/python run_case39.py --smoke --repeats 1 --output case39_smoke_v2.csv
.venv/bin/python summarize_case39.py case39_smoke_v2.csv
.venv/bin/python run_case39.py --repeats 10 --output case39_formal_v2.csv
.venv/bin/python run_case39.py --repeats 10 --output case39_formal_v2.csv --resume
.venv/bin/python summarize_case39.py case39_formal_v2.csv
# 预算敏感性：每个预算单独的输出文件
.venv/bin/python run_case39.py --repeats 10 --max-actions 8 --output case39_budget8_v2.csv
# 可选消融
.venv/bin/python run_case39.py --repeats 10 --ablation --output case39_ablation_v2.csv
```

默认正式10轮，至少10轮；smoke 默认1轮，只允许1–2轮且 formal=0，不能作为正式结果。不同输出文件禁止混合 formal 标志。主三方法每轮循环移位；消融另行循环移位，避免改变主三方法轮转。10轮不能完全均衡3种顺序，需要完全均衡可预先决定用12轮，不能看正式结果后决定扩样。

输出已存在默认拒绝覆盖。`--resume` 跳过完成的 `(method, repeat)`，验证 schema、模型、URL、预算、formal、版本、源代码 hash 和 commit 一致。每条完成结果立即追加、flush、fsync；基础设施失败的未完成 trial 不写失败行，修复后整个未完成 trial 从初始状态重跑。服务失败前该 trial 已产生的请求费用无法挽回，不计进完成结果。CSV 是已完成 trial 的检查点，不是动作级断点续跑。

调用 `check_connection()` 后开始批次。429、全部5xx、超时或网络错误暂停退出（退出码2），保留已完成行，打印续跑说明；不算方法失败。认证/配置错误由预检报告。固定模式每次服务请求不在客户端自动重试；格式错误/空响应/截断属于方法生成失败，消耗候选轮次。macOS 若缺少可信 CA，可设置 `SSL_CERT_FILE=/etc/ssl/cert.pem`；企业代理应使用其可信 CA，不关闭 TLS 校验。

## CSV 指标

| 字段 | 定义 |
|---|---|
| success | 全网终检、预算、区域3三条件全部满足为1 |
| real_actions_used | 真实动作请求总数，包含限幅、失败、无效与随后回滚的请求 |
| wasted_actions | 去重后的真实请求索引数：限幅、无效、不恶化检查失败、动作失败、被回滚的请求 |
| duplicate_tool_calls | **第2轮及之后实际调用的工具名+完整参数，与该 trial 之前调用相同的次数**；逐动作真实提交、逐前缀 simulate_action 均统计；JSON键排序。只提案未调用不算。初轮自身重复不计重复数，但加入已见集合。此值不等于重复动作数，也不比较整份 LLM 回复 |
| illegal_tool_calls | 被边界拒绝的候选调用批次数（格式、参数、未知类型、设备索引或权限/禁止区域错误）；每个拒绝候选批次计1 |
| scope_hit / out_of_scope_proposals | 其中因权限范围或禁止区域被拒的候选批次数，两字段同义 |
| catalog_tokens | 各次叶子目录 JSON 字符数 / 4 向上取整的累计估计（非UTF-8字节数、非模型 tokenizer） |
| llm_calls | trial 内实际 LLM HTTP 请求次数，包括 Planner 和各编排叶子；不含批次预检 |
| total_tokens | API usage.total_tokens 按 trial 差分，不把共用客户端先前 trial 的用量重复累计 |
| tokens_by_source | JSON：planner/orchestration_agent 各来源的 prompt_tokens、completion_tokens、total_tokens 累计差分；服务端未提供的 token 为0，不能视为精确零成本 |
| dag_task_count / partitions | 初始 LLM DAG 节点数（baseline=0）及实际分区任务ID列表 |
| attempts | 消耗的全局候选轮数，最多3；不是 HTTP 请求数 |
| hidden_limit_triggered | 至少一次真实请求与实测电压不符为1（本场景唯一此类机制为隐藏限幅） |
| clipped_actions | JSON列表，真实日志索引、设备ID、请求值与实测值；仅结果分析使用 |
| first_attempt_deviation | 首轮出现候选/仿真/执行/规划失败为1，不仅限于隐藏限幅 |
| replanned_tasks | 实际失效重建入口的任务ID或整体重启 all；不伪称 LLM 一定按区域分解 |
| zone3_touched | 是否实际操作过区域3，而非是否提议操作 |
| tree_depth / d0 | 运行采用的层数与最大区域物理D0；baseline H=2、d0=0（未使用） |
| proposed_actions / actual_actions | 按次保存模型动作提案及真实请求日志；请求值不等于限幅后的实测值 |
| method_config | JSON：planner、permission_shrink、replan_mode、max_attempts |
| git_commit / protocol_hash / prompt_version | 提交、含工作区内容的源码指纹、提示词协议版本 |
| model / base_url / temperature / max_tokens / max_actions / formal | 复现实验设置；URL不含凭证或查询参数；formal=0为离线/冒烟 |
| error / elapsed_seconds | 最终方法失败原因与 trial 墙钟时间，包含规划/拓扑计算 |

`summarize_case39.py` 按方法、formal、模型、URL、预算、源码协议分别汇总：成功率及 Wilson 95% 区间，平均 real_actions_used/wasted_actions/llm_calls/total_tokens/catalog_tokens/illegal_tool_calls/zone3_touched；再单列 hidden_limit_triggered=1 子集。空子集显示 n=0、null，而非0%成功率。陷阱触发子集是**由方法行为决定的事后子集**，不能当作随机对照组，必须同时报告全样本触发率/样本量。

## 修改位置与原因

- `agents/planner.py`：区域多目标、区域D0/H、严格模型DAG解析。
- `agents/permission.py`：由真实任务设备推导区域与物理权限。
- `agents/regional.py`：可复用开关工厂和依赖保持分区。
- `agents/root_agent.py`：真实分层调度、联合暂存、已完成任务保留与整体重启。
- `agents/orchestration_agent.py`：统一叶子生成入口与基线整体重试。
- `agents/execution_agent.py`：所有 case39 真实动作经统一内核边界。
- `agents/replanner.py`：只重建失效路径和受影响未完成后继。
- `grid/dispatch_kernel.py`：无方法分支的共同物理内核、观测规则与指标。
- `grid/tools.py`：增加显式拒绝计数入口，simulate_action 语义不变。
- `llm/client.py`：固定上限模式、实际调用数/来源用量、全部5xx分类、拒绝URL凭证。
- `llm/prompts.py`、`config/settings.py`：冻结统一提示与请求/尝试参数。
- `run_case39.py`：移除方法实现，新增追加/续跑/轮转/冒烟/预算/元数据协议。
- `summarize_case39.py`：分组统计、Wilson区间、陷阱子集。
- `tests/test_case39_eval.py`：脚本化协议、真实类链路和共同来源断言、秘密隔离、重启与预算、权限、续跑和统计测试。

不修改 `run_compare.py`；已有工作区改动保留。`grid/scenario_case39.py` 的隐藏限幅和离线见证不变；`grid.network.py` 的真实限幅接口和 `simulate_action` 的预演语义不变。
