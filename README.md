
# Adaptive-Agent-Workflow

A multi-agent orchestration system that **dynamically determines its own structure** — tree depth, branching, and per-agent permissions — based on task complexity and a structural certainty proxy, instead of using a fixed pipeline.

## Why This Exists

Most LLM agent frameworks use a flat chain or a fixed-depth tree. This project treats the agent topology itself as a runtime output: simple tasks get a shallow tree, complex or uncertain tasks get deeper orchestration with tighter permission scoping. Failed branches are rebuilt locally without disrupting the rest of the tree.

## How It Works

```
Instruction → Planner (S1) → Task DAG + Confidence + Depth
                                  │
                            Root Agent  — activates tasks by dependency
                                  │
                         Orchestration Agents — recursive decomposition
                                  │
                          Execution Agents — tool calls
                                  │
                         Deviation Detection + Replanner 
```

- **1** parses the instruction, estimates complexity from the environment, generates a task DAG, and decides tree depth
- **2** instantiates agents layer by layer; each child's permissions are strictly narrower than its parent's
- **3** selects and calls tools with adaptive sampling parameters
- **4** detects failures and replans the failed task using its current error context

## Demo

Uses IEEE 14-bus + pandapower as a simulation environment. The architecture is domain-agnostic.

```bash
git clone https://github.com/<your-username>/Adaptive-Agent-Workflow.git
cd Adaptive-Agent-Workflow
pip install -r requirements.txt
pytest tests/ -v
```

`main.py` 会先把测试网络中的 Gen 0 电压设为 0.98 p.u.，制造可复现的 Bus 14 低电压，
然后规划、仿真、实际执行调节并检查全网约束。`simulate_action` 只在副本上运行，
只有 `set_gen_voltage` 等执行工具才会修改测试网络。

Run with a real LLM (required for `main.py`):
```bash
export LLM_API_KEY="your-key"
# 默认使用 https://api.deepseek.com 和 deepseek-flash；也可显式设置：
export LLM_BASE_URL="https://api.deepseek.com"
export LLM_MODEL="deepseek-flash"
python main.py
```

真实客户端通过 DeepSeek 的 `response_format: {"type":"json_object"}` 直接请求 JSON。
短 JSON 任务会关闭模型的思考模式，避免思考 token 占用输出额度。
可用 `LLM_TIMEOUT`（默认 30 秒）和 `LLM_MAX_RETRIES`（默认 2 次）调整超时与重试。
启动日志会显示实际使用的 `base_url` 和 `model`（不会显示密钥）。若请求失败，
401 通常检查密钥，400 检查模型与请求参数，429 检查限额或并发，网络超时检查连接。
规划阶段的 API 错误会直接报出原因，不会被当作空任务列表并继续执行。
任务数量与依赖边由 LLM 的 JSON 响应决定，代码只校验缺失依赖、重复 ID 和环。
固定响应仅位于 `tests/fixtures_llm.py`，用于不调用 API 的测试。
重规划只使用本次任务的失败上下文，不保存或检索反例库。
当前的 C 是任务图结构启发式分数，尚不是交底草稿所述的 token 概率差和注意力加权指标。
DeepSeek 这类接口没有提供实现该注意力计算所需的内部权重。
`finish_reason=length` 表示本次输出达到 `max_tokens`，与输入上下文窗口耗尽是不同的问题；
编排修正请求现使用更短的提示词与 2048 个输出 token。

## Project Structure

```
agents/     Planner, Root, Orchestration, Execution, Deviation, Replanner
llm/        Real LLM client and prompt templates
grid/       Simulation environment, complexity estimation, tool library
config/     Thresholds and permission templates
tests/      Unit + end-to-end tests
```

## License

MIT
```
