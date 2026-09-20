
# Adaptive-Agent-Workflow

A multi-agent orchestration system that **dynamically determines its own structure** — tree depth, branching, and per-agent permissions — based on task complexity and model confidence, instead of using a fixed pipeline.

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
- **4** detects failures, locates the responsible layer, rebuilds only the failed subtree, and records the case to avoid repeating it

## Demo

Uses IEEE 14-bus + pandapower as a simulation environment. The architecture is domain-agnostic.

```bash
git clone https://github.com/<your-username>/Adaptive-Agent-Workflow.git
cd Adaptive-Agent-Workflow
pip install -r requirements.txt
python main.py        # runs normal + stress scenarios, no API key needed
pytest tests/ -v
```

Optional real LLM:
```bash
export LLM_API_KEY="your-key"
# 默认使用 https://api.deepseek.com 和 deepseek-chat；也可显式设置：
export LLM_BASE_URL="https://api.deepseek.com"
export LLM_MODEL="deepseek-chat"
python main.py
```

真实客户端通过 DeepSeek 的 `response_format: {"type":"json_object"}` 直接请求 JSON。
可用 `LLM_TIMEOUT`（默认 30 秒）和 `LLM_MAX_RETRIES`（默认 2 次）调整超时与重试。
启动日志会显示实际使用的 `base_url` 和 `model`（不会显示密钥）。若请求失败，
401 通常检查密钥，400 检查模型与请求参数，429 检查限额或并发，网络超时检查连接。
规划阶段的 API 错误会直接报出原因，不会被当作空任务列表并继续执行。

## Project Structure

```
agents/     Planner, Root, Orchestration, Execution, Deviation, Replanner
llm/        LLM client (mock + real) and prompt templates
grid/       Simulation environment, complexity estimation, tool library
config/     Thresholds and permission templates
tests/      Unit + end-to-end tests
```

## License

MIT
```
