
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
export LLM_MODEL="gpt-4o-mini"
python main.py
```

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

