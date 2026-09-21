"""Reusable DAG partitioning and method configuration, independent of evaluation I/O."""
from dataclasses import dataclass, asdict
from agents.permission import Permission
from agents.task import Task


@dataclass(frozen=True)
class MethodConfig:
    planner: bool
    permission_shrink: bool
    replan_mode: str

    def to_dict(self):
        return asdict(self)


METHOD_CONFIGS = {
    "hierarchical": MethodConfig(True, True, "local"),
    "two_layer": MethodConfig(False, False, "local"),
    "two_layer_full_restart": MethodConfig(False, False, "full"),
    "hierarchical_no_shrink": MethodConfig(True, False, "local"),
    "hierarchical_full_restart": MethodConfig(True, True, "full"),
}


def partition_dag(dag, net):
    """Merge same-region tasks AND dependency-connected tasks; never split an edge.

    Stable topological order uses the LLM's insertion order to break ties.
    A global dependency chain remains one subtree, even if that removes locality.
    """
    dag.validate()
    parent = {i: i for i in dag.tasks}

    def find(i):
        while parent[i] != i:
            i = parent[i]
        return i

    def union(a, b):
        parent[find(b)] = find(a)

    zones = {}
    for tid, task in dag.tasks.items():
        regions = Permission.from_task(task, net).regions
        zones[tid] = regions
        for other in dag.tasks:
            if other in zones and zones[other] == regions:
                union(other, tid)
        for dep in task.dependencies:
            union(dep, tid)
    groups = {}
    for tid in dag.tasks:
        groups.setdefault(find(tid), []).append(tid)
    ordered = []
    for ids in groups.values():
        remaining, done, result = list(ids), set(), []
        while remaining:
            tid = next(i for i in remaining if set(dag.tasks[i].dependencies) <= done)
            result.append(tid)
            done.add(tid)
            remaining.remove(tid)
        ordered.append(result)
    return ordered


def build_method(config, kernel):
    """Only factory chooses architecture; physical kernel receives no method switch."""
    from agents.planner import Planner
    from agents.root_agent import RootAgent
    from agents.orchestration_agent import OrchestrationAgent
    if config.planner:
        plan = Planner(kernel.network, kernel.llm).plan(kernel.instruction)
        root = RootAgent(kernel.network, plan["dag"], kernel.llm, plan["tree_depth"],
                         plan["d0_info"], plan["certainty"], permission_shrink=config.permission_shrink,
                         replan_mode=config.replan_mode, mission=kernel.instruction, goal=kernel.goal,
                         kernel=kernel)
        return root, plan
    agent = OrchestrationAgent("whole", Task("whole", kernel.instruction), Permission.root_permission(),
                               kernel.network, kernel.llm, 0, 2, permission_shrink=False,
                               mission=kernel.instruction, goal=kernel.goal, kernel=kernel)
    agent.replan_mode = config.replan_mode
    return agent, {"tree_depth": 2, "d0_info": {"d0": 0}, "dag": None}
