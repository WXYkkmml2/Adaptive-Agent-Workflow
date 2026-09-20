import pytest

from agents.planner import Planner
from agents.root_agent import RootAgent
from grid.network import PowerNetwork
from llm.client import create_llm_client


class PlanLLM:
    def __init__(self, tasks):
        self.tasks = tasks

    def complete_json(self, *args, **kwargs):
        return {"tasks": self.tasks}


def make_task(task_id, dependencies=None):
    return {
        "id": task_id,
        "description": f"任务 {task_id}",
        "dependencies": dependencies or [],
        "devices": [13],
        "device_type": "bus",
        "voltage_level": "MV",
    }


def test_task_count_and_dependencies_come_from_model():
    network = PowerNetwork()
    two = Planner(network, PlanLLM([
        make_task("a"),
        make_task("b", ["a"]),
    ])).plan("Bus 14 电压过低")
    three = Planner(network, PlanLLM([
        make_task("a"),
        make_task("b"),
        make_task("c", ["a", "b"]),
    ])).plan("Bus 14 电压过低")
    assert len(two["dag"].tasks) == 2
    assert len(three["dag"].tasks) == 3
    assert [task.id for task in three["dag"].get_ready_tasks()] == ["a", "b"]


def test_empty_model_plan_is_not_replaced_by_fixed_tasks():
    network = PowerNetwork()
    with pytest.raises(ValueError, match="未返回有效任务图"):
        Planner(network, PlanLLM([])).plan("Bus 14 电压过低")


def test_missing_api_key_does_not_silently_use_mock(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    with pytest.raises(ValueError, match="LLM_API_KEY"):
        create_llm_client()


def test_model_generated_three_task_plan_runs_end_to_end():
    class ThreeTaskLLM:
        def complete_json(self, system_prompt, user_prompt, **kwargs):
            if kwargs.get("source") == "planner":
                return {"tasks": [
                    {**make_task("a"), "description": "仿真分析电压调节"},
                    {**make_task("b", ["a"]), "description": "执行发电机电压调节"},
                    {**make_task("c", ["b"]), "description": "执行后验证电压和约束"},
                ]}
            description = user_prompt.split("任务: ", 1)[1].split("\n", 1)[0]
            if description.startswith("仿真"):
                instruction = {"tool": "simulate_action", "params": {
                    "action": {"type": "set_gen_voltage", "gen_id": 0, "vm_pu": 1.08}}}
            elif description.startswith("执行发电机"):
                instruction = {"tool": "set_gen_voltage", "params": {"gen_id": 0, "vm_pu": 1.08}}
            else:
                instruction = {"tool": "get_bus_voltage", "params": {"bus_id": 14}}
            instruction["description"] = description
            instruction["expected_result"] = "操作成功"
            return {"instructions": [instruction]}

    network = PowerNetwork()
    network.set_gen_voltage(0, 0.98)
    llm = ThreeTaskLLM()
    plan = Planner(network, llm).plan("Bus 14 电压过低，请恢复")
    result = RootAgent(
        network, plan["dag"], llm, plan["tree_depth"],
        plan["d0_info"], plan["certainty"],
    ).execute()
    assert len(plan["dag"].tasks) == 3
    assert result["success"] is True
    assert network.net.res_bus.at[13, "vm_pu"] >= 1.0
