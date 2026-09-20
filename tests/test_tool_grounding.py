from agents.orchestration_agent import OrchestrationAgent
from agents.permission import Permission
from agents.task import Task
from grid.network import PowerNetwork
from grid.tools import get_tool_catalog, validate_tool_call


class RepairingLLM:
    def __init__(self):
        self.prompts = []

    def complete_json(self, system_prompt, user_prompt, **kwargs):
        self.prompts.append((system_prompt, user_prompt))
        if len(self.prompts) == 1:
            return {"instructions": [{
                "tool": "get_line_loading",
                "params": {"from_bus": 8, "to_bus": 13},
                "description": "查询线路",
            }]}
        return {"instructions": [{
            "tool": "get_line_loading",
            "params": {"line_id": 11},
            "description": "查询线路",
        }]}


def test_invalid_line_parameters_are_repaired_before_execution():
    network = PowerNetwork()
    llm = RepairingLLM()
    agent = OrchestrationAgent(
        agent_id="orch_test",
        task=Task(id="t1", description="分析 Bus 14 相邻线路", devices=[13]),
        permission=Permission.root_permission(),
        network=network,
        llm=llm,
        current_depth=1,
        max_depth=3,
    )
    result = agent.execute()
    assert result["success"] is True
    assert len(llm.prompts) == 2
    assert "from_bus" in llm.prompts[1][1]
    assert "line_id" in llm.prompts[0][0]
    assert result["execution_results"][0]["tool_results"][0]["result"]["line_id"] == 11


def test_catalog_uses_internal_device_ids():
    network = PowerNetwork()
    catalog = get_tool_catalog(network.net, ["get_line_loading", "set_gen_voltage"])
    assert {"line_id": 11, "from_bus": 8, "to_bus": 13} in catalog["lines"]
    assert any(g["gen_id"] == 0 and g["bus_id"] == 5 and "vm_pu" in g
               for g in catalog["generators"])
    assert validate_tool_call("get_line_loading", {"line_id": "8-14"}, network.net)[0] is False


def test_target_bus_user_label_normalized_without_second_request():
    class LabelLLM:
        calls = 0

        def complete_json(self, *args, **kwargs):
            self.calls += 1
            return {"instructions": [{
                "tool": "get_bus_voltage",
                "params": {"bus_id": 14},
                "description": "查询 Bus 14",
            }]}

    llm = LabelLLM()
    agent = OrchestrationAgent(
        agent_id="orch_label",
        task=Task(id="t1", description="查询 Bus 14 电压", devices=[13]),
        permission=Permission.root_permission(),
        network=PowerNetwork(),
        llm=llm,
        current_depth=1,
        max_depth=3,
    )
    result = agent.execute()
    assert result["success"] is True
    assert llm.calls == 1
    assert result["execution_results"][0]["tool_results"][0]["result"]["bus_id"] == 13


def test_execution_plan_with_bus14_label_runs_without_repair():
    class ExecutionLLM:
        calls = 0

        def complete_json(self, *args, **kwargs):
            self.calls += 1
            return {"instructions": [
                {"tool": "set_gen_voltage", "params": {"gen_id": 0, "vm_pu": 1.08},
                 "description": "执行发电机调压"},
                {"tool": "get_bus_voltage", "params": {"bus_id": 14},
                 "description": "读取 Bus 14 电压"},
            ]}

    network = PowerNetwork()
    network.set_gen_voltage(0, 0.98)
    llm = ExecutionLLM()
    agent = OrchestrationAgent(
        agent_id="orch_execute",
        task=Task(id="t2", description="执行：恢复 Bus 14 电压", devices=[13]),
        permission=Permission.root_permission(),
        network=network,
        llm=llm,
        current_depth=1,
        max_depth=3,
    )
    result = agent.execute()
    assert result["success"] is True
    assert llm.calls == 1
    assert network.net.res_bus.at[13, "vm_pu"] >= 1.0
