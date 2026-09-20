"""调压方案必须实际达到目标电压。"""

from agents.root_agent import RootAgent
from agents.orchestration_agent import OrchestrationAgent
from agents.permission import Permission
from agents.task import Task, TaskDAG
from grid.network import PowerNetwork
from grid.tools import check_constraints
from run_eval import SCENARIOS, make_network


class WrongGeneratorLLM:
    def complete_json(self, system_prompt, user_prompt, **kwargs):
        if "仿真" in user_prompt.split("\n", 1)[0]:
            return {"instructions": [{"tool": "simulate_action", "params": {
                "action": {"type": "set_gen_voltage", "gen_id": 3, "vm_pu": 1.04}},
                "description": "仿真错误的发电机"}]}
        if "执行" in user_prompt.split("\n", 1)[0]:
            return {"instructions": [{"tool": "set_gen_voltage", "params": {
                "gen_id": 3, "vm_pu": 1.04}, "description": "调整错误的发电机"}]}
        return {"instructions": [{"tool": "get_bus_voltage", "params": {"bus_id": 13},
                                  "description": "检查目标电压"}]}


def test_wrong_llm_action_is_replaced_by_feasible_voltage_action():
    network = PowerNetwork()
    network.set_gen_voltage(0, 0.98)
    dag = TaskDAG()
    dag.add_task(Task(id="t1", description="仿真调压方案", devices=[13]))
    dag.add_task(Task(id="t2", description="执行调压方案", dependencies=["t1"], devices=[13]))
    dag.add_task(Task(id="t3", description="验证目标电压", dependencies=["t2"], devices=[13]))
    root = RootAgent(network, dag, WrongGeneratorLLM(), tree_depth=3,
                     d0_info={}, certainty=0.9, target_bus=13)

    assert root.voltage_action["gen_id"] == 0
    result = root.execute()
    assert result["success"] is True
    assert network.get_bus_voltage(13)["vm_pu"] >= 1.0
    assert check_constraints(network.net)["all_satisfied"]


def test_completed_dag_does_not_count_as_recovery_without_target_voltage():
    network = PowerNetwork()
    network.set_gen_voltage(0, 0.98)
    dag = TaskDAG()
    dag.add_task(Task(id="t1", description="验证目标电压", devices=[13]))
    root = RootAgent(network, dag, WrongGeneratorLLM(), tree_depth=3,
                     d0_info={}, certainty=0.9, target_bus=13)
    assert root.execute()["success"] is False


def test_evaluation_scenarios_start_with_real_violations():
    for scenario in SCENARIOS.values():
        network = make_network(scenario)
        assert not check_constraints(network.net)["all_satisfied"]
        assert network.get_bus_voltage(scenario["target_bus"])["vm_pu"] < 0.95


def test_evaluation_preflight_preserves_existing_results(monkeypatch, tmp_path):
    import sys
    import pytest
    import run_eval

    output = tmp_path / "results.csv"
    output.write_text("existing results", encoding="utf-8")

    class BadAuthClient:
        def check_connection(self):
            raise ValueError("LLM API 认证失败 (HTTP 401)")

    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setattr(run_eval, "RealLLMClient", BadAuthClient)
    monkeypatch.setattr(sys, "argv", ["run_eval.py", "--output", str(output)])
    with pytest.raises(SystemExit):
        run_eval.main()
    assert output.read_text(encoding="utf-8") == "existing results"


def test_initial_violation_is_data_for_analysis_and_simulation():
    network = PowerNetwork()
    network.set_gen_voltage(0, 0.90)

    class AnalysisLLM:
        def complete_json(self, system_prompt, user_prompt, **kwargs):
            return {"instructions": [{"tool": "check_constraints", "params": {},
                                      "description": "读取初始约束违规"}]}

    analysis = OrchestrationAgent("analysis", Task("a", "分析低电压原因", devices=[13]),
                                  Permission.root_permission(), network, AnalysisLLM(), 1, 3)
    assert analysis.execute()["success"] is True

    action = {"type": "set_gen_voltage", "gen_id": 0, "vm_pu": 1.02}

    class SimulationLLM:
        def complete_json(self, system_prompt, user_prompt, **kwargs):
            return {"instructions": [
                {"tool": "simulate_action", "params": {"action": action}, "description": "仿真调压"},
                {"tool": "check_constraints", "params": {}, "description": "读取原始电网约束"},
            ]}

    simulation = OrchestrationAgent("simulation", Task("b", "仿真恢复电压", devices=[13]),
                                    Permission.root_permission(), network, SimulationLLM(), 1, 3,
                                    voltage_action=action)
    assert simulation.execute()["success"] is True
    assert network.get_bus_voltage(13)["vm_pu"] < 0.95


def test_adjustment_task_without_execute_prefix_uses_feasible_action():
    network = PowerNetwork()
    network.set_gen_voltage(0, 0.90)

    class WrongAdjustmentLLM:
        def complete_json(self, system_prompt, user_prompt, **kwargs):
            return {"instructions": [{"tool": "set_gen_voltage",
                                      "params": {"gen_id": 3, "vm_pu": 1.04},
                                      "description": "调节电压"}]}

    from grid.voltage_control import find_voltage_action
    action = find_voltage_action(network.net, 13, 1.0)
    agent = OrchestrationAgent("adjust", Task("a", "调节母线电压", devices=[13]),
                               Permission.root_permission(), network, WrongAdjustmentLLM(), 1, 3,
                               voltage_action=action)
    assert agent.execute()["success"] is True
    assert network.get_bus_voltage(13)["vm_pu"] >= 1.0


def test_initial_constraint_report_does_not_block_recovery_dag():
    network = PowerNetwork()
    network.set_gen_voltage(0, 0.90)
    dag = TaskDAG()
    dag.add_task(Task("a", "分析初始约束", devices=[13]))
    dag.add_task(Task("b", "仿真调压", dependencies=["a"], devices=[13]))
    dag.add_task(Task("c", "执行调压", dependencies=["b"], devices=[13]))
    dag.add_task(Task("d", "验证约束", dependencies=["c"], devices=[13]))

    class ReportingLLM:
        def complete_json(self, system_prompt, user_prompt, **kwargs):
            description = user_prompt.split("任务: ", 1)[1].split("\n", 1)[0]
            if description.startswith("仿真"):
                return {"instructions": [{"tool": "simulate_action", "params": {
                    "action": {"type": "set_gen_voltage", "gen_id": 3, "vm_pu": 1.04}},
                    "description": "仿真错误动作"},
                    {"tool": "check_constraints", "params": {}, "description": "查询初始约束"}]}
            if description.startswith("执行"):
                return {"instructions": [{"tool": "set_gen_voltage",
                                          "params": {"gen_id": 3, "vm_pu": 1.04},
                                          "description": "执行错误动作"}]}
            return {"instructions": [{"tool": "check_constraints", "params": {},
                                      "description": "报告约束"}]}

    result = RootAgent(network, dag, ReportingLLM(), tree_depth=3,
                       d0_info={}, certainty=0.8, target_bus=13).execute()
    assert result["success"] is True
    assert network.get_bus_voltage(13)["vm_pu"] >= 1.0


def test_eval_stops_on_service_failure_and_can_resume(monkeypatch, tmp_path):
    import csv
    import sys
    import pytest
    import run_eval

    output = tmp_path / "partial.csv"
    monkeypatch.setenv("LLM_API_KEY", "test-key")

    class ProbeClient:
        def check_connection(self):
            pass

    monkeypatch.setattr(run_eval, "RealLLMClient", ProbeClient)
    monkeypatch.setattr(run_eval, "make_network", lambda scenario: None)
    calls = []

    def fake_run(scenario, config, repeat):
        calls.append((scenario, config, repeat))
        row = dict.fromkeys(run_eval.FIELDS, "")
        row.update(eval_version=run_eval.EVAL_VERSION, scenario=scenario, config=config, repeat=repeat, success=1,
                   illegal_tool_calls=0, duplicate_tool_calls=0, total_tokens=10,
                   elapsed_seconds=1.0, infrastructure_error=len(calls) == 2)
        return row

    monkeypatch.setattr(run_eval, "run_once", fake_run)
    monkeypatch.setattr(sys, "argv", ["run_eval.py", "--repeats", "1", "--output", str(output)])
    with pytest.raises(SystemExit) as exc:
        run_eval.main()
    assert exc.value.code == 2
    with output.open(newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 1

    monkeypatch.setattr(sys, "argv", ["run_eval.py", "--repeats", "1", "--resume", "--output", str(output)])
    run_eval.main()
    with output.open(newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 8


def test_two_layer_baseline_runs_without_planner_or_root(monkeypatch):
    import run_eval
    from tests.fixtures_llm import MockLLMClient

    class EvalMock(MockLLMClient):
        total_tokens = 0

    monkeypatch.setattr(run_eval, "RealLLMClient", EvalMock)
    for scenario in run_eval.SCENARIOS:
        row = run_eval.run_once(scenario, "two_layer_orch_exec", 1)
        assert row["success"] == 1
        assert row["root_success"] == 1
        assert row["tree_depth"] == 2
        assert row["replan_count"] == 0
        assert row["final_voltage"] >= 1.0


def test_pre_action_constraint_check_does_not_fail_successful_adjustment():
    network = PowerNetwork()
    network.set_gen_voltage(0, 0.90)
    from grid.voltage_control import find_voltage_action
    action = find_voltage_action(network.net, 13, 1.0)

    class ChecksBeforeAction:
        def complete_json(self, system_prompt, user_prompt, **kwargs):
            return {"instructions": [
                {"tool": "check_constraints", "params": {}, "description": "读取调整前约束"},
                {"tool": "set_gen_voltage", "params": {"gen_id": 3, "vm_pu": 1.04},
                 "description": "执行调压"},
                {"tool": "check_constraints", "params": {}, "description": "读取调整后约束"},
            ]}

    agent = OrchestrationAgent("baseline", Task("a", "执行调压", devices=[13]),
                               Permission.root_permission(), network, ChecksBeforeAction(), 0, 2,
                               voltage_action=action)
    result = agent.execute()
    assert result["success"] is True
    assert network.get_bus_voltage(13)["vm_pu"] >= 1.0


def test_two_step_comparison_methods_can_both_solve_feasible_problem(monkeypatch):
    import run_compare

    class ScriptedLLM:
        total_tokens = 0

        def complete_json(self, system_prompt, user_prompt, **kwargs):
            if kwargs.get("source") == "planner":
                return {"tasks": [
                    {"id": "t1", "description": "分析初始约束", "dependencies": [], "devices": [13], "device_type": "bus"},
                    {"id": "t2", "description": "执行第一台发电机调压", "dependencies": ["t1"], "devices": [13], "device_type": "gen"},
                    {"id": "t3", "description": "执行第二台发电机调压", "dependencies": ["t2"], "devices": [13], "device_type": "gen"},
                    {"id": "t4", "description": "验证全网约束", "dependencies": ["t3"], "devices": [13], "device_type": "bus"},
                ]}
            description = user_prompt.split("任务: ", 1)[1].split("\n", 1)[0]
            if description.startswith("执行调度"):
                calls = [(0, 1.04), (3, 1.04)]
            elif "第一台" in description:
                calls = [(0, 1.04)]
            elif "第二台" in description:
                calls = [(3, 1.04)]
            else:
                calls = []
            if calls:
                return {"instructions": [
                    {"tool": "set_gen_voltage", "params": {"gen_id": gen_id, "vm_pu": vm_pu},
                     "description": "执行调压"} for gen_id, vm_pu in calls
                ]}
            return {"instructions": [{"tool": "check_constraints", "params": {}, "description": "检查约束"}]}

    monkeypatch.setattr(run_compare, "RealLLMClient", ScriptedLLM)
    run_compare.validate_scenario()
    for method in run_compare.METHODS:
        row = run_compare.run_once(method, 1)
        assert row["success"] == 1, (method, row["error"])
        assert row["action_count"] == 2
