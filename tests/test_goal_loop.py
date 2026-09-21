from agents.orchestration_agent import OrchestrationAgent
from agents.permission import Permission
from agents.task import Task
from grid.goal import Goal, goal_status
from grid.network import PowerNetwork
from grid.tools import simulate_action, validate_tool_call


def scenario():
    network = PowerNetwork()
    network.net.gen.at[0, "vm_pu"] = 0.90
    network.net.gen.at[3, "vm_pu"] = 0.90
    network._run_power_flow()
    return network


def test_joint_simulation_validates_each_action_and_preserves_real_network():
    network = scenario()
    goal = Goal.from_instruction("Bus 14 至少 1.00 p.u.，最多执行两次真实调整", 13)
    actions = [{"type": "set_gen_voltage", "gen_id": i, "vm_pu": 1.04} for i in (0, 3)]
    assert validate_tool_call("simulate_action", {"action": actions}, network.net)[0]
    assert not validate_tool_call("simulate_action", {"action": actions + [{"type": "set_gen_voltage", "gen_id": 99, "vm_pu": 1.04}]}, network.net)[0]
    result = simulate_action(network.net, actions, goal)
    assert result["goal_met"] is True
    assert goal_status(network.net, goal)["goal_met"] is False
    assert network.mutation_history == []


def test_feedback_refines_before_real_commit():
    network = scenario()
    goal = Goal.from_instruction("Bus 14 至少 1.00 p.u.，最多执行两次真实调整", 13)

    class FeedbackLLM:
        def __init__(self):
            self.calls = []

        def complete_json(self, system_prompt, user_prompt, **kwargs):
            self.calls.append(user_prompt)
            voltage = 1.0 if len(self.calls) == 1 else 1.04
            return {"instructions": [{"tool": "set_gen_voltage", "params": {"gen_id": i, "vm_pu": voltage}}
                                     for i in (0, 3)]}

    llm = FeedbackLLM()
    agent = OrchestrationAgent("direct", Task("direct", "执行调压", devices=[13]),
                               Permission.root_permission(), network, llm, 0, 2,
                               mission="Bus 14 至少 1.00 p.u.，最多执行两次真实调整",
                               goal=goal)
    result = agent.execute()
    assert result["success"] is True
    assert len(network.mutation_history) == 2
    assert goal_status(network.net, goal)["goal_met"] is True
    assert "上一轮联合仿真反馈" in llm.calls[1]


def test_unsolved_candidate_never_mutates_real_network():
    network = scenario()
    goal = Goal(13, 1.0, 2)

    class StubbornLLM:
        def complete_json(self, *args, **kwargs):
            return {"instructions": [{"tool": "set_gen_voltage", "params": {"gen_id": 0, "vm_pu": 1.0}}]}

    agent = OrchestrationAgent("direct", Task("direct", "执行调压", devices=[13]),
                               Permission.root_permission(), network, StubbornLLM(), 0, 2,
                               goal=goal)
    result = agent.execute()
    assert result["success"] is False
    assert network.mutation_history == []


def test_execution_can_commit_verified_simulation_only_plan():
    network = scenario()
    goal = Goal.from_instruction("Bus 14 至少 1.00 p.u.，最多执行两次真实调整", 13)

    class SimulationLLM:
        def complete_json(self, *args, **kwargs):
            return {"instructions": [{"tool": "simulate_action", "params": {"action": [
                {"type": "set_gen_voltage", "gen_id": i, "vm_pu": 1.04} for i in (0, 3)
            ]}}]}

    agent = OrchestrationAgent("direct", Task("direct", "执行调压", devices=[13]),
                               Permission.root_permission(), network, SimulationLLM(), 0, 2,
                               goal=goal)
    result = agent.execute()
    assert result["success"] is True
    assert len(network.action_log) == 2
    assert goal_status(network.net, goal)["goal_met"]


def test_bus_target_is_retained_alongside_network_interval():
    network = scenario()
    goal = Goal.from_instruction(
        "Bus 14 至少 1.00 p.u.，全网母线电压在 [0.95, 1.10] p.u.，最多两次调整", 13)
    candidate = simulate_action(network.net, [
        {"type": "set_gen_voltage", "gen_id": i, "vm_pu": 1.0} for i in (0, 3)
    ], goal)
    assert candidate["violation_count"] == 0
    assert candidate["goal_bus_vm"] < 1.0
    assert candidate["goal_met"] is False
