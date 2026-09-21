"""case39 restoration fixture. Witness validation is strictly offline."""

import copy
import pandapower as pp
from grid.network import PowerNetwork
from grid.goal import Goal, goal_status
from grid.tools import constraints_not_worse

INSTRUCTION = (
    "39节点系统中，区域1多处母线电压偏低，区域2的两台发电机端母线也低于下限。"
    "请分区域定位可调发电机，先查询并仿真联合方案，再只用发电机电压设定值恢复："
    "全网母线电压在 [0.95, 1.10] p.u.、线路负载率不超过 100%。"
    "最多执行六次真实发电机电压调整；区域3的设备不得操作。"
    "中间步骤可以仍有原有越限，但不能新增或恶化越限；最后重新校验全网。"
)


def make_network():
    network = PowerNetwork(case="case39", device_limits={8: {"vm_max": 1.03}})
    for load_id, load in network.net.load.iterrows():
        if int(network.net.bus.at[int(load.bus), "zone"]) == 1:
            network.net.load.at[load_id, "p_mw"] *= 1.35
            network.net.load.at[load_id, "q_mvar"] *= 1.35
    network.net.gen.at[0, "vm_pu"] = 0.92
    network.net.gen.at[6, "vm_pu"] = 0.92
    network._run_power_flow()
    return network


def validate_scenario():
    """Call only from the offline evaluation setup; never include results in prompts."""
    network = make_network()
    goal = Goal.from_instruction(INSTRUCTION)
    initial = goal_status(network.net, goal)
    by_zone = initial["violations_by_zone"]
    assert len(initial["violations"]) == 9
    assert len(by_zone.get("1", [])) == 7
    assert len(by_zone.get("2", [])) == 2
    assert not by_zone.get("3")
    witness = copy.deepcopy(network.net)
    for gen_id, vm in ((0, .98), (6, .98), (1, 1.06)):
        witness.gen.at[gen_id, "vm_pu"] = vm
    pp.runpp(witness, algorithm="nr", init="results")
    assert goal_status(witness, goal)["goal_met"]
    assert constraints_not_worse(initial, goal_status(witness, goal))
    assert len(network.action_log) == 0
    return True
