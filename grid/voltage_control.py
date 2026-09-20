"""用潮流仿真选择满足目标且不造成越限的发电机调压动作。"""

from grid.tools import check_constraints, simulate_action


def find_voltage_action(net, target_bus: int, min_voltage: float) -> dict | None:
    """返回使目标母线达标的最小设定值改变量；不修改原网络。"""
    candidates = []
    for gen_id, gen in net.gen.iterrows():
        if not gen["in_service"]:
            continue
        current = float(gen["vm_pu"])
        for step in range(95, 111):
            vm_pu = step / 100
            if abs(vm_pu - current) < 1e-9:
                continue
            action = {"type": "set_gen_voltage", "gen_id": int(gen_id), "vm_pu": vm_pu}
            result = simulate_action(net, action)
            if not result.get("success"):
                continue
            if result["bus_voltages"][target_bus] < min_voltage:
                continue
            if not check_constraints(result["net_copy"])["all_satisfied"]:
                continue
            candidates.append((abs(vm_pu - current), vm_pu, int(gen_id), action))
    return min(candidates)[3] if candidates else None
