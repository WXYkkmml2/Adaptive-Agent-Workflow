"""
Step 1 演示脚本。
运行方式: python main.py

演示内容：
1. 加载 IEEE 14-bus 网络
2. 查询各母线电压和线路负载
3. 计算 Bus 13（对应 IEEE 14-bus 的第 14 号母线，索引从 0 开始）的 D0
4. 用工具库做一次仿真操作
5. 校验约束
"""

from grid.network import PowerNetwork
from grid.topology import compute_d0, d0_to_h0
from grid.tools import (
    call_tool,
    get_available_tools,
    check_constraints,
)


def main():
    print("=" * 60)
    print("Step 演示：电网仿真沙盒")
    print("=" * 60)

    # ---- 1. 加载网络 ----
    print("\n[1] 加载 IEEE 14-bus 网络...")
    pn = PowerNetwork()
    print(f"    母线数: {len(pn.net.bus)}")
    print(f"    线路数: {len(pn.net.line)}")
    print(f"    发电机数: {len(pn.net.gen)}")
    print(f"    变压器数: {len(pn.net.trafo)}")

    # ---- 2. 查询电压 ----
    print("\n[2] 各母线电压 (p.u.):")
    voltages = pn.get_all_bus_voltages()
    for bus_id, vm in sorted(voltages.items()):
        marker = " ← 目标" if bus_id == 13 else ""
        print(f"    Bus {bus_id:2d}: {vm:.4f}{marker}")

    # ---- 3. 计算 D0 ----
    target_bus = 13  # 对应场景 "Bus 14 电压过低"
    print(f"\n[3] 计算 Bus {target_bus} 的 D0...")
    d0_result = compute_d0(pn.net, target_bus)
    print(f"    电气耦合强度 a = {d0_result['coupling_strength_a']}")
    print(f"    BFS 拓扑深度 b = {d0_result['topology_depth_b']}")
    print(f"    b 归一化       = {d0_result['topology_depth_b_norm']}")
    print(f"    网络直径       = {d0_result['network_diameter']}")
    print(f"    D0             = {d0_result['d0']}")
    h0 = d0_to_h0(d0_result["d0"])
    print(f"    → 映射基础树深 H0 = {h0}")

    # ---- 4. 工具库演示 ----
    print("\n[4] 工具库:")
    tools = get_available_tools()
    for t in tools:
        print(f"    - {t}")

    # 用工具查询 Bus 13 邻居
    result = call_tool("get_neighbor_buses", pn.net, bus_id=target_bus)
    print(f"\n    Bus {target_bus} 的邻居: {result['result']['neighbors']}")

    # ---- 5. 仿真操作 ----
    print("\n[5] 仿真：提高 Gen 0 电压设定值 → 观察 Bus 13 变化")
    old_v = pn.get_bus_voltage(target_bus)["vm_pu"]
    print(f"    操作前 Bus {target_bus} 电压: {old_v}")

    sim_result = call_tool(
        "simulate_action", pn.net,
        action={"type": "set_gen_voltage", "gen_id": 0, "vm_pu": 1.07}
    )
    if sim_result["success"]:
        new_v = sim_result["result"]["bus_voltages"].get(target_bus, "N/A")
        print(f"    仿真后 Bus {target_bus} 电压: {round(new_v, 6) if isinstance(new_v, float) else new_v}")
    else:
        print(f"    仿真失败: {sim_result.get('error')}")

    # ---- 6. 约束校验 ----
    print("\n[6] 约束校验（当前网络）:")
    constraints = check_constraints(pn.net)
    if constraints["all_satisfied"]:
        print("    ✓ 所有约束满足")
    else:
        print(f"    ✗ 发现 {constraints['violation_count']} 项违规:")
        for v in constraints["violations"]:
            print(f"      {v}")

    print("\n" + "=" * 60)
    print("Step 1 完成。基础就绪。")
    print("=" * 60)


if __name__ == "__main__":
    main()