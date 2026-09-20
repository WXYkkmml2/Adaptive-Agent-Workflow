"""最小可运行演示：单场景、单链路。"""

import logging
from grid.network import PowerNetwork
from grid.tools import check_constraints
from llm.client import create_llm_client
from agents.planner import Planner
from agents.root_agent import RootAgent

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main():
    network = PowerNetwork()
    llm = create_llm_client()

    print("\n" + "=" * 60)
    print("  场景 A：正常流程")
    print("=" * 60)

    # 示例先制造确实存在的低电压；默认 case14 的 Bus 14 电压本来合格。
    network.set_gen_voltage(0, 0.98)
    target_bus = 13
    initial_v = network.get_bus_voltage(target_bus)["vm_pu"]
    print(f"\n初始状态: Bus 14（内部索引 {target_bus}）电压 = {initial_v:.4f} p.u.")

    instruction = "Bus 14 电压过低，请分析并恢复，且不能造成其他节点或线路越限。"
    planner = Planner(network, llm)
    plan = planner.plan(instruction)

    print(f"\n  D0 = {plan['d0_info']['d0']:.4f}")
    print(f"  C  = {plan['certainty']:.4f}")
    print(f"  H  = {plan['tree_depth']}")
    print(f"  任务数 = {len(plan['dag'].tasks)}")

    root = RootAgent(
        network=network,
        dag=plan["dag"],
        llm=llm,
        tree_depth=plan["tree_depth"],
        d0_info=plan["d0_info"],
        certainty=plan["certainty"],
    )

    result = root.execute()

    final_v = network.get_bus_voltage(target_bus)["vm_pu"]
    print(f"\n  Bus 14 电压: {initial_v:.4f} → {final_v:.4f} p.u.")
    recovered = final_v >= 1.0
    final_constr = check_constraints(network.net)
    success = result["success"] and recovered and final_constr["all_satisfied"]
    print(f"  整体成功: {'✓' if success else '✗'}")
    if final_constr["all_satisfied"]:
        print("  全网约束: ✓ 全部满足")
    else:
        print(f"  全网约束: ✗ {final_constr['violation_count']} 项违规")
        for v in final_constr["violations"][:3]:
            print(f"    {v}")

    print("\n" + "=" * 60)
    print("  结果总结")
    print("=" * 60)
    print(f"  场景 A (正常): {'✓ 成功' if success else '✗ 失败'}")


if __name__ == "__main__":
    main()
