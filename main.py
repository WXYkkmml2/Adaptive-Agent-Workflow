"""
Step 2 演示：完整的 S1 → S2 → S3 流程。

场景（来自"范围"文档）：
"Bus 14 电压过低，请分析并恢复，且不能造成其他节点或线路越限。"
"""

import logging
from grid.network import PowerNetwork
from llm.client import create_llm_client
from agents.planner import Planner
from agents.root_agent import RootAgent
from agents.anti_example import AntiExampleStore

# 配置日志：让智能体的执行过程可见
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger(__name__)


def main():
    print("=" * 60)
    print("Step 2 演示：S1 → S2 → S3 完整流程")
    print("=" * 60)

    # ---- 0. 初始化 ----
    network = PowerNetwork()
    llm = create_llm_client()           # 无 API Key 时自动使用 Mock
    anti_examples = AntiExampleStore()   # Step 3 再填充

    # 显示初始状态
    target_bus = 13  # "Bus 14"（0-indexed）
    initial_v = network.get_bus_voltage(target_bus)["vm_pu"]
    print(f"\n初始状态: Bus {target_bus} 电压 = {initial_v:.4f} p.u.")

    # ---- 1. S1 规划 ----
    print("\n" + "=" * 60)
    print("阶段一：S1 规划")
    print("=" * 60)

    instruction = "Bus 14 电压过低，请分析并恢复，且不能造成其他节点或线路越限。"
    print(f"调度指令: {instruction}\n")

    planner = Planner(network, llm)
    plan = planner.plan(instruction)

    print(f"\n规划结果:")
    print(f"  目标母线: Bus {plan['target_bus']}")
    print(f"  D0 = {plan['d0_info']['d0']:.4f}")
    print(f"  C  = {plan['certainty']:.4f}")
    print(f"  H  = {plan['tree_depth']}")
    print(f"  任务 DAG ({len(plan['dag'].tasks)} 个任务):")
    print(plan['dag'].summary())

    # ---- 2. S2 + S3 执行 ----
    print("\n" + "=" * 60)
    print("阶段二：S2 编排 + S3 执行")
    print("=" * 60)

    root = RootAgent(
        network=network,
        dag=plan["dag"],
        llm=llm,
        tree_depth=plan["tree_depth"],
        d0_info=plan["d0_info"],
        certainty=plan["certainty"],
        anti_example_store=anti_examples,
    )

    result = root.execute()

    # ---- 3. 最终结果 ----
    print("\n" + "=" * 60)
    print("最终结果")
    print("=" * 60)

    final_v = network.get_bus_voltage(target_bus)["vm_pu"]
    print(f"  Bus {target_bus} 电压: {initial_v:.4f} → {final_v:.4f} p.u.")
    print(f"  整体成功: {'✓' if result['success'] else '✗'}")

    # 检查最终约束
    from grid.tools import check_constraints
    constraints = check_constraints(network.net)
    if constraints["all_satisfied"]:
        print("  全网约束: ✓ 全部满足")
    else:
        print(f"  全网约束: ✗ {constraints['violation_count']} 项违规")
        for v in constraints["violations"][:5]:
            print(f"    {v}")

    print("\n" + "=" * 60)
    print("Step 2 完成。可以推送 GitHub。")
    print("=" * 60)


if __name__ == "__main__":
    main()