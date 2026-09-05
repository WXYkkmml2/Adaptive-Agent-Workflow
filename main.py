"""
Step 3 端到端演示。

两个场景：
  场景 A（正常）: Bus 14 电压偏低 → S1→S2→S3 → 成功
  场景 B（压力）: 电网受压 → 首次修复不足 → S4 重规划 → 成功

运行: python main.py
"""

import logging
import pandapower as pp
from grid.network import PowerNetwork
from grid.tools import check_constraints
from llm.client import create_llm_client, MockLLMClient
from agents.planner import Planner
from agents.root_agent import RootAgent
from agents.anti_example import AntiExampleStore

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def run_scenario(title: str, network: PowerNetwork, llm, target_bus: int = 13):
    """执行一个完整的 S1→S2→S3（→S4）场景。"""

    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

    initial_v = network.get_bus_voltage(target_bus)["vm_pu"]
    print(f"\n初始状态: Bus {target_bus} 电压 = {initial_v:.4f} p.u.")

    # 初始约束检查
    init_constr = check_constraints(network.net)
    if not init_constr["all_satisfied"]:
        print(f"初始约束违规: {init_constr['violation_count']} 项")
        for v in init_constr["violations"][:3]:
            print(f"  {v['type']}: Bus/Line {v.get('bus_id', v.get('line_id', '?'))} "
                  f"= {v['value']}")

    # ---- S1 规划 ----
    print(f"\n{'─' * 40}")
    print("S1 规划")
    print(f"{'─' * 40}")

    instruction = "Bus 14 电压过低，请分析并恢复，且不能造成其他节点或线路越限。"
    anti_examples = AntiExampleStore()
    planner = Planner(network, llm)
    plan = planner.plan(instruction)

    print(f"\n  D0 = {plan['d0_info']['d0']:.4f}")
    print(f"  C  = {plan['certainty']:.4f}")
    print(f"  H  = {plan['tree_depth']}")
    print(f"  任务数 = {len(plan['dag'].tasks)}")

    # ---- S2 + S3（+ S4）执行 ----
    print(f"\n{'─' * 40}")
    print("S2/S3 执行" + ("（含 S4 重规划）" if isinstance(llm, MockLLMClient)
                          and llm.failure_mode else ""))
    print(f"{'─' * 40}")

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

    # ---- 最终结果 ----
    print(f"\n{'─' * 40}")
    print("最终结果")
    print(f"{'─' * 40}")

    final_v = network.get_bus_voltage(target_bus)["vm_pu"]
    print(f"  Bus {target_bus} 电压: {initial_v:.4f} → {final_v:.4f} p.u.")
    print(f"  整体成功: {'✓' if result['success'] else '✗'}")

    final_constr = check_constraints(network.net)
    if final_constr["all_satisfied"]:
        print("  全网约束: ✓ 全部满足")
    else:
        print(f"  全网约束: ✗ {final_constr['violation_count']} 项违规")
        for v in final_constr["violations"][:3]:
            print(f"    {v}")

    if result.get("replan_log"):
        print(f"  重规划次数: {len(result['replan_log'])}")
        for entry in result["replan_log"]:
            print(f"    {entry['task_id']}: "
                  f"{'成功' if entry['replan_success'] else '失败'}")

    if result.get("anti_example_count", 0) > 0:
        print(f"  反例库: {result['anti_example_count']} 条记录")

    return result


def main():
    print("╔" + "═" * 58 + "╗")
    print("║     Adaptive Agent Workflow — 端到端演示              ║")
    print("║     IEEE 14-bus + pandapower + Multi-Agent S1→S4      ║")
    print("╚" + "═" * 58 + "╝")

    # ============================================================
    # 场景 A：正常 — 所有步骤一次成功
    # ============================================================
    network_a = PowerNetwork()
    llm_a = MockLLMClient(failure_mode=False)
    result_a = run_scenario("场景 A：正常流程", network_a, llm_a)

    # ============================================================
    # 场景 B：压力 — 首次不足 → S4 重规划 → 成功
    # ============================================================
    network_b = PowerNetwork()

    # 制造压力：降低发电机电压设定值，使全网电压偏低
    network_b.net.gen.at[0, "vm_pu"] = 0.98
    pp.runpp(network_b.net)

    llm_b = MockLLMClient(failure_mode=True)
    result_b = run_scenario("场景 B：压力场景（S4 重规划）", network_b, llm_b)

    # ============================================================
    # 总结
    # ============================================================
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║  演示总结                                             ║")
    print("╠" + "═" * 58 + "╣")
    print(f"║  场景 A (正常):   {'✓ 成功' if result_a['success'] else '✗ 失败'}{'':>38}║")
    print(f"║  场景 B (压力):   {'✓ 成功' if result_b['success'] else '✗ 失败'}"
          f"  (重规划 {len(result_b.get('replan_log', []))} 次)"
          f"{'':>20}║")
    print(f"║  反例库:          {result_b.get('anti_example_count', 0)} 条"
          f"{'':>40}║")
    print("╚" + "═" * 58 + "╝")

    print("\nStep 3 完成。可以推送 GitHub。")


if __name__ == "__main__":
    main()