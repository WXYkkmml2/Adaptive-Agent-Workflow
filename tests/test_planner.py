"""测试 S4 偏差检测与重规划流程。"""

import pytest
import logging
from grid.network import PowerNetwork
from llm.client import MockLLMClient
from agents.planner import Planner
from agents.root_agent import RootAgent
from agents.deviation import (
    Deviation, DeviationType, detect_deviation,
)
from agents.replanner import Replanner
from agents.task import Task, TaskDAG, TaskStatus
from agents.permission import Permission

logging.basicConfig(level=logging.WARNING)


# ============================================================
# 偏差检测
# ============================================================

def test_detect_no_deviation():
    """正常结果不应检测到偏差。"""
    results = [
        {"success": True, "tool": "get_bus_voltage",
         "result": {"bus_id": 13, "vm_pu": 1.02}}
    ]
    deviation = detect_deviation("t1", "获取电压数据", results)
    assert deviation is None


def test_detect_tool_failure():
    """工具执行失败应检测到偏差。"""
    results = [
        {"success": False, "tool": "get_bus_voltage",
         "error": "母线 999 不存在"}
    ]
    deviation = detect_deviation("t1", "获取电压数据", results)
    assert deviation is not None
    assert deviation.deviation_type == DeviationType.PARAMETER


def test_detect_constraint_violation():
    """约束违规应检测到 CONSTRAINT 类型偏差。"""
    results = [
        {"success": True, "tool": "check_constraints",
         "result": {
             "all_satisfied": False,
             "violation_count": 1,
             "violations": [{"type": "voltage_low", "bus_id": 13,
                             "value": 0.94, "limit": 0.95}],
         }}
    ]
    deviation = detect_deviation("t6", "全部满足", results)
    assert deviation is not None
    assert deviation.deviation_type == DeviationType.CONSTRAINT


def test_detect_empty_results():
    """空结果应检测到 TOOL_FAULT。"""
    deviation = detect_deviation("t1", "获取数据", [])
    assert deviation is not None
    assert deviation.deviation_type == DeviationType.TOOL_FAULT


# ============================================================
# 重规划器
# ============================================================

def test_replanner_basic():
    """重规划器应能成功重规划一个失败任务。"""
    network = PowerNetwork()
    llm = MockLLMClient(failure_mode=False)  # 不注入失败，直接成功
    d0_info = {"coupling_strength_a": 0.5, "topology_depth_b": 2,
               "topology_depth_b_norm": 0.4}

    replanner = Replanner(
        network=network, llm=llm,
        d0_info=d0_info, certainty=0.7, tree_depth=3,
    )

    task = Task(id="t1", description="查询目标节点电压",
                devices=[13], device_type="bus")
    deviation = Deviation(
        deviation_type=DeviationType.TOOL_FAULT,
        description="工具调用超时",
        expected="获取电压数据",
        actual="超时",
        task_id="t1",
    )

    result = replanner.handle_failure(
        task, deviation, Permission.root_permission()
    )
    assert result["success"] is True


def test_replanner_respects_max_attempts():
    """超过最大重试次数应返回 needs_human。"""
    network = PowerNetwork()
    # failure_mode=True 且不给重规划，会持续失败
    # 但 mock 的重规划响应会成功，所以这里用一个始终失败的场景
    # 直接测试 attempt > MAX
    from config.settings import MAX_REPLAN_ATTEMPTS
    llm = MockLLMClient(failure_mode=False)

    replanner = Replanner(
        network=network, llm=llm,
        d0_info={}, certainty=0.7, tree_depth=3,
    )

    task = Task(id="t_fail", description="始终失败",
                devices=[13], device_type="bus")
    deviation = Deviation(
        deviation_type=DeviationType.INSUFFICIENT,
        description="不足",
        expected="成功", actual="失败",
        task_id="t_fail",
    )

    # 直接传入超过限制的 attempt
    result = replanner.handle_failure(
        task, deviation, Permission.root_permission(),
        attempt=MAX_REPLAN_ATTEMPTS + 1,
    )
    assert result["success"] is False
    assert result["needs_human"] is True


# ============================================================
# 端到端：正常场景
# ============================================================

def test_e2e_normal():
    """正常场景端到端：全部任务成功完成。"""
    network = PowerNetwork()
    llm = MockLLMClient(failure_mode=False)
    planner = Planner(network, llm)

    plan = planner.plan("Bus 14 电压过低，请恢复")
    root = RootAgent(
        network=network,
        dag=plan["dag"],
        llm=llm,
        tree_depth=plan["tree_depth"],
        d0_info=plan["d0_info"],
        certainty=plan["certainty"],
    )
    result = root.execute()
    assert result["success"] is True


# ============================================================
# 端到端：失败 → 重规划 → 成功
# ============================================================

def test_e2e_failure_and_replan():
    """
    压力场景：
    1. 降低发电机出力使 Bus 13 电压偏低
    2. MockLLM 首次返回不足的调整 → 约束违规 → 触发 S4
    3. 重规划返回更强的调整 → 成功
    """
    network = PowerNetwork()

    # 制造压力：降低 Gen 0 电压设定值，拉低全网电压
    network.net.gen.at[0, "vm_pu"] = 0.98
    import pandapower as pp
    pp.runpp(network.net)

    bus13_v = network.net.res_bus.at[13, "vm_pu"]
    # 确认 Bus 13 电压确实偏低
    assert bus13_v < 1.0, f"Bus 13 电压应偏低，实际 {bus13_v}"

    llm = MockLLMClient(failure_mode=True)  # 注入失败
    planner = Planner(network, llm)

    plan = planner.plan("Bus 14 电压过低，请恢复")
    root = RootAgent(
        network=network,
        dag=plan["dag"],
        llm=llm,
        tree_depth=plan["tree_depth"],
        d0_info=plan["d0_info"],
        certainty=plan["certainty"],
    )

    result = root.execute()

    assert result["success"] is True
    assert len(root.replan_log) > 0