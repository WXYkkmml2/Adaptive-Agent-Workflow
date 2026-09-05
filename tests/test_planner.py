"""测试 S1 规划器。"""

import pytest
from grid.network import PowerNetwork
from llm.client import MockLLMClient
from agents.planner import Planner


@pytest.fixture
def planner():
    network = PowerNetwork()
    llm = MockLLMClient()
    return Planner(network, llm)


def test_parse_target_bus14(planner):
    """'Bus 14' 应解析为内部索引 13。"""
    assert planner._parse_target("Bus 14 电压过低") == 13


def test_parse_target_chinese(planner):
    """中文'母线5'应解析为内部索引 4。"""
    assert planner._parse_target("母线5 电压异常") == 4


def test_plan_generates_dag(planner):
    result = planner.plan("Bus 14 电压过低，请恢复")
    dag = result["dag"]
    assert len(dag.tasks) >= 3  # 至少有几个任务


def test_plan_has_dependencies(planner):
    result = planner.plan("Bus 14 电压过低，请恢复")
    dag = result["dag"]
    # 至少有一个任务有依赖关系（t4 依赖 t1,t2,t3）
    has_deps = any(t.dependencies for t in dag.tasks.values())
    assert has_deps


def test_certainty_in_range(planner):
    result = planner.plan("Bus 14 电压过低")
    assert 0.0 <= result["certainty"] <= 1.0


def test_tree_depth_reasonable(planner):
    result = planner.plan("Bus 14 电压过低")
    assert 3 <= result["tree_depth"] <= 7


def test_d0_computed(planner):
    result = planner.plan("Bus 14 电压过低")
    assert "d0" in result["d0_info"]
    assert result["d0_info"]["d0"] >= 0