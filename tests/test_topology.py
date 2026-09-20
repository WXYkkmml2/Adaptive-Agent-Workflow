"""测试 D0 计算模块。"""

import pytest
import pandapower as pp
import pandapower.networks as pn

from grid.topology import (
    get_neighbor_buses,
    compute_coupling_strength,
    compute_topology_depth,
    compute_d0,
    d0_to_h0,
)


@pytest.fixture
def net():
    """返回一个已跑过潮流的 IEEE 14-bus 网络。"""
    n = pn.case14()
    pp.runpp(n)
    return n


def test_get_neighbors(net):
    """Bus 0 至少有 1 个邻居。"""
    neighbors = get_neighbor_buses(net, 0, depth=1)
    assert 0 in neighbors           # 包含自身
    assert len(neighbors) >= 2       # 自身 + 至少一个邻居


def test_get_neighbors_depth2(net):
    """depth=2 应该比 depth=1 找到更多或相同数量的母线。"""
    n1 = get_neighbor_buses(net, 0, depth=1)
    n2 = get_neighbor_buses(net, 0, depth=2)
    assert len(n2) >= len(n1)


def test_coupling_strength_range(net):
    """耦合强度 a 应在 [0, 1] 范围。"""
    a = compute_coupling_strength(net, 0)
    assert 0.0 <= a <= 1.0


def test_topology_depth_positive(net):
    """拓扑深度 b 应该至少为 1。"""
    b = compute_topology_depth(net, 0)
    assert b >= 1


def test_compute_d0(net):
    """D0 计算应返回完整的结果字典。"""
    result = compute_d0(net, 0)
    assert "d0" in result
    assert "coupling_strength_a" in result
    assert "topology_depth_b" in result
    assert 0.0 <= result["d0"] <= 1.0


def test_d0_different_buses(net):
    """不同位置的母线应该有不同的 D0（或至少能计算）。"""
    d0_bus0 = compute_d0(net, 0)["d0"]
    d0_bus13 = compute_d0(net, 13)["d0"]
    # 不要求一定不同，但都应在合法范围内
    assert 0.0 <= d0_bus0 <= 1.0
    assert 0.0 <= d0_bus13 <= 1.0


def test_d0_to_h0_mapping():
    """D0 翻倍加一层，并遵守最小及最大层数。"""
    assert d0_to_h0(0.1) == 3
    assert d0_to_h0(0.5) == 3
    assert d0_to_h0(1.0) == 3
    assert d0_to_h0(4.0) == 4
    assert d0_to_h0(8.0) == 5
    assert d0_to_h0(0.0) == 3


def test_c_does_not_change_depth_until_attention_metric_exists():
    from agents.planner import Planner
    from grid.network import PowerNetwork
    planner = Planner(PowerNetwork(), None)
    assert planner._compute_tree_depth(4.0, 0.2) == 4
    assert planner._compute_tree_depth(4.0, 0.9) == 4
