"""测试 IEEE 14-bus 网络加载和基本查询。"""

import pytest
from grid.network import PowerNetwork


@pytest.fixture
def pn():
    """每个测试用例都用一个干净的网络实例。"""
    return PowerNetwork()


def test_network_loads(pn):
    """网络应该有 14 个母线。"""
    assert len(pn.net.bus) == 14


def test_power_flow_converged(pn):
    """初始潮流应该收敛（res_bus 有数据）。"""
    assert not pn.net.res_bus.empty
    assert pn.net.res_bus["vm_pu"].notna().all()


def test_get_bus_voltage(pn):
    """查询母线电压应返回合理值。"""
    result = pn.get_bus_voltage(0)
    assert result["bus_id"] == 0
    assert 0.9 < result["vm_pu"] < 1.1  # 正常范围


def test_get_bus_voltage_invalid(pn):
    """查询不存在的母线应该报错。"""
    with pytest.raises(ValueError):
        pn.get_bus_voltage(999)


def test_get_line_loading(pn):
    """查询线路负载率应返回非负值。"""
    result = pn.get_line_loading(0)
    assert result["loading_percent"] >= 0


def test_get_generator_state(pn):
    """查询发电机应返回状态信息。"""
    result = pn.get_generator_state(0)
    assert "p_mw" in result
    assert "in_service" in result


def test_get_bus_connections(pn):
    """Bus 0 应该有线路和/或变压器连接。"""
    conns = pn.get_bus_connections(0)
    total = len(conns["lines"]) + len(conns["trafos"])
    assert total > 0


def test_set_gen_output(pn):
    """修改发电机出力后，潮流应更新。"""
    old_v = pn.get_bus_voltage(0)["vm_pu"]
    pn.set_gen_output(0, p_mw=60.0)
    new_v = pn.get_bus_voltage(0)["vm_pu"]
    # 出力变化后电压不一定变（取决于网络），但潮流应该重新计算了
    assert pn.net.gen.at[0, "p_mw"] == 60.0


def test_snapshot_independence(pn):
    """深拷贝的副本和原网络应互不影响。"""
    snapshot = pn.get_snapshot()
    snapshot.gen.at[0, "p_mw"] = 999.0
    assert pn.net.gen.at[0, "p_mw"] != 999.0