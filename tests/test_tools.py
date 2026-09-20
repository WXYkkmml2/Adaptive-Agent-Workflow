import pytest

from grid.network import PowerNetwork
from grid.tools import call_tool, clean_tool_params, reset_tool_counters, get_tool_counters, check_constraints, constraints_not_worse


def test_clean_tool_params_removes_invalid_keys():
    cleaned = clean_tool_params(
        "get_bus_voltage",
        {"bus_id": 14, "region": "area_1", "voltage_level": 220},
    )
    assert cleaned == {"bus_id": 14}


def test_clean_tool_params_infers_missing_bus_id_from_context():
    cleaned = clean_tool_params(
        "get_bus_voltage",
        {},
        context="任务描述里提到了 Bus 14，需要检查电压",
    )
    assert cleaned == {"bus_id": 14}


def test_call_tool_removes_invalid_params_for_noarg_tool():
    network = PowerNetwork()
    result = call_tool(
        "check_constraints",
        network.net,
        extra_field="abc",
        context="校验全网约束",
    )
    assert result["success"] is True
    assert result["tool"] == "check_constraints"


def test_illegal_tool_counter_counts_missing_and_forbidden_tools():
    network = PowerNetwork()
    reset_tool_counters()
    assert not call_tool("missing_tool", network.net)["success"]
    assert not call_tool("get_generator_state", network.net,
                         permission={"device_types": ["bus"]}, gen_id=0)["success"]
    assert get_tool_counters()["illegal_tool_calls"] == 2


def test_multi_step_restoration_allows_partial_improvement_but_not_worsening():
    network = PowerNetwork()
    network.set_gen_voltage(0, 0.90)
    network.set_gen_voltage(3, 0.90)
    initial = check_constraints(network.net)
    network.set_gen_voltage(0, 1.04)
    partial = check_constraints(network.net)
    assert partial["violation_count"] < initial["violation_count"]
    assert constraints_not_worse(initial, partial)
    network.set_gen_voltage(3, 1.04)
    assert check_constraints(network.net)["all_satisfied"]

    network = PowerNetwork()
    network.set_gen_voltage(0, 0.90)
    network.set_gen_voltage(3, 0.90)
    initial = check_constraints(network.net)
    network.set_gen_voltage(0, 0.85)
    assert not constraints_not_worse(initial, check_constraints(network.net))
