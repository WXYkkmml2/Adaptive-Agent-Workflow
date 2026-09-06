import pytest

from grid.network import PowerNetwork
from grid.tools import call_tool, clean_tool_params


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
