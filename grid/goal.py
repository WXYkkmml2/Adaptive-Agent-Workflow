"""Acceptance criteria parsed from the operator instruction."""
import re
from dataclasses import dataclass, field
from config.settings import BUS_VOLTAGE_MIN, BUS_VOLTAGE_MAX, LINE_LOADING_MAX

_CN = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}

def _number(value):
    if value.isdigit():
        return int(value)
    if value.startswith("十"):
        return 10 + _CN.get(value[1:], 0)
    if "十" in value:
        left, right = value.split("十", 1)
        return 10 * _CN[left] + _CN.get(right, 0)
    return _CN[value]

@dataclass(frozen=True)
class Goal:
    bus: int | None = None
    vmin: float = BUS_VOLTAGE_MIN
    max_real_actions: int = 99
    mode: str = "all_constraints"
    vmax: float = BUS_VOLTAGE_MAX
    line_max: float = LINE_LOADING_MAX
    forbidden_regions: frozenset = field(default_factory=frozenset)
    bus_target_min: float | None = None

    @classmethod
    def from_instruction(cls, text: str, bus: int | None = None):
        count = re.search(r"最多[^0-9一二两三四五六七八九十]{0,8}([0-9]+|[一二两三四五六七八九十]+)\s*次", text)
        cap = _number(count[1]) if count else 99
        interval = re.search(r"\[\s*([\d.]+)\s*,\s*([\d.]+)\s*\]", text)
        minimum = re.search(r"至少\s*([\d.]+)\s*p\.?u", text, re.I)
        forbidden = frozenset(_number(m) for m in re.findall(r"区域\s*([0-9]+|[一二两三四五六七八九十]+)\s*的设备不得操作", text))
        line = re.search(r"线路负载率不超过\s*([\d.]+)\s*%", text)
        mode = "bus_vmin" if bus is not None and not interval else "all_constraints"
        return cls(bus, float(interval[1]) if interval else float(minimum[1]) if minimum else BUS_VOLTAGE_MIN,
                   cap, mode, float(interval[2]) if interval else BUS_VOLTAGE_MAX,
                   float(line[1]) if line else LINE_LOADING_MAX, forbidden,
                   float(minimum[1]) if bus is not None and minimum else None)

def goal_status(net, goal: Goal) -> dict:
    violations = []
    by_zone = {}
    for bus_id, row in net.res_bus.iterrows():
        value = float(row.vm_pu)
        network_vmin = BUS_VOLTAGE_MIN if goal.mode == "bus_vmin" else goal.vmin
        kind = "voltage_low" if value < network_vmin else "voltage_high" if value > goal.vmax else None
        if kind:
            item = {"type": kind, "bus_id": int(bus_id), "value": round(value, 6),
                    "limit": network_vmin if kind == "voltage_low" else goal.vmax}
            violations.append(item)
            zone = str(int(net.bus.at[bus_id, "zone"]))
            by_zone.setdefault(zone, []).append(item)
    for line_id, row in net.res_line.iterrows():
        if float(row.loading_percent) > goal.line_max:
            item = {"type": "line_overload", "line_id": int(line_id),
                    "value": round(float(row.loading_percent), 4), "limit": goal.line_max}
            violations.append(item)
            zone = str(int(net.bus.at[int(net.line.at[line_id, "from_bus"]), "zone"]))
            by_zone.setdefault(zone, []).append(item)
    bus_vm = float(net.res_bus.at[goal.bus, "vm_pu"]) if goal.bus is not None else None
    bus_target = goal.bus_target_min if goal.bus_target_min is not None else goal.vmin
    met = not violations and (goal.bus is None or bus_vm >= bus_target)
    return {"goal_met": met, "goal_bus_vm": round(bus_vm, 6) if bus_vm is not None else None,
            "goal_shortfall": round(max(0, bus_target - bus_vm), 6) if bus_vm is not None else 0,
            "violation_count": len(violations), "violations": violations,
            "violations_by_zone": by_zone}
