"""
执行智能体的预设工具库。

设计思路
- 每个工具是一个函数，操作 pandapower 网络
- 每个工具在 TOOL_REGISTRY 中注册，带有元数据
- 元数据包括设备类型、风险等级，供后续 S3 权限过滤和策略分支使用

风险等级说明：
  low   — 纯查询，不改变电网状态（对应 "直接执行"策略）
  medium — 修改参数但可逆（对应"仿真验证后执行"策略）
  high   — 开关操作或大幅调整（对应"人工确认"策略）
"""

import copy
import pandapower as pp
from config.settings import BUS_VOLTAGE_MIN, BUS_VOLTAGE_MAX, LINE_LOADING_MAX


# ============================================================
# 工具函数定义
# ============================================================

def get_bus_voltage(net, bus_id: int) -> dict:
    """查询指定母线的电压幅值和相角。"""
    return {
        "bus_id": bus_id,
        "vm_pu": round(net.res_bus.at[bus_id, "vm_pu"], 6),
        "va_degree": round(net.res_bus.at[bus_id, "va_degree"], 4),
    }


def get_neighbor_buses(net, bus_id: int) -> dict:
    """
    查询指定母线的所有相邻母线。
    "相邻"指通过线路或变压器直接相连。
    """
    neighbors = set()

    # 通过线路相连的母线
    for _, row in net.line.iterrows():
        if not row["in_service"]:
            continue
        if row["from_bus"] == bus_id:
            neighbors.add(int(row["to_bus"]))
        elif row["to_bus"] == bus_id:
            neighbors.add(int(row["from_bus"]))

    # 通过变压器相连的母线
    for _, row in net.trafo.iterrows():
        if not row.get("in_service", True):
            continue
        if row["hv_bus"] == bus_id:
            neighbors.add(int(row["lv_bus"]))
        elif row["lv_bus"] == bus_id:
            neighbors.add(int(row["hv_bus"]))

    return {
        "bus_id": bus_id,
        "neighbors": sorted(neighbors),
        "count": len(neighbors),
    }


def get_line_loading(net, line_id: int) -> dict:
    """查询指定线路的负载率和潮流信息。"""
    return {
        "line_id": line_id,
        "loading_percent": round(net.res_line.at[line_id, "loading_percent"], 4),
        "p_from_mw": round(net.res_line.at[line_id, "p_from_mw"], 4),
        "q_from_mvar": round(net.res_line.at[line_id, "q_from_mvar"], 4),
        "in_service": net.line.at[line_id, "in_service"],
    }


def get_generator_state(net, gen_id: int) -> dict:
    """查询指定发电机的设定值和实际出力。"""
    return {
        "gen_id": gen_id,
        "p_mw": round(net.gen.at[gen_id, "p_mw"], 4),
        "vm_pu": round(net.gen.at[gen_id, "vm_pu"], 6),
        "in_service": net.gen.at[gen_id, "in_service"],
        "actual_p_mw": round(net.res_gen.at[gen_id, "p_mw"], 4),
        "actual_q_mvar": round(net.res_gen.at[gen_id, "q_mvar"], 4),
    }


def simulate_action(net, action: dict) -> dict:
    """
    在网络副本上执行操作并返回仿真结果。
        
    不修改原始网络。在副本上操作后跑潮流，返回新的电网状态。
    
    action 格式示例:
        {"type": "set_gen_output", "gen_id": 0, "p_mw": 50.0}
        {"type": "set_gen_voltage", "gen_id": 0, "vm_pu": 1.04}
        {"type": "set_line_status", "line_id": 0, "in_service": False}
    """
    # 在深拷贝上操作，不影响原始网络
    net_copy = copy.deepcopy(net)

    action_type = action.get("type")

    if action_type == "set_gen_output":
        net_copy.gen.at[action["gen_id"], "p_mw"] = action["p_mw"]

    elif action_type == "set_gen_voltage":
        net_copy.gen.at[action["gen_id"], "vm_pu"] = action["vm_pu"]

    elif action_type == "set_line_status":
        net_copy.line.at[action["line_id"], "in_service"] = action["in_service"]

    else:
        return {"success": False, "error": f"未知操作类型: {action_type}"}

    # 在副本上跑潮流
    try:
        pp.runpp(net_copy, algorithm="nr", init="results")
    except pp.powerflow.LoadflowNotConverged:
        return {
            "success": False,
            "error": "潮流不收敛——操作可能导致电网失稳",
            "action": action,
        }

    return {
        "success": True,
        "action": action,
        "bus_voltages": net_copy.res_bus["vm_pu"].to_dict(),
        "line_loadings": net_copy.res_line["loading_percent"].to_dict(),
        "net_copy": net_copy,  # 返回副本，供后续决策使用
    }


def check_constraints(net) -> dict:
    """
    校验当前网络是否满足运行约束。
    
    检查所有母线电压是否在 [0.95, 1.05] p.u.，
    所有线路负载率是否 < 100%。
    
    返回:
        violations: 违规项列表（空列表 = 全部满足）
        all_satisfied: 布尔值
    """
    violations = []

    # 检查母线电压
    for bus_id in net.res_bus.index:
        vm = net.res_bus.at[bus_id, "vm_pu"]
        if vm < BUS_VOLTAGE_MIN:
            violations.append({
                "type": "voltage_low",
                "bus_id": int(bus_id),
                "value": round(vm, 6),
                "limit": BUS_VOLTAGE_MIN,
            })
        elif vm > BUS_VOLTAGE_MAX:
            violations.append({
                "type": "voltage_high",
                "bus_id": int(bus_id),
                "value": round(vm, 6),
                "limit": BUS_VOLTAGE_MAX,
            })

    # 检查线路负载率
    for line_id in net.res_line.index:
        loading = net.res_line.at[line_id, "loading_percent"]
        if loading > LINE_LOADING_MAX:
            violations.append({
                "type": "line_overload",
                "line_id": int(line_id),
                "value": round(loading, 4),
                "limit": LINE_LOADING_MAX,
            })

    return {
        "all_satisfied": len(violations) == 0,
        "violation_count": len(violations),
        "violations": violations,
    }


# ============================================================
# 工具注册表
# ============================================================
# 每个工具注册元数据，供  执行智能体按权限筛选工具时使用。
# device_types: 该工具涉及的设备类型（和权限三元组的 device_types 对应）
# risk_level: 风险等级 → 决定 执行策略（直接/仿真/人工）

TOOL_REGISTRY = {
    "get_bus_voltage": {
        "func": get_bus_voltage,
        "description": "查询母线电压幅值和相角",
        "device_types": {"bus"},
        "risk_level": "low",
        "requires_net_arg": True,
        "allowed_params": {"bus_id"},
        "required_params": {"bus_id"},
    },
    "get_neighbor_buses": {
        "func": get_neighbor_buses,
        "description": "查询母线的相邻母线",
        "device_types": {"bus"},
        "risk_level": "low",
        "requires_net_arg": True,
        "allowed_params": {"bus_id"},
        "required_params": {"bus_id"},
    },
    "get_line_loading": {
        "func": get_line_loading,
        "description": "查询线路负载率和潮流",
        "device_types": {"line"},
        "risk_level": "low",
        "requires_net_arg": True,
        "allowed_params": {"line_id"},
        "required_params": {"line_id"},
    },
    "get_generator_state": {
        "func": get_generator_state,
        "description": "查询发电机出力和状态",
        "device_types": {"gen"},
        "risk_level": "low",
        "requires_net_arg": True,
        "allowed_params": {"gen_id"},
        "required_params": {"gen_id"},
    },
    "simulate_action": {
        "func": simulate_action,
        "description": "在仿真副本上执行操作并返回结果",
        "device_types": {"bus", "line", "gen", "trafo"},
        "risk_level": "medium",
        "requires_net_arg": True,
        "allowed_params": {"action"},
        "required_params": {"action"},
    },
    "check_constraints": {
        "func": check_constraints,
        "description": "校验电网运行约束（电压越限、线路过载）",
        "device_types": {"bus", "line"},
        "risk_level": "low",
        "requires_net_arg": True,
        "allowed_params": set(),
        "required_params": set(),
    },
    "set_gen_voltage": {
        "func": None,
        "description": "设置发电机电压设定值",
        "device_types": {"gen"},
        "risk_level": "medium",
        "requires_net_arg": False,
        "allowed_params": {"gen_id", "vm_pu"},
        "required_params": {"gen_id", "vm_pu"},
    },
    "set_gen_output": {
        "func": None,
        "description": "设置发电机有功出力",
        "device_types": {"gen"},
        "risk_level": "medium",
        "requires_net_arg": False,
        "allowed_params": {"gen_id", "p_mw"},
        "required_params": {"gen_id", "p_mw"},
    },
    "set_line_status": {
        "func": None,
        "description": "设置线路的投入/退出状态",
        "device_types": {"line"},
        "risk_level": "high",
        "requires_net_arg": False,
        "allowed_params": {"line_id", "in_service"},
        "required_params": {"line_id", "in_service"},
    },
}


def get_available_tools(permission: dict = None) -> list:
    """
    根据权限三元组筛选可用工具。
    
    根据 给分配的权限三元组去获取，
    从预设工具库中去选取，符合权限范围的工具
    参数:
        permission: 权限三元组字典，格式如 ROOT_PERMISSION
                   如果为 None，返回所有工具
    
    返回:
        符合权限的工具名称列表
    """
    if permission is None:
        return list(TOOL_REGISTRY.keys())

    allowed_device_types = permission.get("device_types", set())
    if "all" in allowed_device_types:
        return list(TOOL_REGISTRY.keys())

    available = []
    for name, meta in TOOL_REGISTRY.items():
        # 工具涉及的设备类型必须是权限允许的设备类型的子集
        if meta["device_types"].issubset(allowed_device_types):
            available.append(name)
    return available


def _infer_missing_param(tool_name: str, key: str, context: str = ""):
    """从上下文中尝试补全丢失的必需参数。"""
    if not context:
        return None

    context_l = context.lower()
    patterns = {
        "bus_id": [r"bus\s*(?:[:：])?\s*(\d+)", r"母线\s*(?:[:：])?\s*(\d+)", r"bus\s+(\d+)", r"母线\s+(\d+)"] ,
        "gen_id": [r"gen\s*(?:[:：])?\s*(\d+)", r"发电机\s*(?:[:：])?\s*(\d+)", r"gen\s+(\d+)", r"发电机\s+(\d+)"] ,
        "line_id": [r"line\s*(?:[:：])?\s*(\d+)", r"线路\s*(?:[:：])?\s*(\d+)", r"line\s+(\d+)", r"线路\s+(\d+)"] ,
    }

    for pattern in patterns.get(key, []):
        match = __import__("re").search(pattern, context_l)
        if match:
            try:
                return int(match.group(1))
            except (TypeError, ValueError):
                continue

    return None


def clean_tool_params(tool_name: str, params: dict, context: str = "") -> dict:
    """清洗工具参数：删除非法 key，补全缺失的必需参数。"""
    if tool_name not in TOOL_REGISTRY:
        return {}

    meta = TOOL_REGISTRY[tool_name]
    allowed = meta.get("allowed_params")
    required = meta.get("required_params", set())

    cleaned = dict(params or {}) if isinstance(params, dict) else {}

    if allowed is not None:
        cleaned = {k: v for k, v in cleaned.items() if k in allowed}

    if tool_name == "check_constraints":
        return {}

    for key in list(required):
        if key not in cleaned or cleaned[key] in (None, ""):
            inferred = _infer_missing_param(tool_name, key, context)
            if inferred is not None:
                cleaned[key] = inferred

    return cleaned


def validate_tool_params(tool_name: str, params: dict, context: str = "") -> tuple:
    """
    校验给定工具的参数名是否合法，并检查必需参数是否缺失。

    返回 (True, None) 或 (False, error_message)。
    """
    if tool_name not in TOOL_REGISTRY:
        return False, f"工具 '{tool_name}' 不存在"

    meta = TOOL_REGISTRY[tool_name]
    allowed = meta.get("allowed_params")
    required = meta.get("required_params", set())
    cleaned = clean_tool_params(tool_name, params, context)

    if allowed is not None:
        bad = [k for k in (params or {}).keys() if k not in allowed]
        if bad:
            return False, f"参数 {bad} 不存在，合法参数为 {sorted(list(allowed))}"

    if tool_name == "check_constraints":
        return True, None

    missing = [k for k in sorted(required) if k not in cleaned or cleaned[k] in (None, "")]
    if missing:
        return False, f"工具 '{tool_name}' 缺少必需参数: {missing}"

    return True, None


def call_tool(tool_name: str, net, **kwargs) -> dict:
    """
    统一的工具调用入口。

    在真正调用前先做参数清洗：
    - 删除非法参数
    - 补全缺失的必需参数
    - 对无参数工具直接清空参数
    """
    if tool_name not in TOOL_REGISTRY:
        return {"success": False, "error": f"工具 '{tool_name}' 不存在"}

    context = kwargs.pop("context", "")
    cleaned = clean_tool_params(tool_name, kwargs, context)

    if tool_name == "check_constraints":
        cleaned = {}

    ok, err = validate_tool_params(tool_name, cleaned, context)
    if not ok:
        return {"success": False, "tool": tool_name, "error": err}

    func = TOOL_REGISTRY[tool_name]["func"]
    try:
        for key in ("bus_id", "line_id", "gen_id"):
            if key in cleaned:
                try:
                    val = int(cleaned[key])
                except Exception:
                    continue
                table = None
                if key == "bus_id":
                    table = getattr(net, "bus", None)
                elif key == "line_id":
                    table = getattr(net, "line", None)
                elif key == "gen_id":
                    table = getattr(net, "gen", None)

                if table is not None and val not in table.index and (val - 1) in table.index:
                    cleaned[key] = val - 1

        result = func(net, **cleaned)
        return {"success": True, "tool": tool_name, "result": result}
    except Exception as e:
        return {"success": False, "tool": tool_name, "error": str(e)}