"""仅供测试使用的固定 LLM 响应，不参与真实运行。"""

import json
import logging
import re
from llm.client import LLMClient

logger = logging.getLogger(__name__)


class MockLLMClient(LLMClient):
    """
    模拟 LLM 客户端。
    
    Step 3 新增 failure_mode:
      设为 True 时，对"生成调整方案"类任务首次返回一个不够强的修正，
      使仿真/约束校验失败，从而触发 S4 重规划。
      重规划时（prompt 中包含"重规划"关键词）返回更强的修正。
    """

    def __init__(self, failure_mode: bool = False):
        self.failure_mode = failure_mode
        # 追踪每个任务类型的调用次数，用于在重规划时返回不同结果
        self._call_counts = {}
        logger.info(
            f"使用 MockLLMClient（failure_mode={failure_mode}）"
        )

    def complete(self, system_prompt: str, user_prompt: str, temperature: float = 0.7, source: str = "unknown", max_tokens: int = None) -> str:
        if "任务分解专家" in system_prompt:
            return self._planner_response(user_prompt)
        elif "编排智能体" in system_prompt:
            return self._orchestration_response(user_prompt)
        elif "执行智能体" in system_prompt:
            return self._execution_response(user_prompt)
        else:
            return json.dumps({"message": "mock fallback"}, ensure_ascii=False)

    def _planner_response(self, user_prompt: str) -> str:
        bus_match = re.search(r"目标设备:\s*Bus\s*(\d+)", user_prompt) or re.search(r"Bus\s*(\d+)", user_prompt)
        target_bus = int(bus_match.group(1)) if bus_match else 13

        tasks = {
            "tasks": [
                {"id": "t1", "description": "查询目标节点电压",
                 "dependencies": [], "devices": [target_bus],
                 "device_type": "bus", "voltage_level": "MV"},
                {"id": "t2", "description": "查询邻近节点状态",
                 "dependencies": [], "devices": [target_bus],
                 "device_type": "bus", "voltage_level": "MV"},
                {"id": "t3", "description": "查询相关线路负载",
                 "dependencies": [], "devices": [target_bus],
                 "device_type": "line", "voltage_level": "MV"},
                {"id": "t4", "description": "生成调整方案",
                 "dependencies": ["t1", "t2", "t3"], "devices": [target_bus],
                 "device_type": "gen", "voltage_level": "MV"},
                {"id": "t5", "description": "仿真验证方案",
                 "dependencies": ["t4"], "devices": [target_bus],
                 "device_type": "gen", "voltage_level": "MV"},
                {"id": "t6", "description": "验证全网约束",
                 "dependencies": ["t7"], "devices": [target_bus],
                 "device_type": "bus", "voltage_level": "MV"},
                {"id": "t7", "description": "执行发电机电压调节",
                 "dependencies": ["t5"], "devices": [target_bus],
                 "device_type": "gen", "voltage_level": "MV"},
            ]
        }
        return json.dumps(tasks, ensure_ascii=False)

    def _orchestration_response(self, user_prompt: str) -> str:
        bus_match = re.search(r"设备.*?(\d+)", user_prompt)
        target_bus = int(bus_match.group(1)) if bus_match else 13

        is_replan = "重规划" in user_prompt or "replan" in user_prompt.lower()

        # ---- 根据任务关键词路由 ----
        # 优先识别"调整/生成方案"的任务，避免被 prior_results 中的关键词误导
        if "执行发电机电压调节" in user_prompt:
            return json.dumps({"instructions": [{
                "tool": "set_gen_voltage",
                "params": {"gen_id": 0, "vm_pu": 1.08},
                "description": "执行 Gen 0 电压调节",
                "expected_result": "目标母线电压恢复且全网约束满足",
            }]}, ensure_ascii=False)

        if "调整方案" in user_prompt or ("生成" in user_prompt and "方案" in user_prompt):
            # ---- failure_mode 关键逻辑 ----
            # 首次调用：如果 failure_mode=True，返回过小的调整量（故意不够）
            # 重规划调用：返回更大的调整量
            count_key = "adjust_plan"
            self._call_counts[count_key] = self._call_counts.get(count_key, 0) + 1

            if self.failure_mode and not is_replan:
                # 首次：微弱调整，不足以修复电压（故意设置为更保守的 1.00）
                # 且把仿真操作设为无效类型，保证 simulate_action 返回 failure，触发 S4
                vm_target = 1.00
                invalid_action = True
                logger.info("  [MockLLM] failure_mode: 首次返回不足的调整量")
            else:
                # 正常或重规划：足够的调整
                vm_target = 1.08
                if is_replan:
                    logger.info("  [MockLLM] 重规划: 返回更强的调整量")
            # 根据是否需要故意触发失败，生成不同的仿真 action
            if 'invalid_action' in locals() and invalid_action:
                sim_action = {"type": "invalid_action", "gen_id": 0, "vm_pu": vm_target}
            else:
                sim_action = {"type": "set_gen_voltage", "gen_id": 0, "vm_pu": vm_target}

            return json.dumps({"instructions": [
                {"tool": "get_generator_state", "params": {"gen_id": 0},
                 "description": "查询 Gen 0 当前状态",
                 "expected_result": "获取发电机参数"},
                {"tool": "simulate_action",
                 "params": {"action": sim_action},
                 "description": f"仿真提高 Gen 0 电压至 {vm_target}",
                 "expected_result": "仿真收敛且目标母线电压改善"},
            ]}, ensure_ascii=False)

        elif "仿真" in user_prompt:
            # t5 本身就是验证，用已在 t4 中 simulate 过的方案
            # 再做一次仿真确认
            count_key = "simulate"
            self._call_counts[count_key] = self._call_counts.get(count_key, 0) + 1

            if self.failure_mode and self._call_counts.get("adjust_plan", 0) <= 1 and not is_replan:
                vm_target = 1.02
            else:
                vm_target = 1.08

            return json.dumps({"instructions": [{
                "tool": "simulate_action",
                "params": {"action": {"type": "set_gen_voltage",
                                       "gen_id": 0, "vm_pu": vm_target}},
                "description": "仿真验证调压方案",
                "expected_result": "仿真收敛且目标母线电压改善",
            }]}, ensure_ascii=False)

        elif "约束" in user_prompt or "验证" in user_prompt:
            return json.dumps({"instructions": [{
                "tool": "check_constraints",
                "params": {},
                "description": "校验全网运行约束",
                "expected_result": "所有母线电压和线路负载率在限值内",
            }]}, ensure_ascii=False)

        # 其次处理其他查询类任务
        elif "查询目标节点电压" in user_prompt or ("目标" in user_prompt and "电压" in user_prompt):
            return json.dumps({"instructions": [{
                "tool": "get_bus_voltage",
                "params": {"bus_id": target_bus},
                "description": f"查询 Bus {target_bus} 电压",
                "expected_result": "获取电压数据",
            }]}, ensure_ascii=False)

        elif "邻近" in user_prompt or "邻居" in user_prompt:
            return json.dumps({"instructions": [{
                "tool": "get_neighbor_buses",
                "params": {"bus_id": target_bus},
                "description": f"查询 Bus {target_bus} 的相邻母线",
                "expected_result": "获取邻近节点列表",
            }]}, ensure_ascii=False)

        elif "线路" in user_prompt and "负载" in user_prompt:
            return json.dumps({"instructions": [{
                "tool": "get_line_loading",
                "params": {"line_id": 0},
                "description": "查询线路负载率",
                "expected_result": "获取线路负载数据",
            }]}, ensure_ascii=False)

        else:
            return json.dumps({"instructions": [{
                "tool": "get_bus_voltage",
                "params": {"bus_id": target_bus},
                "description": "查询目标母线电压",
                "expected_result": "获取电压数据",
            }]}, ensure_ascii=False)

    def _execution_response(self, user_prompt: str) -> str:
        tool_match = re.search(r'"tool":\s*"(\w+)"', user_prompt)
        tool_name = tool_match.group(1) if tool_match else "get_bus_voltage"

        if tool_name.startswith("get_") or tool_name == "check_constraints":
            strategy = "direct"
        elif tool_name == "simulate_action":
            strategy = "simulate"
        else:
            strategy = "simulate"

        params_match = re.search(r'"params":\s*(\{[^}]*\})', user_prompt)
        if params_match:
            try:
                params = json.loads(params_match.group(1))
            except json.JSONDecodeError:
                params = {}
        else:
            params = {}

        return json.dumps({
            "tool_sequence": [{"tool": tool_name, "params": params}],
            "strategy": strategy,
        }, ensure_ascii=False)


