"""Shared physical safety boundary. Knows no method names or restart policies."""
import copy
import json
import math
from grid import tools
from grid.goal import goal_status
from llm.client import LLMServiceUnavailable
from config.settings import CASE39_TEMPERATURE, CASE39_MAX_TOKENS


class DispatchKernel:
    def __init__(self, network, goal, instruction, llm):
        self.network, self.goal, self.instruction, self.llm = network, goal, instruction, llm
        self.initial_net = copy.deepcopy(network.net)
        self.proposed = []
        self.replanned = []
        self.wasted_indices = set()
        self.clipped_actions = []
        self.scope_hit = 0
        self.illegal = 0
        self.catalog_tokens = 0
        self.duplicates = 0
        self.seen = set()
        self.feedback = ""
        self.attempts = 0
        self.first_attempt_deviation = False
        self.api_calls = 0
        self._reset_at = 0
        self.audit = []

    def status(self, net=None):
        self.audit.append("grid.goal.goal_status")
        return goal_status(self.network.net if net is None else net, self.goal)

    def catalog(self, permission):
        catalog = tools.get_tool_catalog(self.network.net, tools.get_available_tools(permission.to_dict()),
                                         permission=permission.to_dict())
        # Uniform public observation policy, never consult device_limits. The fixture's
        # initial healthy voltage happens to equal a private cap. Report healthy voltages
        # categorically for every device/method; retain exact violation measurements.
        def observation(value):
            return "within_goal_range" if self.goal.vmin <= value <= self.goal.vmax else value
        for gen in catalog["generators"]:
            gen["vm_pu"] = observation(gen["vm_pu"])
        catalog["bus_voltages"] = {i: observation(v) for i, v in catalog["bus_voltages"].items()
                                   if "bus" not in permission.scope or int(i) in permission.scope["bus"]}
        return catalog

    def ask(self, task, permission):
        from llm.prompts import CASE39_LEAF_SYSTEM
        catalog = self.catalog(permission)
        self.catalog_tokens += (len(json.dumps(catalog, ensure_ascii=False, default=str)) + 3) // 4
        context = {"instruction": {"mission": self.instruction, "task": task.description,
                                   "devices": task.devices, "device_type": task.device_type},
                   "current_goal": self.status(), "used_actions": len(self.network.action_log),
                   "budget": self.goal.max_real_actions, "feedback": self.feedback, "catalog": catalog}
        self.api_calls += 1
        response = self.llm.complete_json(CASE39_LEAF_SYSTEM, json.dumps(context, ensure_ascii=False, default=str),
                                          temperature=CASE39_TEMPERATURE, max_tokens=CASE39_MAX_TOKENS,
                                          source="orchestration_agent")
        if not isinstance(response, dict):
            raise ValueError("LLM response must be an object")
        if response.get("error") == "LLM_ERROR":
            if response.get("retryable"):
                raise LLMServiceUnavailable(response.get("message", "LLM unavailable"))
            raise ValueError(response.get("message", "Invalid LLM response"))
        actions = response.get("actions")
        self.proposed.append({"task_id": task.id, "attempt": self.attempts, "actions": actions})
        self.validate(actions, permission)
        return actions

    def validate(self, actions, permission):
        error = None
        scope_error = False
        if not isinstance(actions, list) or not actions:
            error = "actions must be a nonempty array"
        else:
            # Validate each proposed action once, without coercion or repair.
            for action in actions:
                if not isinstance(action, dict) or action.get("type") != "set_gen_voltage":
                    error = "Only set_gen_voltage is allowed"
                    break
                vm = action.get("vm_pu")
                if isinstance(vm, bool) or not isinstance(vm, (int, float)) or not math.isfinite(vm):
                    error = "vm_pu must be finite numeric"
                    break
                ok, error = tools.validate_tool_call("set_gen_voltage", {k: v for k, v in action.items() if k != "type"},
                                                     self.network.net, permission.to_dict())
                if not ok:
                    scope_error = "权限" in error
                    break
                bus = int(self.network.net.gen.at[action["gen_id"], "bus"])
                if int(self.network.net.bus.at[bus, "zone"]) in self.goal.forbidden_regions:
                    error, scope_error = "设备超出允许操作区域权限", True
                    break
        if error:
            self.illegal += 1
            self.scope_hit += int(scope_error)
            # Existing public counter remains usable; avoid double counting scope validation.
            if not scope_error or "允许操作区域" in error:
                tools.record_illegal_call()
            raise ValueError(error)

    def _record_call(self, tool, params):
        signature = json.dumps({"tool": tool, "params": params}, sort_keys=True)
        if self.attempts > 1 and signature in self.seen:
            self.duplicates += 1
        self.seen.add(signature)

    def joint_gate(self, actions):
        if not actions or len(self.network.action_log) + len(actions) > self.goal.max_real_actions:
            return {"goal_met": False, "error": "真实动作预算不足或空方案"}
        before = self.status()
        result = None
        for count in range(1, len(actions) + 1):
            self._record_call("simulate_action", {"action": actions[:count]})
            self.audit.append("grid.tools.simulate_action")
            result = tools.simulate_action(self.network.net, actions[:count], self.goal, len(self.network.action_log))
            if not result.get("success"):
                return {"goal_met": False, "error": result.get("error", "仿真失败")}
            after = self.status(result["net_copy"])
            if not tools.constraints_not_worse(before, after):
                return {"goal_met": False, "error": "预演中间步骤新增或恶化越限", "prefix": count}
            before = after
        return {k: v for k, v in result.items() if k not in ("net_copy", "bus_voltages", "line_loadings")}

    def commit_action(self, action, permission):
        self.validate([action], permission)
        if len(self.network.action_log) >= self.goal.max_real_actions:
            return {"success": False, "error": "真实动作预算耗尽"}
        before = self.status()
        old = float(self.network.net.gen.at[action["gen_id"], "vm_pu"])
        index = len(self.network.action_log)
        self._record_call(action["type"], {k: v for k, v in action.items() if k != "type"})
        self.audit.append("grid.network.PowerNetwork.set_gen_voltage")
        try:
            self.network.set_gen_voltage(action["gen_id"], action["vm_pu"])
        except Exception as exc:
            self.wasted_indices.add(index)
            return {"success": False, "error": f"真实动作失败: {type(exc).__name__}"}
        actual = float(self.network.net.gen.at[action["gen_id"], "vm_pu"])
        clipped = abs(actual - action["vm_pu"]) > 1e-6
        if clipped:
            self.clipped_actions.append({"index": index, "gen_id": action["gen_id"],
                                         "requested": action["vm_pu"], "actual": actual})
            self.wasted_indices.add(index)
        if abs(actual - old) < 1e-6:
            self.wasted_indices.add(index)
        after = self.status()
        if not tools.constraints_not_worse(before, after):
            self.wasted_indices.add(index)
            return {"success": False, "error": "真实操作新增或恶化越限"}
        if clipped:
            return {"success": False, "error": (
                f"PARAMETER 偏差：gen_id={action['gen_id']} 请求 vm_pu={action['vm_pu']}，"
                f"实测仅达到 {actual:.4f}。该设备本轮可能已达到其物理调节上限，"
                f"请勿再对同一设备发起更高目标值，应改为提高其他允许操作范围内机组的调节量以补偿剩余缺口。"
            )}
        return {"success": True}

    def failure(self, error):
        self.feedback = str(error)
        if self.attempts == 1:
            self.first_attempt_deviation = True

    def rollback(self):
        self.wasted_indices.update(range(self._reset_at, len(self.network.action_log)))
        self._reset_at = len(self.network.action_log)
        self.network.net = copy.deepcopy(self.initial_net)
        self.network.mutation_history.clear()

    def zone3_touched(self):
        return any(int(self.network.net.bus.at[int(self.network.net.gen.at[i, "bus"]), "zone"]) == 3
                   for tool, i, _ in self.network.action_log if tool == "set_gen_voltage")

    def success(self):
        return bool(self.status()["goal_met"] and len(self.network.action_log) <= self.goal.max_real_actions
                    and not self.zone3_touched())
