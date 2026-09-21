"""Fair case39 regional restoration evaluation. No witness enters an agent prompt."""

import argparse
import copy
import csv
import json
import time
from pathlib import Path

from config.settings import CASE39_TEMPERATURE
from grid.goal import Goal, goal_status
from grid.scenario_case39 import INSTRUCTION, make_network, validate_scenario
from grid.tools import (check_constraints, constraints_not_worse, get_tool_catalog,
                        reset_tool_counters, get_tool_counters, simulate_action,
                        validate_tool_call, get_available_tools)
from grid.topology import regional_scope, regional_tree_depth
from llm.client import RealLLMClient

METHODS = ("hierarchical", "two_layer", "two_layer_full_restart")
FIELDS = ("method", "repeat", "success", "real_actions_used", "wasted_actions",
          "duplicate_tool_calls", "illegal_tool_calls", "catalog_tokens", "total_tokens",
          "replanned_tasks", "zone3_touched", "scope_hit", "tree_depth", "d0",
          "proposed_actions", "actual_actions", "error", "elapsed_seconds")


def _targets(net, goal):
    status = goal_status(net, goal)
    by_zone = {}
    for zone, violations in status["violations_by_zone"].items():
        voltage = [v for v in violations if v["type"] == "voltage_low"]
        if voltage:
            minimum = min(v["value"] for v in voltage)
            by_zone[int(zone)] = [v["bus_id"] for v in voltage if abs(v["value"] - minimum) < 1e-5]
    return by_zone


def _allowed_scope(net, goal):
    forbidden = {int(i) for i in goal.forbidden_regions}
    return {kind: {int(i) for i, row in getattr(net, kind).iterrows()
                   if int(net.bus.at[int(row.bus if kind == "gen" else
                       row.from_bus if kind == "line" else row.hv_bus), "zone"]) not in forbidden}
            for kind in ("gen", "line", "trafo")}


class Case39Trial:
    def __init__(self, method, llm):
        self.method = method
        self.llm = llm
        self.network = make_network()
        self.goal = Goal.from_instruction(INSTRUCTION)
        self.initial_net = copy.deepcopy(self.network.net)
        self.proposed = []
        self.seen = set()
        self.duplicates = 0
        self.catalog_tokens = 0
        self.replanned = []
        self.scope_hit = 0
        self.wasted_indices = set()
        self.feedback = ""
        self.target_by_zone = _targets(self.network.net, self.goal)
        targets = [i for zone in sorted(self.target_by_zone) for i in self.target_by_zone[zone]]
        self.scope = regional_scope(self.network.net, targets, self.goal.forbidden_regions)
        self.depth = regional_tree_depth(self.scope, self.network.net) if method == "hierarchical" else 2
        self.d0 = max(item["d0"] for item in self.scope["d0_info"])
        self.global_scope = _allowed_scope(self.network.net, self.goal)

    def _permission(self, region=None):
        net = self.network.net
        if self.method == "hierarchical":
            scope = {kind: set(self.scope[kind]) for kind in ("bus", "line", "gen", "trafo")}
            if region is not None:
                scope["gen"] = {i for i in scope["gen"] if int(net.bus.at[int(net.gen.at[i, "bus"]), "zone"]) == region}
                scope["bus"] = {i for i in scope["bus"] if int(net.bus.at[i, "zone"]) == region}
        else:
            scope = {"bus": {int(i) for i in net.bus.index},
                     "line": {int(i) for i in net.line.index},
                     "gen": {int(i) for i in net.gen.index},
                     "trafo": {int(i) for i in net.trafo.index}}
        return {"device_types": {"bus", "line", "gen", "trafo"}, "scope": scope}

    def _ask(self, region=None, retained=None):
        permission = self._permission(region)
        catalog = get_tool_catalog(self.network.net, get_available_tools(permission),
                                   permission=permission)
        serialized = json.dumps(catalog, ensure_ascii=False, default=str)
        self.catalog_tokens += (len(serialized) + 3) // 4
        context = {"instruction": INSTRUCTION, "region": region, "catalog": catalog,
                   "current_goal": goal_status(self.network.net, self.goal),
                   "used_actions": len(self.network.action_log), "budget": self.goal.max_real_actions,
                   "retained_other_region_plan": retained or [], "feedback": self.feedback}
        system = ("你是电网调度智能体。仅使用目录中的 gen_id，先给完整联合方案供仿真。"
                  "只允许 set_gen_voltage，不得操作区域3。输出 JSON: "
                  '{"actions":[{"type":"set_gen_voltage","gen_id":整数,"vm_pu":数值}]}。'
                  "不得猜测隐藏设备限制。")
        response = self.llm.complete_json(system, json.dumps(context, ensure_ascii=False, default=str),
                                          temperature=CASE39_TEMPERATURE, source="case39", max_tokens=1024)
        if response.get("error") == "LLM_ERROR":
            raise RuntimeError(response.get("message", "LLM_ERROR"))
        actions = response.get("actions", [])
        self.proposed.append({"region": region, "actions": actions})
        signature = json.dumps(actions, sort_keys=True, default=str)
        if signature in self.seen:
            self.duplicates += 1
        self.seen.add(signature)
        if not isinstance(actions, list) or not actions:
            raise ValueError("模型未返回动作列表")
        for action in actions:
            if action.get("type") != "set_gen_voltage":
                raise ValueError("只允许发电机电压设定值")
        validation_permission = {**permission, "scope": dict(permission["scope"])}
        validation_permission["scope"]["gen"] = permission["scope"]["gen"] & self.global_scope["gen"]
        ok, error = validate_tool_call("simulate_action", {"action": actions}, self.network.net, validation_permission)
        if not ok:
            self.scope_hit += int("权限" in error)
            raise ValueError(error)
        for action in actions:
            bus = int(self.network.net.gen.at[action["gen_id"], "bus"])
            if int(self.network.net.bus.at[bus, "zone"]) in self.goal.forbidden_regions:
                raise ValueError("区域3设备不得操作")
        return actions

    def _joint_gate(self, actions):
        if len(self.network.action_log) + len(actions) > self.goal.max_real_actions:
            return {"goal_met": False, "budget_error": "真实动作预算不足"}
        before = check_constraints(self.network.net)
        for count in range(1, len(actions) + 1):
            prefix = simulate_action(self.network.net, actions[:count], self.goal,
                                     len(self.network.action_log))
            if not prefix.get("success") or not constraints_not_worse(before, check_constraints(prefix["net_copy"])):
                return {"goal_met": False, "not_worse": False, "prefix": count,
                        "violations": prefix.get("violations", []), "error": prefix.get("error")}
        result = simulate_action(self.network.net, actions, self.goal, len(self.network.action_log))
        return {k: v for k, v in result.items() if k not in ("net_copy", "bus_voltages", "line_loadings")}

    def _commit(self, actions):
        before = check_constraints(self.network.net)
        for action in actions:
            if len(self.network.action_log) >= self.goal.max_real_actions:
                return "真实动作预算不足"
            old_vm = float(self.network.net.gen.at[action["gen_id"], "vm_pu"])
            feedback = self.network.set_gen_voltage(action["gen_id"], action["vm_pu"])
            actual = self.network.get_generator_state(action["gen_id"])["vm_pu"]
            after = check_constraints(self.network.net)
            if not constraints_not_worse(before, after):
                self.wasted_indices.add(len(self.network.action_log) - 1)
                return "真实操作新增或恶化越限"
            before = after
            if abs(actual - action["vm_pu"]) > 1e-3:
                self.wasted_indices.add(len(self.network.action_log) - 1)
                return f"PARAMETER 偏差: gen{action['gen_id']} 请求 {action['vm_pu']}，实际 {actual}"
            if abs(actual - old_vm) < 1e-6:
                self.wasted_indices.add(len(self.network.action_log) - 1)
        return None

    def run(self):
        error = "未达到验收标准"
        retained = []
        for attempt in range(3):
            if self.method == "two_layer_full_restart" and attempt:
                self.wasted_indices.update(range(getattr(self, "_last_restart_action_count", 0),
                                                 len(self.network.action_log)))
                self._last_restart_action_count = len(self.network.action_log)
                self.network.net = copy.deepcopy(self.initial_net)
                retained = []
            try:
                if self.method == "hierarchical":
                    # Region 2 commits first; on a region 1 deviation, retain completed region 2.
                    if not retained:
                        region2 = self._ask(2)
                    else:
                        region2 = []
                    region1 = self._ask(1, retained=region2 or retained)
                    actions = region2 + region1
                else:
                    actions = self._ask()
                gate = self._joint_gate(actions)
                if not gate.get("goal_met"):
                    self.feedback = json.dumps(gate, ensure_ascii=False, default=str)
                    error = "联合仿真未达标"
                    self.replanned.append("region1" if self.method == "hierarchical" and retained else "all")
                    continue
                if self.method == "hierarchical" and region2:
                    error = self._commit(region2)
                    if error:
                        self.feedback = error
                        self.replanned.append("region2")
                        continue
                    retained = region2
                    actions = region1
                error = self._commit(actions)
                if error:
                    self.feedback = error + "; 当前断面=" + json.dumps(goal_status(self.network.net, self.goal), ensure_ascii=False)
                    self.replanned.append("region1" if self.method == "hierarchical" else "all")
                    continue
                if goal_status(self.network.net, self.goal)["goal_met"]:
                    return True, ""
                self.feedback = json.dumps(goal_status(self.network.net, self.goal), ensure_ascii=False)
                error = "真实执行后全网未达标"
                self.replanned.append("region1" if self.method == "hierarchical" else "all")
            except (ValueError, RuntimeError) as exc:
                error = str(exc)
                self.feedback = error
                self.replanned.append("region1" if self.method == "hierarchical" and retained else "all")
        return False, error


def run_once(method, repeat, llm=None):
    started = time.perf_counter()
    reset_tool_counters()
    client = llm or RealLLMClient()
    trial = Case39Trial(method, client)
    success, error = trial.run()
    status = goal_status(trial.network.net, trial.goal)
    touched = any(int(trial.network.net.bus.at[int(trial.network.net.gen.at[item[1], "bus"]), "zone"]) == 3
                  for item in trial.network.action_log if item[0] == "set_gen_voltage")
    success = bool(success and status["goal_met"] and len(trial.network.action_log) <= trial.goal.max_real_actions and not touched)
    return {"method": method, "repeat": repeat, "success": int(success),
            "real_actions_used": len(trial.network.action_log), "wasted_actions": len(trial.wasted_indices),
            "duplicate_tool_calls": trial.duplicates,
            "illegal_tool_calls": get_tool_counters()["illegal_tool_calls"],
            "catalog_tokens": trial.catalog_tokens, "total_tokens": getattr(client, "total_tokens", 0),
            "replanned_tasks": json.dumps(trial.replanned, ensure_ascii=False),
            "zone3_touched": int(touched), "scope_hit": trial.scope_hit,
            "tree_depth": trial.depth, "d0": round(trial.d0, 4),
            "proposed_actions": json.dumps(trial.proposed, ensure_ascii=False),
            "actual_actions": json.dumps(trial.network.action_log, ensure_ascii=False),
            "error": error, "elapsed_seconds": round(time.perf_counter() - started, 3)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", default="case39_results.csv")
    args = parser.parse_args()
    if args.repeats < 10:
        parser.error("正式比较每种方法至少运行10次")
    validate_scenario()
    rows = []
    for repeat in range(1, args.repeats + 1):
        for method in METHODS:
            row = run_once(method, repeat)
            rows.append(row)
            with Path(args.output).open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            print(f"{method} #{repeat}: success={row['success']}, actions={row['real_actions_used']}", flush=True)
    for method in METHODS:
        group = [row for row in rows if row["method"] == method]
        print(f"{method}: {sum(row['success'] for row in group)}/{len(group)}")


if __name__ == "__main__":
    main()
