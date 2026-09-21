"""Legacy case14 smoke comparison; not the case39 regional evaluation."""

import argparse
import copy
import csv
import json
import logging
import os
import time
from pathlib import Path

import pandapower as pp

from agents.orchestration_agent import OrchestrationAgent
from agents.permission import Permission
from agents.planner import Planner
from agents.root_agent import RootAgent
from agents.task import Task
from grid.network import PowerNetwork
from grid.goal import Goal
from grid.tools import check_constraints, get_tool_counters, reset_tool_counters
from llm.client import LLMServiceUnavailable, RealLLMClient


METHODS = ("hierarchical", "two_layer")
INSTRUCTION = (
    "Bus 14 电压过低，且系统多处母线欠压。请定位相互独立的调压设备，"
    "先查询并仿真，再只用发电机电压设定值恢复：最终 Bus 14 至少 1.00 p.u.，"
    "全网母线电压在 [0.95, 1.10] p.u.、线路负载率不超过 100%。"
    "最多执行两次真实发电机电压调整。中间步骤可以仍有原有欠压，但不能新增或恶化越限；"
    "最后重新校验全网。"
)
FIELDS = ["method", "repeat", "success", "physical_success", "workflow_success",
          "action_count", "actions", "proposed_actions", "illegal_tool_calls", "duplicate_tool_calls",
          "total_tokens", "elapsed_seconds", "final_bus14_voltage", "tree_depth",
          "d0", "replan_count", "failed_tasks", "error"]


def make_network() -> PowerNetwork:
    network = PowerNetwork()
    # 评测注入直接作用于初始断面，不经过真实动作接口。
    network.net.gen.at[0, "vm_pu"] = 0.90
    network.net.gen.at[3, "vm_pu"] = 0.90
    network._run_power_flow()
    if check_constraints(network.net)["all_satisfied"]:
        raise ValueError("比较场景的初始状态没有违规")
    return network


def validate_scenario() -> None:
    """离线验证题目有解；可行参数不传给两个被评测方法。"""
    network = make_network()
    witness = copy.deepcopy(network.net)
    witness.gen.at[0, "vm_pu"] = 1.04
    witness.gen.at[3, "vm_pu"] = 1.04
    pp.runpp(witness, algorithm="nr", init="results")
    if witness.res_bus.at[13, "vm_pu"] < 1.0 or not check_constraints(witness)["all_satisfied"]:
        raise ValueError("比较场景未找到已验证的两动作可行解")


def collect_proposed_actions(value) -> list[dict]:
    """从嵌套执行结果中收集模型提出的动作，包括仿真被拒的动作。"""
    actions = []
    def visit(node):
        if isinstance(node, list):
            for item in node:
                visit(item)
        elif isinstance(node, dict):
            instruction = node.get("instruction")
            if isinstance(instruction, dict) and instruction.get("tool") in {
                "simulate_action", "set_gen_voltage", "set_gen_output", "set_line_status"
            }:
                actions.append({"tool": instruction["tool"], "params": instruction.get("params", {})})
            for key in ("result", "execution_log", "execution_results", "child_results", "replan_result"):
                if key in node:
                    visit(node[key])
    visit(value)
    return actions


def run_once(method: str, repeat: int) -> dict:
    started = time.perf_counter()
    reset_tool_counters()
    network = None
    llm = None
    root = None
    physical_success = False
    workflow_success = False
    final_voltage = ""
    tree_depth = ""
    d0 = ""
    replan_count = 0
    failed_tasks = ""
    error = ""
    infrastructure_error = False
    proposed_actions = []
    try:
        network = make_network()
        goal = Goal.from_instruction(INSTRUCTION, 13)
        llm = RealLLMClient()
        if method == "hierarchical":
            plan = Planner(network, llm).plan(INSTRUCTION)
            tree_depth = plan["tree_depth"]
            d0 = plan["d0_info"]["d0"]
            root = RootAgent(network, plan["dag"], llm, tree_depth, plan["d0_info"],
                             plan["certainty"], permission_shrink=True, replan_mode="local",
                             mission=plan["instruction"], goal=goal)
            result = root.execute()
            proposed_actions = collect_proposed_actions(result)
            if any(entry["result"].get("retryable") for entry in result["execution_log"]):
                raise LLMServiceUnavailable("分层方法调用期间 LLM 服务暂时不可用")
            workflow_success = bool(result["success"])
            replan_count = len(result["replan_log"])
            failed_tasks = ",".join(task_id for task_id, status in result["task_status"].items()
                                    if status != "completed")
            if not workflow_success:
                failures = [f"{item['task_id']}: {item['result'].get('error')}"
                            for item in result["execution_log"] if not item["result"].get("success")]
                error = "; ".join(failures[-3:]) or "任务图未完成"
        elif method == "two_layer":
            tree_depth = 2
            agent = OrchestrationAgent(
                agent_id="direct_orchestration", task=Task(id="direct", description=f"执行调度：{INSTRUCTION}",
                                                          devices=[13], device_type="bus"),
                permission=Permission.root_permission(), network=network, llm=llm,
                current_depth=0, max_depth=2, permission_shrink=False,
                mission=INSTRUCTION, goal=goal,
            )
            result = agent.execute()
            proposed_actions = collect_proposed_actions(result)
            if result.get("retryable"):
                raise LLMServiceUnavailable("两层方法调用期间 LLM 服务暂时不可用")
            workflow_success = bool(result["success"])
            failed_tasks = "" if workflow_success else "direct"
            error = result.get("error") or "" if not workflow_success else ""
        else:
            raise ValueError(f"未知方法: {method}")

        pp.runpp(network.net, algorithm="nr", init="results")
        final_voltage = round(float(network.net.res_bus.at[13, "vm_pu"]), 6)
        physical_success = bool(final_voltage >= 1.0 and check_constraints(network.net)["all_satisfied"])
    except LLMServiceUnavailable as exc:
        infrastructure_error = True
        error = str(exc)
        logging.warning("比较试验暂停: %s / %s: %s", method, repeat, exc)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        logging.exception("比较试验失败: %s / %s", method, repeat)

    actions = network.mutation_history if network else []
    allowed_actions = len(actions) <= 2 and all(item[0] == "set_gen_voltage" for item in actions)
    return {
        "method": method, "repeat": repeat,
        "success": int(physical_success and workflow_success and allowed_actions),
        "physical_success": int(physical_success), "workflow_success": int(workflow_success),
        "action_count": len(actions), "actions": json.dumps(actions, ensure_ascii=False),
        "proposed_actions": json.dumps(proposed_actions, ensure_ascii=False),
        "illegal_tool_calls": get_tool_counters()["illegal_tool_calls"],
        "duplicate_tool_calls": root.duplicate_tool_calls if root else 0,
        "total_tokens": llm.total_tokens if llm else 0,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "final_bus14_voltage": final_voltage, "tree_depth": tree_depth, "d0": d0,
        "replan_count": replan_count, "failed_tasks": failed_tasks, "error": error,
        "infrastructure_error": infrastructure_error,
    }


def print_summary(rows: list[dict]) -> None:
    print("仅为 case14 冒烟诊断；不能作为 case39 分区域恢复的公平方法比较。")
    print("\n方法 | 严格成功 | 物理达标 | 流程完成 | token | 耗时(s)")
    print("--- | ---: | ---: | ---: | ---: | ---:")
    for method in METHODS:
        group = [row for row in rows if row["method"] == method]
        print(f"{method} | {sum(row['success'] for row in group)}/{len(group)} | "
              f"{sum(row['physical_success'] for row in group)}/{len(group)} | "
              f"{sum(row['workflow_success'] for row in group)}/{len(group)} | "
              f"{sum(row['total_tokens'] for row in group)} | "
              f"{sum(row['elapsed_seconds'] for row in group):.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("compare_two_step.csv"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats 必须大于 0")
    if not os.environ.get("LLM_API_KEY"):
        parser.error("未设置 LLM_API_KEY")
    try:
        RealLLMClient().check_connection()
    except (ValueError, LLMServiceUnavailable) as exc:
        parser.error(str(exc))
    validate_scenario()
    if args.output.exists() and not args.resume:
        parser.error("输出文件已存在；请指定新文件名或使用 --resume")
    rows = []
    has_header = False
    if args.resume and args.output.exists():
        with args.output.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != FIELDS:
                parser.error("已有 CSV 列名不匹配，请指定新文件")
            has_header = True
            for row in reader:
                for key in ("repeat", "success", "physical_success", "workflow_success", "total_tokens"):
                    row[key] = int(row[key])
                row["elapsed_seconds"] = float(row["elapsed_seconds"])
                rows.append(row)
    completed = {(row["method"], row["repeat"]) for row in rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a" if has_header else "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if not has_header:
            writer.writeheader()
            handle.flush()
        try:
            for repeat in range(1, args.repeats + 1):
                order = METHODS if repeat % 2 else tuple(reversed(METHODS))
                for method in order:
                    if (method, repeat) in completed:
                        continue
                    row = run_once(method, repeat)
                    if row.pop("infrastructure_error"):
                        print(f"LLM 服务暂时不可用；停在 {method}/{repeat}，已完成的结果保留。"
                              "稍后用 --resume 继续。")
                        raise SystemExit(2)
                    writer.writerow(row)
                    handle.flush()
                    rows.append(row)
                    print(f"{method}/{repeat}: 严格成功={row['success']}, 动作={row['action_count']}, "
                          f"token={row['total_tokens']}, {row['elapsed_seconds']:.2f}s")
        except KeyboardInterrupt:
            print("\n已中断；已完成的结果保留，可用 --resume 继续。")
            raise SystemExit(130)
    print_summary(rows)
    print(f"明细已写入 {args.output.resolve()}")


if __name__ == "__main__":
    main()
