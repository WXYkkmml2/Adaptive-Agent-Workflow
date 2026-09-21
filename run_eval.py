"""运行两种低压场景的消融评测，并逐次写入 CSV。"""

import argparse
import csv
import logging
import os
import time
from pathlib import Path

import pandapower as pp

from agents.planner import Planner
from agents.root_agent import RootAgent
from agents.orchestration_agent import OrchestrationAgent
from agents.permission import Permission
from agents.task import Task
from grid.network import PowerNetwork
from grid.tools import check_constraints, get_tool_counters, reset_tool_counters
from grid.voltage_control import find_voltage_action
from llm.client import RealLLMClient, LLMServiceUnavailable


SCENARIOS = {
    "bus14_low": {"target_bus": 13, "initial_gen_voltage": 0.90},
    "bus13_low": {"target_bus": 12, "initial_gen_voltage": 0.92},
}
EVAL_VERSION = "log2_depth_two_layer_v3"
CONFIGS = {
    "full_method": {"permission_shrink": True, "replan_mode": "local", "depth_mode": "adaptive"},
    "no_permission_shrink": {"permission_shrink": False, "replan_mode": "local", "depth_mode": "adaptive"},
    "full_replan_fixed_depth": {"permission_shrink": True, "replan_mode": "full", "depth_mode": "fixed"},
    "two_layer_orch_exec": {"permission_shrink": False, "replan_mode": "none", "depth_mode": "two_layer"},
}
FIELDS = ["eval_version", "scenario", "config", "repeat", "success", "illegal_tool_calls",
          "duplicate_tool_calls", "total_tokens", "elapsed_seconds", "final_voltage",
          "tree_depth", "d0", "root_success", "failed_tasks", "replan_count", "error"]


def make_network(scenario: dict) -> PowerNetwork:
    network = PowerNetwork()
    network.set_gen_voltage(0, scenario["initial_gen_voltage"])
    initial_check = check_constraints(network.net)
    if initial_check["all_satisfied"]:
        raise ValueError("初始状态没有实际约束违规")
    return network


def run_once(scenario_name: str, config_name: str, repeat: int) -> dict:
    started = time.perf_counter()
    reset_tool_counters()
    scenario = SCENARIOS[scenario_name]
    config = CONFIGS[config_name]
    llm = None
    root = None
    network = None
    success = False
    final_voltage = ""
    error = ""
    root_success = ""
    failed_tasks = ""
    replan_count = 0
    infrastructure_error = False
    tree_depth = ""
    d0 = ""
    try:
        network = make_network(scenario)
        llm = RealLLMClient()
        target = scenario["target_bus"]
        instruction = f"Bus {target + 1} 电压过低，请分析并恢复至至少 1.0 p.u.，且不能造成其他节点或线路越限。"
        if config_name == "two_layer_orch_exec":
            tree_depth = 2
            action = find_voltage_action(network.net, target, 1.0)
            if action is None:
                raise ValueError("两层基线未找到满足目标电压和约束的候选动作")
            agent = OrchestrationAgent(
                agent_id="baseline_orchestrator",
                task=Task(id="baseline", description=f"执行调压：{instruction}", devices=[target],
                          device_type="bus"),
                permission=Permission.root_permission(), network=network, llm=llm,
                current_depth=0, max_depth=2, voltage_action=action,
                permission_shrink=False,
            )
            baseline_result = agent.execute()
            if baseline_result.get("retryable"):
                raise LLMServiceUnavailable("两层基线调用期间 LLM 服务暂时不可用")
            root_success = int(baseline_result["success"])
            failed_tasks = "" if baseline_result["success"] else "baseline"
            error = baseline_result.get("error") or "" if not baseline_result["success"] else ""
        else:
            plan = Planner(network, llm, depth_mode=config["depth_mode"]).plan(instruction)
            tree_depth = plan["tree_depth"]
            d0 = plan["d0_info"]["d0"]
            root = RootAgent(network, plan["dag"], llm, plan["tree_depth"], plan["d0_info"],
                             plan["certainty"], target_bus=target,
                             oracle=True,
                             permission_shrink=config["permission_shrink"],
                             replan_mode=config["replan_mode"])
            root_result = root.execute()
            if any(entry["result"].get("retryable") for entry in root_result["execution_log"]):
                raise LLMServiceUnavailable("任务执行期间 LLM 服务暂时不可用")
            root_success = int(root_result["success"])
            failed_tasks = ",".join(task_id for task_id, status in root_result["task_status"].items()
                                    if status != "completed")
            replan_count = len(root_result["replan_log"])
            if not root_result["success"]:
                failures = [f"{entry['task_id']}: {entry['result'].get('error')}"
                            for entry in root_result["execution_log"] if not entry["result"].get("success")]
                error = "; ".join(failures[-3:]) or "任务未完成或目标电压未达到"
        # 始终在执行代理修改过的同一个最终网络对象上重新计算潮流。
        pp.runpp(network.net, algorithm="nr", init="results")
        final_voltage = round(float(network.net.res_bus.at[target, "vm_pu"]), 6)
        success = bool(final_voltage >= 1.0 and check_constraints(network.net)["all_satisfied"])
    except LLMServiceUnavailable as exc:
        infrastructure_error = True
        error = str(exc)
        logging.warning("评测暂停: %s / %s / %s: %s", scenario_name, config_name, repeat, exc)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        logging.exception("评测失败: %s / %s / %s", scenario_name, config_name, repeat)
    return {
        "eval_version": EVAL_VERSION, "scenario": scenario_name, "config": config_name, "repeat": repeat,
        "success": int(success),
        "illegal_tool_calls": get_tool_counters()["illegal_tool_calls"],
        "duplicate_tool_calls": root.duplicate_tool_calls if root else 0,
        "total_tokens": llm.total_tokens if llm else 0,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "final_voltage": final_voltage, "error": error,
        "root_success": root_success, "failed_tasks": failed_tasks, "replan_count": replan_count,
        "tree_depth": tree_depth, "d0": d0,
        "infrastructure_error": infrastructure_error,
    }


def print_summary(rows: list[dict]) -> None:
    print("\n配置 | 物理成功次数 | 非法调用数 | 重复执行调用数 | token | 耗时(s)")
    print("--- | ---: | ---: | ---: | ---: | ---:")
    for name in CONFIGS:
        group = [row for row in rows if row["config"] == name]
        print(f"{name} | {sum(row['success'] for row in group)}/{len(group)} | "
              f"{sum(row['illegal_tool_calls'] for row in group)} | "
              f"{sum(row['duplicate_tool_calls'] for row in group)} | "
              f"{sum(row['total_tokens'] for row in group)} | "
              f"{sum(row['elapsed_seconds'] for row in group):.2f}")
    print("注：物理成功由最终潮流和约束判定；代理流程是否完成请看 CSV 的 root_success。")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5, help="每个场景和配置重复次数")
    parser.add_argument("--output", type=Path, default=Path("eval_results.csv"))
    parser.add_argument("--resume", action="store_true", help="从已有 CSV 的下一次评测继续")
    parser.add_argument("--overwrite", action="store_true", help="明确覆盖已有 CSV")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats 必须大于 0")
    if not os.environ.get("LLM_API_KEY"):
        parser.error("未设置 LLM_API_KEY；评测需要真实模型，不能以模拟结果代替")
    try:
        probe = RealLLMClient()
        probe.check_connection()  # 在创建/覆盖结果文件之前检查认证。
    except (ValueError, LLMServiceUnavailable) as exc:
        parser.error(str(exc))
    for scenario in SCENARIOS.values():
        make_network(scenario)
    if args.resume and args.overwrite:
        parser.error("--resume 和 --overwrite 不能同时使用")
    if args.output.exists() and not (args.resume or args.overwrite):
        parser.error("输出文件已存在；请使用 --resume、--overwrite 或指定新文件名")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    file_has_header = False
    if args.resume and args.output.exists():
        with args.output.open(newline="", encoding="utf-8") as previous:
            reader = csv.DictReader(previous)
            if reader.fieldnames != FIELDS:
                parser.error("已有 CSV 列名与当前版本不一致；请指定新文件")
            file_has_header = True
            for row in reader:
                if row.get("eval_version") != EVAL_VERSION:
                    parser.error("已有 CSV 使用不同的评测逻辑；请指定新的输出文件")
                if any(marker in row.get("error", "").lower()
                       for marker in ("http 429", "http 500", "http 502", "http 503", "http 504", "timed out", "超时")):
                    parser.error("已有 CSV 含服务异常误记的失败行；请使用新的输出文件重新评测")
                for key in ("repeat", "success", "illegal_tool_calls", "duplicate_tool_calls", "total_tokens"):
                    row[key] = int(row[key])
                row["elapsed_seconds"] = float(row["elapsed_seconds"])
                rows.append(row)
    completed = {(row["scenario"], row["config"], row["repeat"]) for row in rows}
    with args.output.open("a" if args.resume and args.output.exists() else "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if not file_has_header:
            writer.writeheader()
        handle.flush()
        try:
            for scenario_name in SCENARIOS:
                for config_name in CONFIGS:
                    for repeat in range(1, args.repeats + 1):
                        if (scenario_name, config_name, repeat) in completed:
                            continue
                        row = run_once(scenario_name, config_name, repeat)
                        if row.pop("infrastructure_error"):
                            print(f"LLM 服务暂时不可用；停在 {scenario_name} / {config_name} / {repeat}，"
                                  f"本次已耗 token={row['total_tokens']}、{row['elapsed_seconds']:.2f}s；"
                                  "已完成的结果保留。稍后用 --resume 继续。")
                            raise SystemExit(2)
                        writer.writerow(row)
                        handle.flush()
                        rows.append(row)
                        print(f"{scenario_name} / {config_name} / {repeat}: "
                              f"{'成功' if row['success'] else '失败'}, "
                              f"tokens={row['total_tokens']}, {row['elapsed_seconds']:.2f}s")
        except KeyboardInterrupt:
            print("\n已中断；已完成的结果保留，可用 --resume 继续。")
            raise SystemExit(130)
    print_summary(rows)
    print(f"明细已写入 {args.output.resolve()}")


if __name__ == "__main__":
    main()
