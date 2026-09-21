"""case39 evaluation I/O only; architecture and execution live in agents/ and grid/."""
import argparse
import copy
import csv
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from urllib.parse import urlsplit, urlunsplit

from agents.regional import METHOD_CONFIGS, build_method
from config.settings import (CASE39_TEMPERATURE, CASE39_MAX_TOKENS,
                             CASE39_MAX_ATTEMPTS, CASE39_PROMPT_VERSION)
from grid.dispatch_kernel import DispatchKernel
from grid.goal import Goal
from grid.scenario_case39 import INSTRUCTION, make_network
from grid.tools import reset_tool_counters
from llm.client import RealLLMClient, LLMServiceUnavailable

METHODS = ("hierarchical", "two_layer", "two_layer_full_restart")
ABLATIONS = ("hierarchical_no_shrink", "hierarchical_full_restart")
FIELDS = ("method", "repeat", "success", "real_actions_used", "wasted_actions",
          "duplicate_tool_calls", "illegal_tool_calls", "catalog_tokens", "total_tokens",
          "replanned_tasks", "zone3_touched", "scope_hit", "tree_depth", "d0",
          "proposed_actions", "actual_actions", "error", "elapsed_seconds",
          "llm_calls", "tokens_by_source", "dag_task_count", "attempts", "hidden_limit_triggered",
          "first_attempt_deviation", "clipped_actions", "out_of_scope_proposals", "method_config",
          "git_commit", "model", "base_url", "temperature", "max_tokens", "max_actions", "formal",
          "prompt_version", "protocol_hash", "partitions")


def dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def protocol_hash():
    """Include uncommitted source changes, not just HEAD, when resuming a batch."""
    root = Path(__file__).resolve().parent
    paths = [*root.glob("agents/*.py"), *root.glob("grid/*.py"), *root.glob("llm/*.py"),
             root / "config/settings.py", root / "run_case39.py"]
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def metadata(client, max_actions, formal):
    url = urlsplit(getattr(client, "base_url", ""))
    safe_url = urlunsplit((url.scheme, url.netloc.rsplit("@", 1)[-1], url.path, "", ""))
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
                            capture_output=True, text=True)
    return {"git_commit": commit.stdout.strip() if commit.returncode == 0 else "unknown",
            "model": getattr(client, "model", "scripted-offline"), "base_url": safe_url,
            "temperature": CASE39_TEMPERATURE, "max_tokens": CASE39_MAX_TOKENS,
            "max_actions": max_actions, "formal": int(formal), "prompt_version": CASE39_PROMPT_VERSION,
            "protocol_hash": protocol_hash()}


class Case39Trial:
    """Thin composition adapter, retained for offline callers; contains no method logic."""
    def __init__(self, method, llm, max_actions=6):
        self.config = METHOD_CONFIGS[method]
        instruction = INSTRUCTION.replace("六次", f"{max_actions}次")
        goal = replace(Goal.from_instruction(instruction), max_real_actions=max_actions)
        self.kernel = DispatchKernel(make_network(), goal, instruction, llm)
        self.agent = None
        self.plan = {"dag": None, "tree_depth": 0, "d0_info": {"d0": 0}}

    def __getattr__(self, name):
        return getattr(self.kernel, name)

    def run(self):
        try:
            self.kernel.attempts = 1
            self.agent, self.plan = build_method(self.config, self.kernel)
            result = self.agent.execute()
            return bool(result.get("success") and self.kernel.success()), result.get("error", "")
        except ValueError as exc:
            self.kernel.failure(str(exc))
            return False, str(exc)


def run_once(method, repeat, llm=None, max_actions=6, formal=False):
    started = time.perf_counter()
    reset_tool_counters()
    client = llm or RealLLMClient(fixed_limits=True)
    token_start = getattr(client, "total_tokens", 0)
    calls_start = getattr(client, "llm_calls", 0)
    source_start = copy.deepcopy(getattr(client, "tokens_by_source", {}))
    trial = Case39Trial(method, client, max_actions)
    success, error = trial.run()
    kernel = trial.kernel
    source_delta = {source: {key: value - source_start.get(source, {}).get(key, 0)
                              for key, value in counts.items()}
                    for source, counts in getattr(client, "tokens_by_source", {}).items()}
    return {"method": method, "repeat": repeat, "success": int(success),
            "real_actions_used": len(kernel.network.action_log), "wasted_actions": len(kernel.wasted_indices),
            "duplicate_tool_calls": kernel.duplicates, "illegal_tool_calls": kernel.illegal,
            "catalog_tokens": kernel.catalog_tokens, "total_tokens": getattr(client, "total_tokens", 0) - token_start,
            "replanned_tasks": dumps(kernel.replanned), "zone3_touched": int(kernel.zone3_touched()),
            "scope_hit": kernel.scope_hit, "tree_depth": trial.plan["tree_depth"],
            "d0": round(trial.plan["d0_info"]["d0"], 4),
            "proposed_actions": dumps(kernel.proposed), "actual_actions": dumps(kernel.network.action_log),
            "error": error, "elapsed_seconds": round(time.perf_counter() - started, 3),
            "llm_calls": getattr(client, "llm_calls", calls_start + kernel.api_calls + int(trial.config.planner)) - calls_start,
            "tokens_by_source": dumps(source_delta),
            "dag_task_count": len(trial.plan["dag"].tasks) if trial.plan["dag"] else 0,
            "attempts": kernel.attempts, "hidden_limit_triggered": int(bool(kernel.clipped_actions)),
            "first_attempt_deviation": int(kernel.first_attempt_deviation), "clipped_actions": dumps(kernel.clipped_actions),
            "out_of_scope_proposals": kernel.scope_hit,
            "method_config": dumps({**trial.config.to_dict(), "max_attempts": CASE39_MAX_ATTEMPTS}),
            "partitions": dumps(getattr(trial.agent, "partitions", [])),
            **metadata(client, max_actions, formal)}


def method_order(repeat, methods=METHODS):
    offset = (repeat - 1) % len(methods)
    return methods[offset:] + methods[:offset]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=None, help="default: 10 (smoke: 1)")
    parser.add_argument("--output", default="case39_results.csv")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--max-actions", type=int, default=6)
    args = parser.parse_args(argv)
    repeats = args.repeats if args.repeats is not None else (1 if args.smoke else 10)
    if args.max_actions < 1 or (args.smoke and repeats not in (1, 2)) or (not args.smoke and repeats < 10):
        parser.error("max-actions 必须为正数；smoke 为1–2轮；正式实验至少10轮")
    path = Path(args.output)
    if path.exists() and not args.resume:
        parser.error("输出已存在，拒绝覆盖；使用 --resume 续跑")
    methods = METHODS + ABLATIONS if args.ablation else METHODS
    client = RealLLMClient(fixed_limits=True)
    expected = metadata(client, args.max_actions, not args.smoke)
    rows = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != list(FIELDS):
                parser.error("CSV schema 不匹配，不能续跑旧协议")
            rows = list(reader)
        seen = set()
        for row in rows:
            key = (row["method"], int(row["repeat"]))
            if key in seen or key[0] not in methods or not 1 <= key[1] <= repeats:
                parser.error("CSV 包含重复记录或与本次协议不兼容的 method/repeat")
            seen.add(key)
            if any(row[k] != str(v) for k, v in expected.items()):
                parser.error("模型、预算、formal 标志、代码或提示词版本不匹配；请使用新文件")
    completed = {(r["method"], int(r["repeat"])) for r in rows}
    try:
        client.check_connection()
        with path.open("a" if path.exists() else "x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            if not path.stat().st_size:
                writer.writeheader()
                stream.flush()
            for repeat in range(1, repeats + 1):
                # Rotate the three primary methods independently of optional ablations.
                order = method_order(repeat) + (method_order(repeat, ABLATIONS) if args.ablation else ())
                for method in order:
                    if (method, repeat) in completed:
                        continue
                    row = run_once(method, repeat, client, args.max_actions, not args.smoke)
                    writer.writerow(row)
                    stream.flush()
                    os.fsync(stream.fileno())
                    print(f"{method} #{repeat}: success={row['success']}, actions={row['real_actions_used']}", flush=True)
    except LLMServiceUnavailable as exc:
        print(f"基础设施错误，暂停；已完成结果已保留。修复服务后使用 --resume。{exc}", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
