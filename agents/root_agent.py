"""
S2 根智能体。

负责任务 DAG 调度和 S4 重规划循环。
"""

import logging
import copy
import json
from agents.task import TaskDAG, TaskStatus
from agents.permission import Permission
from agents.orchestration_agent import OrchestrationAgent
from agents.replanner import Replanner
from agents.deviation import Deviation, DeviationType
from llm.client import LLMClient
from config.settings import MAX_REPLAN_ATTEMPTS

logger = logging.getLogger(__name__)


class RootAgent:
    """根智能体：任务 DAG 调度器 + S4 重规划入口。"""

    def __init__(
        self,
        network,
        dag: TaskDAG,
        llm: LLMClient,
        tree_depth: int,
        d0_info: dict,
        certainty: float,
        target_bus: int | None = None,
        min_target_voltage: float = 1.0,
        permission_shrink: bool = True,
        replan_mode: str = "local",
        mission: str = "",
        goal=None,
        oracle: bool = False,
        kernel=None,
    ):
        self.kernel = kernel
        self.network = network
        self.dag = dag
        self.llm = llm
        self.tree_depth = tree_depth
        self.d0_info = d0_info
        self.certainty = certainty
        self.target_bus = target_bus
        self.oracle = oracle
        self.min_target_voltage = min_target_voltage
        self.mission = mission
        self.goal = goal
        self.shared = {}
        if replan_mode not in ("local", "full"):
            raise ValueError("replan_mode 必须是 local 或 full")
        self.permission_shrink = permission_shrink
        self.replan_mode = replan_mode
        self.duplicate_tool_calls = 0
        self._seen_tool_calls = set()
        self.voltage_action = None
        if oracle and target_bus is not None:
            from grid.voltage_control import find_voltage_action
            self.voltage_action = find_voltage_action(network.net, target_bus, min_target_voltage)
        self.permission = Permission.root_permission()

        self.execution_log = []
        self.replan_log = []

        self.replanner = Replanner(
            network=self.network,
            llm=self.llm,
            d0_info=self.d0_info,
            certainty=self.certainty,
            tree_depth=self.tree_depth,
            permission_shrink=self.permission_shrink,
            mission=self.mission,
            goal=self.goal,
            shared=self.shared,
        )

    def execute(self) -> dict:
        if self.kernel is not None:
            return self._execute_joint()
        if self.oracle and self.target_bus is not None and self.voltage_action is None:
            voltage = self.network.get_bus_voltage(self.target_bus)["vm_pu"]
            if voltage < self.min_target_voltage:
                logger.error("[根智能体] 未找到同时满足目标电压与全网约束的调压动作")
                return {"success": False, "error": "未找到可行调压动作", "task_status": {},
                        "execution_log": [], "replan_log": [], "task_results": {}}
        logger.info("=" * 50)
        logger.info("[根智能体] 开始执行任务 DAG")
        logger.info(f"  树深度 H = {self.tree_depth}")
        logger.info(f"  任务数量 K = {len(self.dag.tasks)}")
        logger.info(f"  确定性 C = {self.certainty:.4f}")
        logger.info("=" * 50)

        iteration = 0
        max_iterations = len(self.dag.tasks) * (MAX_REPLAN_ATTEMPTS + 2)
        initial_net = copy.deepcopy(self.network.net)
        full_restarts = 0

        while not self.dag.all_completed() and iteration < max_iterations:
            iteration += 1

            ready_tasks = self.dag.get_ready_tasks()

            if not ready_tasks:
                if self.dag.has_failed():
                    logger.error("[根智能体] 存在失败任务且无新任务可执行")
                    break
                logger.warning("[根智能体] 无可执行任务")
                break

            logger.info(
                f"\n[根智能体] 第 {iteration} 轮: "
                f"激活 {len(ready_tasks)} 个任务 "
                f"({', '.join(t.id for t in ready_tasks)})"
            )

            restart_requested = False
            for task in ready_tasks:
                self.dag.update_status(task.id, TaskStatus.ACTIVE)
                result = self._dispatch_task(task)
                self._count_tool_calls(result, replanned=self.replan_mode == "full" and full_restarts > 0)

                if result.get("success", False):
                    self.dag.update_status(task.id, TaskStatus.COMPLETED, result)
                    logger.info(f"[根智能体] ✓ 任务 {task.id} 完成")
                else:
                    if self.replan_mode == "full":
                        self.execution_log.append({"task_id": task.id, "iteration": iteration, "result": result})
                        if full_restarts >= MAX_REPLAN_ATTEMPTS or result.get("llm_error"):
                            self.dag.update_status(task.id, TaskStatus.FAILED, result)
                        else:
                            full_restarts += 1
                            self.replan_log.append({"task_id": task.id, "mode": "full", "attempt": full_restarts})
                            self.network.net = copy.deepcopy(initial_net)
                            self.network.mutation_history.clear()
                            self.shared.clear()
                            for old_task in self.dag.tasks.values():
                                old_task.status = TaskStatus.PENDING
                                old_task.result = None
                                old_task.device_instructions = []
                            restart_requested = True
                        break
                    replan_result = self._handle_task_failure(task, result)
                    result["replan_result"] = replan_result

                    if replan_result.get("llm_error"):
                        logger.warning(f"[根智能体] 任务失败：LLM/API错误，不触发S4 ({task.id})")
                        self.dag.update_status(task.id, TaskStatus.FAILED, replan_result)
                    else:
                        # ---- Step 3: S4 重规划 ----
                        logger.warning(f"[根智能体] ✗ 任务 {task.id} 失败，启动 S4")
                        if replan_result.get("success", False):
                            self.dag.update_status(
                                task.id, TaskStatus.COMPLETED, replan_result
                            )
                            logger.info(
                                f"[根智能体] ✓ 任务 {task.id} 重规划后成功"
                            )
                        elif replan_result.get("needs_human", False):
                            self.dag.update_status(task.id, TaskStatus.FAILED, replan_result)
                            logger.error(
                                f"[根智能体] ⚠ 任务 {task.id} 需要人工介入"
                            )
                        else:
                            self.dag.update_status(task.id, TaskStatus.FAILED, replan_result)
                            logger.error(
                                f"[根智能体] ✗ 任务 {task.id} 重规划失败"
                            )

                self.execution_log.append({
                    "task_id": task.id,
                    "iteration": iteration,
                    "result": result,
                })
            if restart_requested:
                continue

        success = self.dag.all_completed()
        if self.target_bus is not None:
            from grid.tools import check_constraints
            target_voltage = self.network.get_bus_voltage(self.target_bus)["vm_pu"]
            success = bool(success and target_voltage >= self.min_target_voltage
                           and check_constraints(self.network.net)["all_satisfied"])
        if self.goal is not None:
            from grid.goal import goal_status
            success = bool(success and goal_status(self.network.net, self.goal)["goal_met"]
                           and len(self.network.action_log) <= self.goal.max_real_actions)
        status_summary = {
            tid: t.status.value for tid, t in self.dag.tasks.items()
        }

        logger.info("\n" + "=" * 50)
        logger.info(f"[根智能体] 执行{'完成' if success else '未完成'}")
        for tid, status in status_summary.items():
            marker = "✓" if status == "completed" else "✗"
            logger.info(f"  {marker} {tid}: {status}")
        if self.replan_log:
            logger.info(f"  重规划次数: {len(self.replan_log)}")
        logger.info("=" * 50)

        return {
            "success": success,
            "task_status": status_summary,
            "execution_log": self.execution_log,
            "replan_log": self.replan_log,
            "duplicate_tool_calls": self.duplicate_tool_calls,
            "task_results": self.dag.get_task_results(),
        }

    def _dispatch_task(self, task) -> dict:
        task_permission = Permission.from_task(task, self.network.net)
        child_permission = self.permission.intersect(task_permission) if self.permission_shrink else Permission.root_permission()
        agent_id = f"orch_1_{task.id}"

        prior_results = {}
        for dep_id in task.dependencies:
            dep_task = self.dag.tasks.get(dep_id)
            if dep_task and dep_task.result:
                prior_results[dep_id] = dep_task.result

        orch_agent = OrchestrationAgent(
            agent_id=agent_id,
            task=task,
            permission=child_permission,
            network=self.network,
            llm=self.llm,
            current_depth=1,
            max_depth=self.tree_depth,
            prior_results=prior_results,
            d0_info=self.d0_info,
            certainty=self.certainty,
            voltage_action=self.voltage_action,
            permission_shrink=self.permission_shrink,
            mission=self.mission,
            goal=self.goal,
            shared=self.shared,
        )

        return orch_agent.execute()

    def _count_tool_calls(self, result: dict, replanned: bool = False) -> None:
        """统计重规划后再次执行的相同工具与参数组合。"""
        def walk(node):
            if isinstance(node, dict):
                instruction = node.get("instruction")
                if isinstance(instruction, dict) and "tool_results" in node:
                    signature = json.dumps({"tool": instruction.get("tool"),
                                            "params": instruction.get("params")},
                                           sort_keys=True, default=str)
                    if replanned and signature in self._seen_tool_calls:
                        self.duplicate_tool_calls += 1
                    self._seen_tool_calls.add(signature)
                for key in ("execution_results", "child_results"):
                    for child in node.get(key, []):
                        walk(child)
        walk(result)

    def _handle_task_failure(self, task, result: dict) -> dict:
        """
        S4 入口：处理失败任务。

        对应原文档：
        "如果是失败，出现了问题，S4 来做反馈、自学习。
         假设 t1 失败了，t2 依赖于 t1，那 t2 先不跑，
         等 t1 重规划成功了再跑 t2。"
        """
        if result.get("llm_error"):
            return {"success": False, "task_id": task.id, "error": result.get("error") or "LLM/API 请求失败",
                    "llm_error": True, "retryable": bool(result.get("retryable"))}
        # 从结果中提取偏差信息
        deviation = result.get("deviation")

        if deviation is None:
            deviation = Deviation(
                deviation_type=DeviationType.INSUFFICIENT,
                description=result.get("error", "任务执行失败"),
                expected="任务成功完成",
                actual=str(result.get("error", "未知")),
                task_id=task.id,
            )

        failure_text = f"{deviation.description} {deviation.actual}".lower()
        llm_markers = [
            "llm_error",
            "llm",
            "api",
            "json",
            "解析失败",
            "网络错误",
            "连接失败",
            "请求失败",
            "无法解析",
        ]
        if any(marker in failure_text for marker in llm_markers):
            logger.warning(f"[根智能体] 识别到 LLM/API 失败，不触发 S4 重规划: {deviation.summary()}")
            return {
                "success": False,
                "task_id": task.id,
                "error": result.get("error", "LLM/API 请求失败"),
                "needs_human": False,
                "llm_error": True,
                "deviation": deviation.summary(),
            }

        prior_results = {}
        for dep_id in task.dependencies:
            dep_task = self.dag.tasks.get(dep_id)
            if dep_task and dep_task.result:
                prior_results[dep_id] = dep_task.result

        replan_result = self.replanner.handle_failure(
            task=task,
            deviation=deviation,
            parent_permission=self.permission,
            prior_results=prior_results,
        )
        self._count_tool_calls(replan_result, replanned=True)

        self.replan_log.append({
            "task_id": task.id,
            "deviation": deviation.summary(),
            "replan_success": replan_result.get("success", False),
        })

        return replan_result

    def _joint_agent(self, task, permission=None):
        if permission is None:
            permission = (self.permission.intersect(Permission.from_task(task, self.network.net, self.goal.forbidden_regions))
                          if self.permission_shrink else Permission.root_permission())
        branch = next(tuple(ids) for ids in self.partitions if task.id in ids)
        if branch not in self.branch_agents:
            self.branch_agents[branch] = OrchestrationAgent(
                "regional/" + "/".join(branch), task, permission, self.network, self.llm, 1, self.tree_depth,
                permission_shrink=self.permission_shrink, mission=self.mission, goal=self.goal,
                kernel=self.kernel)
        agent = self.branch_agents[branch]
        agent.task, agent.permission = task, permission
        return agent

    def _execute_joint(self):
        """Stage the DAG, gate its joint plan, then commit only dependency-ready tasks."""
        from agents.regional import partition_dag
        from config.settings import CASE39_MAX_ATTEMPTS
        self.partitions = partition_dag(self.dag, self.network.net)
        order = [tid for group in self.partitions for tid in group]
        self.joint_plans, self.joint_agents, self.branch_agents = {}, {}, {}
        self.joint_permissions = {}
        failed = None
        for attempt in range(1, CASE39_MAX_ATTEMPTS + 1):
            self.kernel.attempts = attempt
            if attempt > 1:
                if self.replan_mode == "full":
                    self.kernel.rollback()
                    self.kernel.replanned.append("all")
                    self.joint_plans.clear()
                    self.joint_agents.clear()
                    self.joint_permissions.clear()
                    self.branch_agents.clear()
                    for task in self.dag.tasks.values():
                        task.status, task.result = TaskStatus.PENDING, None
                elif failed is not None:
                    self.replanner.rebuild_joint(self, failed)
                else:
                    # Joint simulation failure has no unique physical culprit.
                    for tid in order:
                        if self.dag.tasks[tid].status != TaskStatus.COMPLETED:
                            self.replanner.rebuild_joint(self, tid)
            failed = None
            try:
                pending = [tid for tid in order if self.dag.tasks[tid].status != TaskStatus.COMPLETED]
                for tid in pending:
                    if tid not in self.joint_plans:
                        failed = tid
                        agent = self._joint_agent(self.dag.tasks[tid])
                        self.joint_agents[tid] = agent
                        self.joint_permissions[tid] = agent.permission
                        self.joint_plans[tid] = agent.propose_joint()
                failed = None
                actions = [action for tid in pending for action in self.joint_plans[tid]]
                gate = self.kernel.joint_gate(actions)
                if not gate.get("goal_met"):
                    self.kernel.failure(json.dumps(gate, ensure_ascii=False))
                    continue
                for tid in pending:
                    task = self.dag.tasks[tid]
                    if not all(self.dag.tasks[d].status == TaskStatus.COMPLETED for d in task.dependencies):
                        raise ValueError("Dependent task is waiting for failed predecessor")
                    task.status = TaskStatus.ACTIVE
                    result = self._joint_agent(task, self.joint_permissions[tid]).commit_joint(self.joint_plans[tid])
                    if not result.get("success"):
                        failed = tid
                        task.status = TaskStatus.FAILED
                        self.kernel.failure(result.get("error", "执行失败"))
                        break
                    task.status, task.result = TaskStatus.COMPLETED, result
                if failed is None:
                    if self.kernel.success():
                        return {"success": True}
                    self.kernel.failure("真实执行后全网未达标")
                    # Do not silently create recovery tasks for an inadequate DAG.
                    return {"success": False, "error": self.kernel.feedback}
            except ValueError as exc:
                self.kernel.failure(str(exc))
        return {"success": False, "error": self.kernel.feedback}
