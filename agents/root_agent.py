"""
S2 根智能体。

负责任务 DAG 调度和 S4 重规划循环。
"""

import logging
from agents.task import TaskDAG, TaskStatus
from agents.permission import Permission
from agents.orchestration_agent import OrchestrationAgent
from agents.replanner import Replanner
from agents.deviation import Deviation, DeviationType
from llm.client import LLMClient

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
    ):
        self.network = network
        self.dag = dag
        self.llm = llm
        self.tree_depth = tree_depth
        self.d0_info = d0_info
        self.certainty = certainty
        self.permission = Permission.root_permission()

        self.execution_log = []
        self.replan_log = []

        self.replanner = Replanner(
            network=self.network,
            llm=self.llm,
            d0_info=self.d0_info,
            certainty=self.certainty,
            tree_depth=self.tree_depth,
        )

    def execute(self) -> dict:
        logger.info("=" * 50)
        logger.info("[根智能体] 开始执行任务 DAG")
        logger.info(f"  树深度 H = {self.tree_depth}")
        logger.info(f"  任务数量 K = {len(self.dag.tasks)}")
        logger.info(f"  确定性 C = {self.certainty:.4f}")
        logger.info("=" * 50)

        iteration = 0
        max_iterations = len(self.dag.tasks) * 3  # 留余量给重规划

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

            for task in ready_tasks:
                self.dag.update_status(task.id, TaskStatus.ACTIVE)
                result = self._dispatch_task(task)

                if result.get("success", False):
                    self.dag.update_status(task.id, TaskStatus.COMPLETED, result)
                    logger.info(f"[根智能体] ✓ 任务 {task.id} 完成")
                else:
                    # ---- Step 3: S4 重规划 ----
                    logger.warning(f"[根智能体] ✗ 任务 {task.id} 失败，启动 S4")
                    replan_result = self._handle_task_failure(task, result)

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

        success = self.dag.all_completed()
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
            "task_results": self.dag.get_task_results(),
        }

    def _dispatch_task(self, task) -> dict:
        task_permission = Permission.from_task(task)
        child_permission = self.permission.intersect(task_permission)
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
        )

        return orch_agent.execute()

    def _handle_task_failure(self, task, result: dict) -> dict:
        """
        S4 入口：处理失败任务。

        对应原文档：
        "如果是失败，出现了问题，S4 来做反馈、自学习。
         假设 t1 失败了，t2 依赖于 t1，那 t2 先不跑，
         等 t1 重规划成功了再跑 t2。"
        """
        # 从结果中提取偏差信息
        deviation = result.get("deviation")

        if deviation is None:
            # 没有结构化偏差信息，构造一个通用的
            deviation = Deviation(
                deviation_type=DeviationType.INSUFFICIENT,
                description=result.get("error", "任务执行失败"),
                expected="任务成功完成",
                actual=str(result.get("error", "未知")),
                task_id=task.id,
            )

        # 收集前置任务结果
        prior_results = {}
        for dep_id in task.dependencies:
            dep_task = self.dag.tasks.get(dep_id)
            if dep_task and dep_task.result:
                prior_results[dep_id] = dep_task.result

        # 调用重规划器
        replan_result = self.replanner.handle_failure(
            task=task,
            deviation=deviation,
            parent_permission=self.permission,
            prior_results=prior_results,
        )

        self.replan_log.append({
            "task_id": task.id,
            "deviation": deviation.summary(),
            "replan_success": replan_result.get("success", False),
        })

        return replan_result