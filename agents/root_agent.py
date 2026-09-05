"""
S2 根智能体。

"根智能体本质上是一个任务调度器，按 G 依次或并行放行。
 根智能体维护一个任务状态表，检查 G 中哪些任务的前置依赖已经全部完成，
 对这些任务分别实例化编排层智能体。"

根智能体只和第一编排层交互，不直接跳到执行层。
"""

import logging
from agents.task import TaskDAG, TaskStatus
from agents.permission import Permission
from agents.orchestration_agent import OrchestrationAgent
from agents.anti_example import AntiExampleStore
from llm.client import LLMClient

logger = logging.getLogger(__name__)


class RootAgent:
    """
    根智能体：任务 DAG 调度器。

    持有 G（任务 DAG）和全局权限，按依赖关系分批激活任务。
    每批任务中互不依赖的可以并行（MVP 中串行模拟）。
    """

    def __init__(
        self,
        network,
        dag: TaskDAG,
        llm: LLMClient,
        tree_depth: int,
        d0_info: dict,
        certainty: float,
        anti_example_store: AntiExampleStore = None,
    ):
        self.network = network
        self.dag = dag
        self.llm = llm
        self.tree_depth = tree_depth  # H
        self.d0_info = d0_info
        self.certainty = certainty
        self.permission = Permission.root_permission()
        self.anti_example_store = anti_example_store or AntiExampleStore()

        # 执行记录（用于 S4 审计和回传）
        self.execution_log = []

    def execute(self) -> dict:
        """
        执行整个任务 DAG。

        流程：
        1. 检查 DAG 中哪些任务可以执行（前置依赖已完成）
        2. 为每个就绪任务实例化编排智能体
        3. 收集结果，更新任务状态
        4. 重复直到所有任务完成或出现不可恢复的失败
        
        对应原文档：
        "根智能体检查 G 中哪些任务的前置依赖已经全部完成，
         对这些任务分别实例化编排层智能体"
        """
        logger.info("=" * 50)
        logger.info("[根智能体] 开始执行任务 DAG")
        logger.info(f"  树深度 H = {self.tree_depth}")
        logger.info(f"  任务数量 K = {len(self.dag.tasks)}")
        logger.info(f"  确定性 C = {self.certainty:.4f}")
        logger.info("=" * 50)

        iteration = 0
        max_iterations = len(self.dag.tasks) * 2  # 防止死循环

        while not self.dag.all_completed() and iteration < max_iterations:
            iteration += 1

            # 获取当前可执行的任务
            ready_tasks = self.dag.get_ready_tasks()

            if not ready_tasks:
                if self.dag.has_failed():
                    logger.error("[根智能体] 存在失败任务且无新任务可执行，流程终止")
                    break
                logger.warning("[根智能体] 无可执行任务（可能存在循环依赖）")
                break

            logger.info(
                f"\n[根智能体] 第 {iteration} 轮: "
                f"激活 {len(ready_tasks)} 个任务 "
                f"({', '.join(t.id for t in ready_tasks)})"
            )

            # 对每个就绪任务实例化编排智能体
            # 对应原文档："同时激活两个编排智能体分别处理 t1 和 t5"
            # MVP 中串行执行，生产环境可改为并发
            for task in ready_tasks:
                self.dag.update_status(task.id, TaskStatus.ACTIVE)
                result = self._dispatch_task(task)

                if result.get("success", False):
                    self.dag.update_status(task.id, TaskStatus.COMPLETED, result)
                    logger.info(f"[根智能体] ✓ 任务 {task.id} 完成")
                else:
                    self.dag.update_status(task.id, TaskStatus.FAILED, result)
                    logger.error(
                        f"[根智能体] ✗ 任务 {task.id} 失败: "
                        f"{result.get('error', '未知原因')}"
                    )
                    # 失败处理：Step 3 实现 S4 重规划
                    # 当前先标记失败，不阻塞无依赖关系的其他任务

                self.execution_log.append({
                    "task_id": task.id,
                    "iteration": iteration,
                    "result": result,
                })

        # 汇总最终结果
        success = self.dag.all_completed()
        status_summary = {
            tid: t.status.value for tid, t in self.dag.tasks.items()
        }

        logger.info("\n" + "=" * 50)
        logger.info(f"[根智能体] 执行{'完成' if success else '未完成'}")
        for tid, status in status_summary.items():
            marker = "✓" if status == "completed" else "✗"
            logger.info(f"  {marker} {tid}: {status}")
        logger.info("=" * 50)

        return {
            "success": success,
            "task_status": status_summary,
            "execution_log": self.execution_log,
            "task_results": self.dag.get_task_results(),
        }

    def _dispatch_task(self, task) -> dict:
        """
        为一个任务实例化编排智能体。

        "根智能体只和第一编排层交互，不直接跳到执行层。"
        """
        # 从任务推导子权限，和根权限取交集
        task_permission = Permission.from_task(task)
        child_permission = self.permission.intersect(task_permission)

        agent_id = f"orch_1_{task.id}"

        # 收集前置任务的结果，作为编排智能体的上下文
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
            current_depth=1,           # 第一编排层
            max_depth=self.tree_depth,  # H
            prior_results=prior_results,
            anti_example_store=self.anti_example_store,
            d0_info=self.d0_info,
            certainty=self.certainty,
        )

        return orch_agent.execute()