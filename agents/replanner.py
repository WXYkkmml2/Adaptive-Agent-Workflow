"""
S4 失效定位与子树重规划。

重规划是在失效路径上重新实例化编排智能体，
重新拆解相关任务并修正调用参数。
"""

import logging
from agents.task import Task
from agents.deviation import Deviation, DeviationType
from agents.permission import Permission
from agents.orchestration_agent import OrchestrationAgent
from llm.client import LLMClient
from config.settings import MAX_REPLAN_ATTEMPTS

logger = logging.getLogger(__name__)


class Replanner:
    """S4 重规划器。"""

    def __init__(
        self,
        network,
        llm: LLMClient,
        d0_info: dict,
        certainty: float,
        tree_depth: int,
        permission_shrink: bool = True,
        mission: str = "",
        goal=None,
        shared: dict = None,
    ):
        self.network = network
        self.llm = llm
        self.d0_info = d0_info
        self.certainty = certainty
        self.tree_depth = tree_depth
        self.permission_shrink = permission_shrink
        self.mission = mission
        self.goal = goal
        self.shared = shared if shared is not None else {}

    def handle_failure(
        self,
        task: Task,
        deviation: Deviation,
        parent_permission: Permission,
        prior_results: dict = None,
        attempt: int = 1,
    ) -> dict:
        """
        处理一个失败任务的完整 S4 流程。

        只在真实物理异常或业务偏差下触发重规划；
        API 超时、LLM 请求失败、JSON 格式错误不触发 S4。
        """
        if attempt > MAX_REPLAN_ATTEMPTS:
            logger.error(
                f"[重规划] 任务 {task.id} 已重试 {MAX_REPLAN_ATTEMPTS} 次，"
                f"需要人工介入"
            )
            return {
                "success": False,
                "task_id": task.id,
                "error": f"超过最大重试次数 ({MAX_REPLAN_ATTEMPTS})",
                "needs_human": True,
            }

        if self._is_llm_or_api_failure(deviation):
            logger.warning(f"[重规划] 识别到 LLM/API 错误，不触发重规划: {deviation.summary()}")
            return {
                "success": False,
                "task_id": task.id,
                "error": "LLM/API 错误，不触发 S4 重规划",
                "needs_human": False,
                "llm_error": True,
            }

        logger.info(
            f"\n[重规划] 任务 {task.id} 第 {attempt} 次重规划"
            f"\n  偏差类型: {deviation.deviation_type.value}"
            f"\n  偏差描述: {deviation.description}"
        )

        failure_context = self._build_failure_context(task, deviation, prior_results)
        self.shared.pop("verified_plan", None)
        self.shared.pop("verified_at", None)

        # ---- 4. 在失效节点重新实例化编排智能体 ----
        # 对应原文档：
        # "在 B 的位置实例化一个新的编排智能体 B*，
        #  重新走 S2 流程，B* 拆出来的结果可能和 B 不一样
        # （因为大模型这次拿到了 B 失败的上下文信息作为额外输入）"
        result = self._replan_subtree(
            task, parent_permission, failure_context, attempt
        )

        if result.get("success", False):
            logger.info(f"[重规划] ✓ 任务 {task.id} 重规划成功")
            return result

        # 重规划也失败了，递归重试
        new_deviation = result.get("deviation") or Deviation(
            deviation_type=DeviationType.INSUFFICIENT,
            description=f"第 {attempt} 次重规划后仍然失败",
            expected="任务成功完成",
            actual=str(result.get("error", "未知")),
            task_id=task.id,
        )
        return self.handle_failure(
            task, new_deviation, parent_permission,
            prior_results, attempt + 1,
        )

    def _replan_subtree(
        self,
        task: Task,
        parent_permission: Permission,
        failure_context: dict,
        attempt: int,
    ) -> dict:
        """
        重建失效路径的子树。

        对应原文档：
        "不管哪种情况，树的其他路径完全不受影响，继续运行。"
        """
        task_permission = Permission.from_task(task)
        child_permission = parent_permission.intersect(task_permission) if self.permission_shrink else Permission.root_permission()

        agent_id = f"orch_replan_{task.id}_attempt{attempt}"

        orch_agent = OrchestrationAgent(
            agent_id=agent_id,
            task=task,
            permission=child_permission,
            network=self.network,
            llm=self.llm,
            current_depth=1,
            max_depth=self.tree_depth,
            prior_results=failure_context,
            d0_info=self.d0_info,
            certainty=self.certainty,
            is_replan=True,
            failure_info=failure_context,
            permission_shrink=self.permission_shrink,
            mission=self.mission,
            goal=self.goal,
            shared=self.shared,
        )

        return orch_agent.execute()

    @staticmethod
    def _is_llm_or_api_failure(deviation: Deviation) -> bool:
        text = f"{deviation.description} {deviation.actual}".lower()
        llm_markers = [
            "llm_error",
            "llm",
            "api",
            "json",
            "解析失败",
            "网络错误",
            "连接失败",
            "无法解析",
            "请求失败",
        ]
        if any(marker in text for marker in llm_markers):
            return True

        # 仅在与 LLM/API 直接相关的超时场景下忽略 S4；通用工具超时仍允许重规划。
        return "timeout" in text and any(marker in text for marker in ["llm", "api", "openai", "http", "request"])

    def _build_failure_context(
        self, task: Task, deviation: Deviation, prior_results: dict = None
    ) -> dict:
        """
        构建失败上下文，传给重规划的 LLM。

        这些信息让 LLM 在重新拆解时避开上次的错误。
        """
        safe_prior = {}
        for key, value in (prior_results or {}).items():
            if isinstance(value, dict):
                safe_prior[key] = {
                    "success": value.get("success"),
                    "tool": value.get("tool"),
                    "error": value.get("error"),
                    "result_summary": self._summarize_result(value.get("result")),
                }
            else:
                safe_prior[key] = str(value)

        return {
            "is_replan": True,
            "previous_failure": {
                "type": deviation.deviation_type.value,
                "description": deviation.description,
                "expected": deviation.expected,
                "actual": deviation.actual,
            },
            "prior_task_results": safe_prior,
            "replan_guidance": "请根据上次失败原因修正工具、参数或步骤；不要重复相同调用。",
        }

    @staticmethod
    def _summarize_result(result):
        if not isinstance(result, dict):
            return str(result)[:200] if result is not None else None
        summary = {}
        for key in ["bus_voltages", "line_loadings", "violations", "constraint_result", "all_satisfied", "violation_count"]:
            if key in result:
                summary[key] = result[key]
        if not summary and "vm_pu" in result:
            summary["vm_pu"] = result["vm_pu"]
        return summary

    def rebuild_joint(self, root, task_id):
        """Invalidate the failed path only; completed tasks are immutable locally."""
        from agents.task import TaskStatus
        invalid = {task_id}
        changed = True
        while changed:
            changed = False
            for tid, task in root.dag.tasks.items():
                if set(task.dependencies) & invalid and tid not in invalid:
                    invalid.add(tid)
                    changed = True
        for tid in invalid:
            task = root.dag.tasks[tid]
            if task.status != TaskStatus.COMPLETED:
                task.status = TaskStatus.PENDING
                root.joint_plans.pop(tid, None)
                root.joint_agents.pop(tid, None)
                root.joint_permissions.pop(tid, None)
        for branch in list(root.branch_agents):
            if set(branch) & invalid:
                root.branch_agents.pop(branch)
        root.kernel.replanned.append(task_id)
        return invalid
