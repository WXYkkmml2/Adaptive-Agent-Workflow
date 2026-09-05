"""
S4 失效定位与子树重规划。

"只是重规划失效的路径，其他的不动。
 重规划是在失效的路径上递归新的实例的子编排智能体，
 重新拆解相关的任务、调用顺序、修正调用的参数。"

"重规划时新实例化的编排智能体在调用大模型的时候，
 可以把前一次的失败信息（偏差特征、失效原因）作为上下文一起输入大模型，
 让大模型在重新拆解的时候避开上次的错误。"
"""

import logging
from agents.task import Task, TaskDAG, TaskStatus
from agents.deviation import Deviation, DeviationType
from agents.permission import Permission
from agents.orchestration_agent import OrchestrationAgent
from agents.anti_example import AntiExampleStore, FailureCase
from llm.client import LLMClient
from config.settings import MAX_REPLAN_ATTEMPTS

logger = logging.getLogger(__name__)


class Replanner:
    """
    S4 重规划器。

    职责：
    1. 根据偏差特征判断失效层级
    2. 在失效节点位置实例化新的智能体
    3. 将失败信息作为上下文传给新智能体
    4. 将失败案例存入反例库
    """

    def __init__(
        self,
        network,
        llm: LLMClient,
        anti_example_store: AntiExampleStore,
        d0_info: dict,
        certainty: float,
        tree_depth: int,
    ):
        self.network = network
        self.llm = llm
        self.anti_example_store = anti_example_store
        self.d0_info = d0_info
        self.certainty = certainty
        self.tree_depth = tree_depth

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

        对应原文档的三种情况：
        1. 失效在执行层 → 用同样指令重走 S3（反例库会降权）
        2. 失效在编排层 → 重新拆解（带失败上下文）
        3. 失效在更上层 → 从更上层重建子树

        MVP 简化为：所有情况都从编排层重新拆解，
        因为编排层会带着失败上下文重新调用 LLM。
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

        logger.info(
            f"\n[重规划] 任务 {task.id} 第 {attempt} 次重规划"
            f"\n  偏差类型: {deviation.deviation_type.value}"
            f"\n  偏差描述: {deviation.description}"
        )

        # ---- 1. 记录失败案例到反例库 ----
        self._record_failure(task, deviation)

        # ---- 2. 判断处理方式 ----
        # 对应原文档：
        # "权限越界 → 回到上层重新拆解
        #  违反物理约束 → 回到编排层重规划
        #  参数有效性 → 回到 S3 重新调 LLM
        #  工具接口故障 → 等待或切换备用"
        if deviation.deviation_type == DeviationType.TOOL_FAULT:
            logger.info("[重规划] 工具故障，等待后重试")
            # MVP 中直接重试，生产中应等待/切换
            pass  # 下面统一走重新编排

        # ---- 3. 构建失败上下文 ----
        failure_context = self._build_failure_context(task, deviation, prior_results)

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
        new_deviation = Deviation(
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
        child_permission = parent_permission.intersect(task_permission)

        agent_id = f"orch_replan_{task.id}_attempt{attempt}"

        orch_agent = OrchestrationAgent(
            agent_id=agent_id,
            task=task,
            permission=child_permission,
            network=self.network,
            llm=self.llm,
            current_depth=1,
            max_depth=self.tree_depth,
            prior_results=failure_context,  # 把失败信息作为上下文传入
            anti_example_store=self.anti_example_store,
            d0_info=self.d0_info,
            certainty=self.certainty,
            is_replan=True,                 # 标记为重规划
            failure_info=failure_context,   # 失败详情
        )

        return orch_agent.execute()

    def _record_failure(self, task: Task, deviation: Deviation):
        """
        将失败案例存入反例库。

        
        "每个失败案例记录包括：失败路径的任务描述、涉及设备的电气耦合强度 a、
         BFS 距离 b、确定性指标 C、使用的工具调用序列、失败分类标签、
         触发失败的上下文参数。"
        """
        case = FailureCase(
            case_id=f"fail_{task.id}_{self.anti_example_store.size()}",
            task_description=task.description,
            coupling_strength=self.d0_info.get("coupling_strength_a", 0),
            topology_depth=self.d0_info.get("topology_depth_b", 1),
            certainty=self.certainty,
            tool_sequence=[],  # 从 deviation context 中提取
            failure_type=deviation.deviation_type.value,
            context={
                "deviation": deviation.description,
                "expected": deviation.expected,
                "actual": deviation.actual,
            },
        )
        self.anti_example_store.add_case(case)
        logger.info(
            f"[反例库] 已记录失败案例 {case.case_id}，"
            f"库中共 {self.anti_example_store.size()} 条"
        )

    def _build_failure_context(
        self, task: Task, deviation: Deviation, prior_results: dict = None
    ) -> dict:
        """
        构建失败上下文，传给重规划的 LLM。

        这些信息让 LLM 在重新拆解时避开上次的错误。
        """
        return {
            "is_replan": True,
            "previous_failure": {
                "type": deviation.deviation_type.value,
                "description": deviation.description,
                "expected": deviation.expected,
                "actual": deviation.actual,
            },
            "prior_task_results": prior_results or {},
            "replan_guidance": self._get_guidance(deviation),
        }

    def _get_guidance(self, deviation: Deviation) -> str:
        """
        根据偏差类型生成重规划指导建议。
        """
        guidance_map = {
            DeviationType.PERMISSION: "请检查操作是否在权限范围内，改用有权限的设备或操作。",
            DeviationType.CONSTRAINT: "上次操作导致约束违规，请采用更保守的调整幅度或分步执行。",
            DeviationType.PARAMETER: "参数超出设备允许范围，请校验参数有效性。",
            DeviationType.TOOL_FAULT: "上次工具调用出错，请尝试替代工具或不同调用方式。",
            DeviationType.INSUFFICIENT: "上次调整幅度不足，请加大调整力度。",
        }
        return guidance_map.get(deviation.deviation_type, "请重新分析并生成方案。")