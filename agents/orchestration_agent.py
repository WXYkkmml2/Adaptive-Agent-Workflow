"""
S2 编排智能体。

对应原文档：
"编排智能体接受任务，根据权限进行剪枝，
 然后编排会去二次调用大模型，在约束拓扑子图上进行细化任务。"

关键行为取决于是否为最后编排层：
- 是最后编排层 → 输出设备级执行语义（设备ID、操作类型、参数值）
- 不是 → 继续拆子任务，实例化下一层编排智能体
"""

import logging
from agents.task import Task
from agents.permission import Permission
from agents.execution_agent import ExecutionAgent
from llm.client import LLMClient
from llm.prompts import ORCHESTRATION_SYSTEM, ORCHESTRATION_USER
from grid.tools import get_available_tools

logger = logging.getLogger(__name__)


class OrchestrationAgent:
    """
    编排智能体。

    每个编排智能体负责一个任务的细化：
    加载权限 → 裁剪搜索空间 → 调用 LLM 细化 → 实例化下一层

    参数:
        agent_id: 智能体标识（如 "orch_1_t1" 表示第 1 层负责 t1 的编排）
        task: 分配给该智能体的任务
        permission: 该智能体的权限三元组（已和父智能体取交集）
        network: 电网对象
        llm: LLM 客户端
        current_depth: 当前所在层级（根=0, 第一编排层=1, ...）
        max_depth: 最大深度 H（执行层在 H-1）
        prior_results: 前置任务的结果（供上下文使用）
        anti_example_store: 反例库（传给执行智能体用）
        d0_info: D0 相关信息（传给执行智能体用）
        certainty: 确定性指标 C（传给执行智能体用）
    """

    def __init__(
        self,
        agent_id: str,
        task: Task,
        permission: Permission,
        network,
        llm: LLMClient,
        current_depth: int,
        max_depth: int,
        prior_results: dict = None,
        anti_example_store=None,
        d0_info: dict = None,
        certainty: float = 0.7,
    ):
        self.agent_id = agent_id
        self.task = task
        self.permission = permission
        self.network = network
        self.llm = llm
        self.current_depth = current_depth
        self.max_depth = max_depth
        self.prior_results = prior_results or {}
        self.anti_example_store = anti_example_store
        self.d0_info = d0_info or {}
        self.certainty = certainty

    @property
    def is_last_orchestration_layer(self) -> bool:
        """
        判断当前是否为最后编排层。
        
        如果是最后编排层，LLM 需要输出设备级指令；
        否则继续拆子任务给下一层编排。
        
        树结构（以 H=4 为例）：
          depth 0: 根
          depth 1: 编排01
          depth 2: 编排02 ← 最后编排层（下一层是执行层 depth 3 = H-1）
          depth 3: 执行
        
        所以最后编排层的 current_depth == max_depth - 2
        """
        return self.current_depth >= self.max_depth - 2

    def execute(self) -> dict:
        """
        执行编排流程：

        1. 根据权限裁剪可用工具（模拟拓扑裁剪）
        2. 调用 LLM 细化任务
        3a. 如果是最后编排层 → 实例化执行智能体
        3b. 如果不是 → 实例化下一层编排智能体
        4. 收集子智能体结果，回传给父
        """
        logger.info(
            f"{'  ' * self.current_depth}[{self.agent_id}] "
            f"编排层 depth={self.current_depth}, "
            f"任务: {self.task.description}"
        )

        # ---- 1. 权限裁剪 ----
        # 根据权限过滤可用工具，模拟"在约束拓扑子图上"操作
        available_tools = get_available_tools(self.permission.to_dict())
        logger.info(
            f"{'  ' * self.current_depth}  可用工具({len(available_tools)}): "
            f"{available_tools}"
        )

        # ---- 2. 调用 LLM 细化任务 ----
        instructions = self._call_llm_decompose(available_tools)

        if not instructions:
            return {
                "agent_id": self.agent_id,
                "task_id": self.task.id,
                "success": False,
                "error": "LLM 未返回有效指令",
            }

        # ---- 3. 分发到下一层 ----
        if self.is_last_orchestration_layer:
            # 最后编排层 → 实例化执行智能体
            return self._dispatch_to_execution(instructions)
        else:
            # 非最后编排层 → 实例化下一层编排智能体
            return self._dispatch_to_next_orchestration(instructions)

    def _call_llm_decompose(self, available_tools: list) -> list:
        """
        调用 LLM 将任务细化。

        对应原文档：
        "编排智能体把设备级任务和约束拓扑子图输入大模型"
        """
        system_prompt = ORCHESTRATION_SYSTEM.format(
            permission=self.permission.to_dict(),
            available_devices=self.task.devices,
        )

        user_prompt = ORCHESTRATION_USER.format(
            task_description=self.task.description,
            devices=self.task.devices,
            prior_results=str(self.prior_results),
        )

        # 调试输出：记录发送给 LLM 的 prompt（便于定位解析错误）
        logger.debug(f"System prompt:\n{system_prompt}")
        logger.debug(f"User prompt:\n{user_prompt}")

        response = self.llm.complete_json(system_prompt, user_prompt)
        return response.get("instructions", [])

    def _dispatch_to_execution(self, instructions: list) -> dict:
        """
        实例化执行智能体并收集结果。

        对应原文档：
        "编排智能体把当前任务拆成多个互不依赖的执行级子任务，
         就分别实例化多个并列执行智能体"
        """
        results = []

        for i, inst in enumerate(instructions):
            exec_id = f"exec_{self.task.id}_{i}"

            # 为执行智能体生成权限（和当前编排智能体取交集）
            exec_permission = self.permission  # 执行层继承编排层权限

            exec_agent = ExecutionAgent(
                agent_id=exec_id,
                instruction=inst,
                permission=exec_permission,
                network=self.network,
                llm=self.llm,
                anti_example_store=self.anti_example_store,
                d0_info=self.d0_info,
                certainty=self.certainty,
                depth=self.current_depth + 1,
            )

            result = exec_agent.execute()
            results.append(result)

        # 汇总结果回传
        all_success = all(r.get("success", False) for r in results)
        return {
            "agent_id": self.agent_id,
            "task_id": self.task.id,
            "success": all_success,
            "execution_results": results,
            # 编排智能体保留预期状态，用于 S4 偏差检测
            "expected_results": [inst.get("expected_result", "") for inst in instructions],
        }

    def _dispatch_to_next_orchestration(self, instructions: list) -> dict:
        """
        实例化下一层编排智能体。

        对应原文档（4 层示例）：
        "编排01 的智能体 A 把 t1 细化为 t1a 和 t1b，
         因为当前层不是最后编排层，向编排02 层分别实例化智能体 B 和 C"
        """
        results = []

        for i, inst in enumerate(instructions):
            # 把指令包装成子任务
            sub_task = Task(
                id=f"{self.task.id}_sub{i}",
                description=inst.get("description", ""),
                devices=self.task.devices,
                device_type=self.task.device_type,
                voltage_level=self.task.voltage_level,
            )

            child_id = f"orch_{self.current_depth + 1}_{sub_task.id}"

            # 子编排智能体的权限 = 当前权限 ∩ 子任务所需权限
            child_required = Permission.from_task(sub_task)
            child_permission = self.permission.intersect(child_required)

            child_agent = OrchestrationAgent(
                agent_id=child_id,
                task=sub_task,
                permission=child_permission,
                network=self.network,
                llm=self.llm,
                current_depth=self.current_depth + 1,
                max_depth=self.max_depth,
                prior_results=self.prior_results,
                anti_example_store=self.anti_example_store,
                d0_info=self.d0_info,
                certainty=self.certainty,
            )

            result = child_agent.execute()
            results.append(result)

        all_success = all(r.get("success", False) for r in results)
        return {
            "agent_id": self.agent_id,
            "task_id": self.task.id,
            "success": all_success,
            "child_results": results,
        }