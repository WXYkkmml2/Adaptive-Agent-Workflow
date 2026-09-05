"""
S2 编排智能体。（Step 3 更新：加入偏差检测 + 重规划支持）
"""

import json
import logging
from agents.task import Task
from agents.permission import Permission
from agents.execution_agent import ExecutionAgent
from agents.deviation import detect_deviation
from llm.client import LLMClient
from llm.prompts import ORCHESTRATION_SYSTEM, ORCHESTRATION_USER
from grid.tools import get_available_tools

logger = logging.getLogger(__name__)


class OrchestrationAgent:
    """
    编排智能体。
    
    Step 3 新增：
    - 偏差检测：执行智能体回传后，对比预期状态
    - 重规划标记：如果是重规划实例，LLM prompt 中附带失败上下文
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
        is_replan: bool = False,       # ← Step 3 新增
        failure_info: dict = None,     # ← Step 3 新增
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
        self.is_replan = is_replan
        self.failure_info = failure_info or {}

    @property
    def is_last_orchestration_layer(self) -> bool:
        return self.current_depth >= self.max_depth - 2

    def execute(self) -> dict:
        indent = "  " * self.current_depth
        replan_tag = " [重规划]" if self.is_replan else ""
        logger.info(
            f"{indent}[{self.agent_id}]{replan_tag} "
            f"编排层 depth={self.current_depth}, "
            f"任务: {self.task.description}"
        )

        # ---- 1. 权限裁剪 ----
        available_tools = get_available_tools(self.permission.to_dict())
        logger.info(
            f"{indent}  可用工具({len(available_tools)}): {available_tools}"
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
            result = self._dispatch_to_execution(instructions)
        else:
            result = self._dispatch_to_next_orchestration(instructions)

        # ---- 4. 偏差检测（Step 3 新增） ----
        # 对应原文档：
        # "编排智能体拿回传数据和自己保留的预期做比对，
        #  发现对不上就生成偏差特征。"
        if result.get("success", False):
            deviation = self._check_deviation(result, instructions)
            if deviation is not None:
                logger.warning(
                    f"{indent}  ⚠ 检测到偏差: {deviation.summary()}"
                )
                result["success"] = False
                result["deviation"] = deviation
                result["error"] = deviation.description

        return result

    def _call_llm_decompose(self, available_tools: list) -> list:
        """调用 LLM 细化任务，重规划时附带失败上下文。"""

        system_prompt = ORCHESTRATION_SYSTEM.format(
            permission=self.permission.to_dict(),
            available_devices=self.task.devices,
        )

        # 构建用户 prompt
        prior = str(self.prior_results)

        # ---- Step 3 新增：重规划时附带失败信息 ----
        if self.is_replan and self.failure_info:
            prior += (
                f"\n\n【注意：这是重规划。上次失败信息如下】\n"
                f"失败类型: {self.failure_info.get('previous_failure', {}).get('type', '未知')}\n"
                f"失败描述: {self.failure_info.get('previous_failure', {}).get('description', '无')}\n"
                f"建议: {self.failure_info.get('replan_guidance', '请重新分析')}\n"
            )

        user_prompt = ORCHESTRATION_USER.format(
            task_description=self.task.description,
            devices=self.task.devices,
            prior_results=prior,
        )

        response = self.llm.complete_json(system_prompt, user_prompt)
        return response.get("instructions", [])

    def _dispatch_to_execution(self, instructions: list) -> dict:
        results = []
        for i, inst in enumerate(instructions):
            exec_id = f"exec_{self.task.id}_{i}"
            exec_agent = ExecutionAgent(
                agent_id=exec_id,
                instruction=inst,
                permission=self.permission,
                network=self.network,
                llm=self.llm,
                anti_example_store=self.anti_example_store,
                d0_info=self.d0_info,
                certainty=self.certainty,
                depth=self.current_depth + 1,
            )
            result = exec_agent.execute()
            results.append(result)

        all_success = all(r.get("success", False) for r in results)
        return {
            "agent_id": self.agent_id,
            "task_id": self.task.id,
            "success": all_success,
            "execution_results": results,
            "expected_results": [
                inst.get("expected_result", "") for inst in instructions
            ],
        }

    def _dispatch_to_next_orchestration(self, instructions: list) -> dict:
        results = []
        for i, inst in enumerate(instructions):
            sub_task = Task(
                id=f"{self.task.id}_sub{i}",
                description=inst.get("description", ""),
                devices=self.task.devices,
                device_type=self.task.device_type,
                voltage_level=self.task.voltage_level,
            )
            child_id = f"orch_{self.current_depth + 1}_{sub_task.id}"
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
                is_replan=self.is_replan,
                failure_info=self.failure_info,
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

    def _check_deviation(self, result: dict, instructions: list) -> "Deviation | None":
        """
        S4 偏差检测。
        
        对应原文档：
        "父编排智能体知道自己下发了什么指令、预期结果应该是什么，
         它拿回传数据和预期做对比。"
        """
        exec_results = result.get("execution_results", [])
        if not exec_results:
            exec_results = result.get("child_results", [])

        # 收集所有子结果中的工具执行结果（可能嵌套）
        flat_results = self._flatten_results(exec_results)

        expected_desc = "; ".join(
            inst.get("expected_result", "") for inst in instructions
        )

        return detect_deviation(
            task_id=self.task.id,
            expected_description=expected_desc,
            execution_results=flat_results,
            agent_path=[self.agent_id],
            network=self.network,
        )

    def _flatten_results(self, results: list) -> list:
        """递归展开嵌套的执行结果。"""
        flat = []
        for r in results:
            if "tool_results" in r:
                flat.extend(r["tool_results"])
            elif "execution_results" in r:
                flat.extend(self._flatten_results(r["execution_results"]))
            elif "child_results" in r:
                flat.extend(self._flatten_results(r["child_results"]))
            else:
                flat.append(r)
        return flat