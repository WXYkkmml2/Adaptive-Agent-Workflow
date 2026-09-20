"""
S2 编排智能体。（Step 3 更新：加入偏差检测 + 重规划支持）
"""

import json
import logging
from agents.task import Task
from agents.permission import Permission
from agents.execution_agent import ExecutionAgent
from agents.deviation import Deviation, DeviationType, detect_deviation
from llm.client import LLMClient
from llm.prompts import ORCHESTRATION_SYSTEM, ORCHESTRATION_USER
from grid.tools import get_available_tools, get_tool_catalog, validate_tool_call

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
        d0_info: dict = None,
        certainty: float = 0.7,
        is_replan: bool = False,
        failure_info: dict = None,
    ):
        self.agent_id = agent_id
        self.task = task
        self.permission = permission
        self.network = network
        self.llm = llm
        self.current_depth = current_depth
        self.max_depth = max_depth
        self.prior_results = prior_results or {}
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
        if isinstance(instructions, dict) and instructions.get("error") == "LLM_ERROR":
            return {
                "agent_id": self.agent_id,
                "task_id": self.task.id,
                "success": False,
                "error": instructions.get("message", "LLM 请求失败"),
                "llm_error": True,
            }
        if not instructions:
            return {
                "agent_id": self.agent_id,
                "task_id": self.task.id,
                "success": False,
                "error": "LLM 未返回有效指令",
            }
        validation_error = self._validate_instructions(instructions, available_tools)
        if validation_error:
            logger.warning("%s  编排指令校验失败: %s", indent, validation_error)
            return {
                "agent_id": self.agent_id,
                "task_id": self.task.id,
                "success": False,
                "error": validation_error,
                "deviation": Deviation(
                    deviation_type=DeviationType.PARAMETER,
                    description="编排模型生成了无效工具调用",
                    expected="工具参数与设备索引有效",
                    actual=validation_error,
                    task_id=self.task.id,
                ),
            }
        self.task.device_instructions = instructions

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

    def _validate_instructions(self, instructions: list, available_tools: list) -> str | None:
        if not isinstance(instructions, list):
            return "instructions 必须是数组"
        task_description = (self.task.description or "").strip()
        query_only = task_description.startswith(
            ("分析", "评估", "查询", "检查", "验证", "执行后验证")
        )
        mutation_tools = {"set_gen_voltage", "set_gen_output", "set_line_status", "simulate_action"}
        actual_mutations = {"set_gen_voltage", "set_gen_output", "set_line_status"}
        for index, instruction in enumerate(instructions):
            if not isinstance(instruction, dict):
                return f"instructions[{index}] 必须是对象"
            tool = instruction.get("tool")
            if tool not in available_tools:
                return f"instructions[{index}] 工具 {tool!r} 不在当前权限内"
            if query_only and tool in mutation_tools:
                return f"instructions[{index}] 查询/分析任务不能调用修改或仿真工具 {tool}"
            ok, error = validate_tool_call(tool, instruction.get("params"), self.network.net)
            if not ok:
                return f"instructions[{index}] {tool}: {error}"
        execution_task = task_description.startswith("执行") and not task_description.startswith("执行后")
        if execution_task and not any(
            inst.get("tool") in actual_mutations for inst in instructions
        ):
            return "执行任务必须包含真实修改工具；simulate_action 只修改副本"
        return None

    def _normalize_bus_label(self, instructions: list) -> list:
        """只把明确指向目标母线的越界用户编号转换为内部索引。"""
        if not isinstance(instructions, list) or not self.task.devices:
            return instructions
        target = self.task.devices[0]
        if not isinstance(target, int) or target not in self.network.net.bus.index:
            return instructions
        normalized = []
        for instruction in instructions:
            if not isinstance(instruction, dict):
                normalized.append(instruction)
                continue
            item = dict(instruction)
            params = item.get("params")
            if isinstance(params, dict):
                params = dict(params)
                if params.get("bus_id") == target + 1 and target + 1 not in self.network.net.bus.index:
                    params["bus_id"] = target
                item["params"] = params
            normalized.append(item)
        return normalized

    def _call_llm_decompose(self, available_tools: list) -> list:
        """调用 LLM 细化任务，重规划时附带失败上下文。"""
        system_prompt = ORCHESTRATION_SYSTEM.format(
            permission=self.permission.to_dict(),
            available_devices=self.task.devices,
            available_tools=json.dumps(available_tools, ensure_ascii=False),
            tool_catalog=json.dumps(get_tool_catalog(self.network.net, available_tools, self.task.devices), ensure_ascii=False),
        )

        prior = self._summarize_prior_results(self.prior_results)

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

        indent = "  " * self.current_depth
        response = self.llm.complete_json(
            system_prompt,
            user_prompt,
            temperature=0.2,
            source="orchestration_agent",
            max_tokens=1024,
        )
        if response.get("error") == "LLM_ERROR":
            logger.error(f"{indent}  LLM 错误，取消 S2 编排: {response.get('message', response)}")
            return response
        instructions = self._normalize_bus_label(response.get("instructions", []))
        validation_error = self._validate_instructions(instructions, available_tools)
        if validation_error:
            logger.warning("%s  请求模型修正工具调用: %s", indent, validation_error)
            retry = self.llm.complete_json(
                system_prompt,
                f"任务: {self.task.description}\n目标设备内部索引: {self.task.devices}"
                f"\n上次工具调用不合法: {validation_error}"
                "\n请按系统提供的工具参数和设备索引，重新输出最多4条必要的 instructions JSON。",
                temperature=0.1,
                source="orchestration_repair",
                max_tokens=2048,
            )
            if retry.get("error") == "LLM_ERROR":
                return retry
            instructions = self._normalize_bus_label(retry.get("instructions", []))
        return instructions

    @staticmethod
    def _summarize_prior_results(prior_results: dict) -> str:
        if not prior_results:
            return "无前置结果"

        summary = {}
        for key, value in prior_results.items():
            if isinstance(value, dict):
                if "success" in value:
                    item = {
                        "success": value.get("success"),
                        "tool": value.get("tool"),
                        "error": value.get("error"),
                    }
                    result = value.get("result")
                    if isinstance(result, dict):
                        item["result_summary"] = {
                            "bus_voltages": result.get("bus_voltages") if isinstance(result.get("bus_voltages"), dict) else result.get("bus_voltage"),
                            "line_loadings": result.get("line_loadings") if isinstance(result.get("line_loadings"), dict) else result.get("line_loading"),
                            "violations": result.get("violations") or result.get("constraint_result"),
                        }
                    summary[key] = item
                else:
                    summary[key] = {
                        "success": value.get("success"),
                        "tool": value.get("tool"),
                        "error": value.get("error"),
                    }
            else:
                summary[key] = str(value)

        return json.dumps(summary, ensure_ascii=False)

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
                d0_info=self.d0_info,
                certainty=self.certainty,
                depth=self.current_depth + 1,
            )
            result = exec_agent.execute()
            results.append(result)
            if not result.get("success", False):
                break

        all_success = all(r.get("success", False) for r in results)
        return {
            "agent_id": self.agent_id,
            "task_id": self.task.id,
            "success": all_success,
            "error": next((r.get("error") for r in results if not r.get("success", False)), None),
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
