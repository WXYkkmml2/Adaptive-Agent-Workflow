"""
S4 偏差检测。

"执行智能体回传的状态和预期状态比对，发现差异、偏差。
 偏差特征包括 2 个部分：
   一是偏差类型（权限越权、违反物理约束、参数有效异常、工具调用故障）
   二是上下文信息（触发偏差时的电网信息、已执行动作序列、设备状态）"

"预期状态来自编排智能体下发的任务本身，
 比如编排智能体给执行智能体的指令是断开断路器，
 那预期状态就是断路器分闸。"
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class DeviationType(Enum):
    """
    偏差类型。不同类型采用不同的处理方式：
    - PERMISSION:  权限越界 → 回到上层重新拆解
    - CONSTRAINT:  违反物理约束 → 回到编排层重规划
    - PARAMETER:   参数有效异常 → 回到 S3 重新调用 LLM
    - TOOL_FAULT:  工具接口故障 → 等待修复或切换备用
    - INSUFFICIENT: 操作执行了但效果不足 → 重新生成方案
    """
    PERMISSION = "permission_violation"
    CONSTRAINT = "constraint_violation"
    PARAMETER = "parameter_error"
    TOOL_FAULT = "tool_fault"
    INSUFFICIENT = "insufficient_effect"


@dataclass
class Deviation:
    """一条偏差记录。"""
    deviation_type: DeviationType
    description: str
    expected: str                                  # 预期结果
    actual: str                                    # 实际结果
    task_id: str = ""
    agent_path: list = field(default_factory=list)  # 从根到出错节点的路径
    context: dict = field(default_factory=dict)      # 出错时的电网快照等

    def summary(self) -> str:
        return (
            f"[{self.deviation_type.value}] {self.description}\n"
            f"  预期: {self.expected}\n"
            f"  实际: {self.actual}"
        )


def detect_deviation(
    task_id: str,
    expected_description: str,
    execution_results: list,
    agent_path: list = None,
    network=None,
) -> Optional[Deviation]:
    """
    检测执行结果与预期之间的偏差。

    "编排智能体在下发指令给执行智能体的同时保留预期，
     执行智能体执行完回传实际设备状态，
     编排智能体拿回传数据和自己保留的预期做比对。"

    参数:
        task_id: 任务 ID
        expected_description: 编排智能体下发时的预期结果描述
        execution_results: 执行智能体回传的结果列表
        agent_path: 从根到当前节点的路径
        network: 电网对象（用于获取当前状态作为上下文）
    """
    if not execution_results:
        return Deviation(
            deviation_type=DeviationType.TOOL_FAULT,
            description="执行智能体未返回任何结果",
            expected=expected_description,
            actual="无结果",
            task_id=task_id,
            agent_path=agent_path or [],
        )

    for result in execution_results:
        # ---- 工具调用失败 ----
        if not result.get("success", False):
            error = result.get("error", "未知错误")

            # 判断失败类型
            if "权限" in error or "permission" in error.lower():
                dtype = DeviationType.PERMISSION
            elif "不收敛" in error or "越限" in error:
                dtype = DeviationType.CONSTRAINT
            elif "不存在" in error or "参数" in error:
                dtype = DeviationType.PARAMETER
            else:
                dtype = DeviationType.TOOL_FAULT

            return Deviation(
                deviation_type=dtype,
                description=f"工具 {result.get('tool', '?')} 执行失败: {error}",
                expected=expected_description,
                actual=error,
                task_id=task_id,
                agent_path=agent_path or [],
            )

        # ---- 约束违规检测 ----
        tool_result = result.get("result", result.get("tool_results", {}))
        if isinstance(tool_result, dict):
            # check_constraints 返回的结果
            if "violations" in tool_result and tool_result.get("violation_count", 0) > 0:
                violations = tool_result["violations"]
                return Deviation(
                    deviation_type=DeviationType.CONSTRAINT,
                    description=f"操作后存在 {len(violations)} 项约束违规",
                    expected=expected_description,
                    actual=str(violations[:3]),  # 只记前 3 条
                    task_id=task_id,
                    agent_path=agent_path or [],
                    context={"violations": violations},
                )

            # simulate_action 返回的结果中嵌套了约束信息
            if "all_satisfied" in tool_result and not tool_result["all_satisfied"]:
                return Deviation(
                    deviation_type=DeviationType.CONSTRAINT,
                    description="仿真结果显示约束不满足",
                    expected=expected_description,
                    actual=str(tool_result.get("violations", [])),
                    task_id=task_id,
                    agent_path=agent_path or [],
                )

    return None  # 无偏差