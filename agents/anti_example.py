"""进程内失败案例库：按任务描述和电网特征检索，供重规划参考。"""

from dataclasses import dataclass, field
from difflib import SequenceMatcher


@dataclass
class FailureCase:
    """一条反例记录。"""
    case_id: str
    task_description: str           # 失败时的任务描述
    coupling_strength: float        # 当时的电气耦合强度 a
    topology_depth: int             # 当时的 BFS 深度 b
    certainty: float                # 当时的确定性指标 C
    tool_sequence: list             # 失败的工具调用序列
    failure_type: str               # 失败分类：execution/decomposition/permission/constraint
    context: dict = field(default_factory=dict)  # 触发失败时的上下文


class AntiExampleStore:
    """反例库，使用标准库文本相似度和电网特征匹配。"""

    def __init__(self):
        self.cases: list[FailureCase] = []

    def add_case(self, case: FailureCase):
        self.cases.append(case)

    def match(
        self,
        task_description: str,
        coupling_strength: float,
        topology_depth: int,
        top_k: int = 3,
        similarity_threshold: float = 0.3,
    ) -> list[FailureCase]:
        """
        匹配相似的失败案例。

        匹配策略（简化版）：
        1. 文本相似度：当前任务描述 vs 反例的任务描述
        2. 数值距离：a 和 b 的差异
        3. 综合排序，返回 top_k 个超过阈值的

        如果库是空的，返回空列表（不影响正常流程）。
        """
        if not self.cases:
            return []

        # 数值相似度（a 和 b 的归一化距离）
        scored_cases = []
        for i, case in enumerate(self.cases):
            text_sim = SequenceMatcher(None, task_description, case.task_description).ratio()
            a_diff = abs(coupling_strength - case.coupling_strength)
            b_diff = abs(topology_depth - case.topology_depth) / max(topology_depth, 1)
            # 数值距离转相似度
            num_sim = 1.0 - min((a_diff + b_diff) / 2, 1.0)
            # 综合得分：文本 70% + 数值 30%
            score = 0.7 * text_sim + 0.3 * num_sim
            if score >= similarity_threshold:
                scored_cases.append((score, case))

        # 按得分降序排，取 top_k
        scored_cases.sort(key=lambda x: x[0], reverse=True)
        return [case for _, case in scored_cases[:top_k]]

    def get_downweight_tools(self, matched_cases: list[FailureCase]) -> dict:
        """
        从匹配到的反例中提取需要降权的工具调用模式。

        对应原文档：
        "降权就是把其中与高风险模式相似的候选方案的概率乘以一个小于 1 的系数"

        返回: {tool_name: downweight_factor}
        系数越小，降权越狠。
        """
        tool_penalties = {}
        for case in matched_cases:
            for tool_call in case.tool_sequence:
                tool_name = tool_call if isinstance(tool_call, str) else tool_call.get("tool", "")
                if tool_name:
                    # 每多一个反例命中同一个工具，降权更多
                    current = tool_penalties.get(tool_name, 1.0)
                    tool_penalties[tool_name] = current * 0.7  # 每次乘 0.7
        return tool_penalties

    def size(self) -> int:
        return len(self.cases)
