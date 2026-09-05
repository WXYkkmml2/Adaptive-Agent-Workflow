"""
反例库（基础结构）。

"利用当前的任务描述和电气耦合强度 a，做相似度匹配，
 提取类似的失败案例。如果匹配到了，获取失败案例的调用工具的特征，
 识别出高风险动作，然后在大模型输出的工具调用概率分布中降低权重。"

Step 2 先建好结构和匹配接口，库里是空的。
Step 3（S4）会往里填充失败案例。
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


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
    """
    反例库。使用简单的文本相似度匹配（TF-IDF + 余弦相似度）。
    对应原文档："反例库采用向量数据库存储，支持基于特征相似度的快速检索。"
    在 MVP 中用 scikit-learn 的 TF-IDF 代替专门的向量数据库。
    """

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

        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        # 构建文本语料：当前描述 + 所有反例描述
        corpus = [task_description] + [c.task_description for c in self.cases]

        try:
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(corpus)
        except ValueError:
            # 语料太短或全是停用词，无法构建 TF-IDF
            return []

        # 当前描述和各反例的文本相似度
        text_sims = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

        # 数值相似度（a 和 b 的归一化距离）
        scored_cases = []
        for i, case in enumerate(self.cases):
            a_diff = abs(coupling_strength - case.coupling_strength)
            b_diff = abs(topology_depth - case.topology_depth) / max(topology_depth, 1)
            # 数值距离转相似度
            num_sim = 1.0 - min((a_diff + b_diff) / 2, 1.0)
            # 综合得分：文本 70% + 数值 30%
            score = 0.7 * text_sims[i] + 0.3 * num_sim
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