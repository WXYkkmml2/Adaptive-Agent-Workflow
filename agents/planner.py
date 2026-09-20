"""
S1 规划器。

职责链：
  自然语言指令 → 解析目标设备 → 计算 D0 → 调用 LLM 生成 G
  → 计算确定性指标 C → 确定树深度 H

"""

import re
import logging
from grid.network import PowerNetwork
from grid.topology import compute_d0, d0_to_h0
from agents.task import Task, TaskDAG, TaskStatus
from llm.client import LLMClient
from llm.prompts import PLANNER_SYSTEM, PLANNER_USER
from config.settings import (
    BUS_VOLTAGE_MIN, BUS_VOLTAGE_MAX, LINE_LOADING_MAX,
    C_THRESHOLD_HIGH, C_THRESHOLD_LOW, H_MAX,
)

logger = logging.getLogger(__name__)


class Planner:
    """
    S1 规划器。

    输入：自然语言调度指令
    输出：PlanResult，包含 G（任务DAG）、C、D0、H
    """

    def __init__(self, network: PowerNetwork, llm: LLMClient):
        self.network = network
        self.llm = llm

    def plan(self, instruction: str) -> dict:
        """
        执行完整的 S1 流程。

        返回 PlanResult 字典:
        {
            "instruction": 原始指令,
            "target_bus": 解析出的目标母线,
            "d0_info": D0 计算详情,
            "dag": TaskDAG 对象,
            "certainty": C 值,
            "tree_depth": H 值,
        }
        """
        logger.info(f"S1 开始规划: {instruction}")

        # ---- 1. 解析目标设备 ----
        # "这里直接简单解析就行，先不用大模型"
        target_bus = self._parse_target(instruction)
        logger.info(f"  解析目标母线: Bus {target_bus}")

        # ---- 2. 计算 D0 ----
        d0_info = compute_d0(self.network.net, target_bus)
        d0 = d0_info["d0"]
        logger.info(f"  D0 = {d0:.4f} (a={d0_info['coupling_strength_a']:.4f}, "
                     f"b={d0_info['topology_depth_b']})")

        # ---- 3. 调用 LLM 生成 G ----
        dag = self._generate_task_dag(instruction, target_bus, d0_info)
        logger.info(f"  生成任务 DAG: {len(dag.tasks)} 个任务\n{dag.summary()}")

        # ---- 4. 计算确定性指标 C ----
        certainty = self._compute_certainty(dag)
        logger.info(f"  确定性指标 C = {certainty:.4f}")

        # ---- 5. 确定树深度 H ----
        tree_depth = self._compute_tree_depth(d0, certainty)
        logger.info(f"  智能体树深度 H = {tree_depth}")

        return {
            "instruction": instruction,
            "target_bus": target_bus,
            "d0_info": d0_info,
            "dag": dag,
            "certainty": certainty,
            "tree_depth": tree_depth,
        }

    def _parse_target(self, instruction: str) -> int:
        """
        简单 NLP 解析：从指令中提取目标母线编号。
        "Bus 14 电压过低" → 目标是 Bus 13（0-indexed）。

        原文档："稿件有缺失，这里可以用 NLP 等等"
        MVP 用正则匹配。
        """
        # 匹配 "Bus X" 或 "母线X" 或 "节点X"
        patterns = [
            r"[Bb]us\s*(\d+)",
            r"母线\s*(\d+)",
            r"节点\s*(\d+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, instruction)
            if match:
                bus_num = int(match.group(1))
                # IEEE 14-bus 编号从 1 开始，pandapower 索引从 0 开始
                # 如果用户说 "Bus 14"，内部用 13
                if bus_num >= 1 and bus_num <= 14:
                    return bus_num - 1
                return bus_num

        # 没找到明确目标，默认 Bus 13（最末端，常用测试点）
        logger.warning("未能解析目标母线，默认使用 Bus 13")
        return 13

    def _generate_task_dag(self, instruction: str, target_bus: int, d0_info: dict) -> TaskDAG:
        """
        调用 LLM 生成结构化任务集合 G。

        对应原文档：
        "用额外提示词要求大模型按照预定结构化格式输出，
         例如用 JSON 结构，分别给出任务 ID、任务描述、依赖等等，
         然后程序解析结构化输出形成 G"
        """
        # 获取当前电网状态信息，作为上下文
        target_voltage = self.network.net.res_bus.at[target_bus, "vm_pu"]
        neighbors = self.network.get_bus_connections(target_bus)
        neighbor_ids = []
        for line_id in neighbors["lines"]:
            from_bus = int(self.network.net.line.at[line_id, "from_bus"])
            to_bus = int(self.network.net.line.at[line_id, "to_bus"])
            neighbor_ids.append(from_bus if from_bus != target_bus else to_bus)

        user_prompt = PLANNER_USER.format(
            instruction=instruction,
            target_bus=target_bus,
            d0=d0_info["d0"],
            v_min=BUS_VOLTAGE_MIN,
            v_max=BUS_VOLTAGE_MAX,
            l_max=LINE_LOADING_MAX,
            target_voltage=target_voltage,
            neighbor_buses=neighbor_ids,
        )

        # 调用 LLM
        response = self.llm.complete_json(
            PLANNER_SYSTEM,
            user_prompt,
            temperature=0.2,
            source="planner",
            max_tokens=1200,
        )
        if response.get("error") == "LLM_ERROR":
            raise RuntimeError(f"任务规划 API 请求失败: {response.get('message', response)}")

        # 解析为 TaskDAG
        dag = TaskDAG()
        tasks_list = response.get("tasks", []) if isinstance(response, dict) else []
        for idx, t_data in enumerate(tasks_list):
            if not isinstance(t_data, dict):
                logger.warning(f"LLM 返回的任务项不是字典，跳过：{str(t_data)[:200]}")
                continue

            task_id = t_data.get("id")
            if not task_id:
                # 自动生成一个任务 id，避免 KeyError；同时记录警告和原始条目
                task_id = f"t{idx+1}"
                logger.warning(f"LLM 返回的任务缺少 'id' 字段，使用自动 id={task_id}，原始条目预览: {str(t_data)[:200]}")

            # 若生成的 id 与已存在冲突，则调整序号以保证唯一性
            base_id = task_id
            counter = 1
            while task_id in dag.tasks:
                task_id = f"{base_id}_{counter}"
                counter += 1

            task = Task(
                id=task_id,
                description=t_data.get("description", ""),
                dependencies=t_data.get("dependencies", []),
                devices=t_data.get("devices", [target_bus]),
                device_type=t_data.get("device_type", "bus"),
                voltage_level=t_data.get("voltage_level", "MV"),
            )
            dag.add_task(task)

        if not dag.tasks:
            logger.error("LLM 未返回有效任务，使用降级方案")
            dag = self._fallback_dag(target_bus)

        dag.validate()

        return dag

    def _compute_certainty(self, dag: TaskDAG) -> float:
        """
        计算确定性指标 C（简化版）。

        原文档描述的完整 C 计算需要：
        1. 获取每个 token 的 logits，算最高概率 - 次高概率
        2. 获取多头注意力权重，筛选超过阈值的头做融合
        3. 用融合注意力权重对概率差做加权平均

        标准 API 不暴露这些内部状态，所以 MVP 用启发式方法：
        - 任务有完整依赖链 → 更确定
        - 任务描述具体（包含设备类型）→ 更确定
        - 任务数量合理（不太少也不太多）→ 更确定
        
        后续接入支持 logprobs 的 API 后可替换为真实计算。
        """
        if not dag.tasks:
            return 0.0

        scores = []

        # 因素 1：依赖关系完整性（有依赖说明任务间逻辑清晰）
        tasks_with_deps = sum(1 for t in dag.tasks.values() if t.dependencies)
        dep_ratio = tasks_with_deps / len(dag.tasks) if len(dag.tasks) > 1 else 0.5
        scores.append(dep_ratio)

        # 因素 2：任务描述具体性（有设备类型 = 更具体）
        typed_tasks = sum(1 for t in dag.tasks.values() if t.device_type)
        type_ratio = typed_tasks / len(dag.tasks)
        scores.append(type_ratio)

        # 因素 3：任务数量合理性（3~8 个最佳）
        count = len(dag.tasks)
        if 3 <= count <= 8:
            count_score = 1.0
        elif count < 3:
            count_score = count / 3
        else:
            count_score = max(0.3, 1.0 - (count - 8) * 0.1)
        scores.append(count_score)

        # 加权平均
        c = 0.4 * scores[0] + 0.3 * scores[1] + 0.3 * scores[2]
        return round(min(max(c, 0.0), 1.0), 4)

    def _compute_tree_depth(self, d0: float, certainty: float) -> int:
        """
        确定智能体树深度 H = f(C, D0)。

        对应原文档：
        "先将 D0 通过预设区间，映射为基础纵向深度 H0；
         高 C 时保持 H0；
         C 中间位置，额外增加编排中间层数量"

        具体逻辑：
        1. D0 → H0（基础深度）
        2. C > T1（高确定性）→ H = H0
        3. T2 < C < T1（中间）→ 按 C 在 [T1,T2] 的位置线性增加层数
        4. C < T2（低确定性）→ 按文档走独立子树（Step 3 实现），这里先给最大增量
        """
        h0 = d0_to_h0(d0)

        if certainty >= C_THRESHOLD_HIGH:
            # 高确定性，保持基础深度
            h = h0
        elif certainty >= C_THRESHOLD_LOW:
            # 中间区域：C 越低，加越多层
            # 把 [T1, T2] 等分，C 靠近 T2 加更多层
            range_size = C_THRESHOLD_HIGH - C_THRESHOLD_LOW
            position = (C_THRESHOLD_HIGH - certainty) / range_size  # 0~1, 越大越不确定
            # 最多额外加 2 层
            extra_layers = round(position * 2)
            h = h0 + extra_layers
        else:
            # 低确定性，给最大增量（实际应拆子树，Step 3 实现）
            h = h0 + 2

        # 硬上限
        h = min(h, H_MAX)
        return h

    def _fallback_dag(self, target_bus: int) -> TaskDAG:
        """
        降级方案：LLM 失败时的最小可用任务 DAG。
        """
        dag = TaskDAG()
        dag.add_task(Task(
            id="t1", description="查询目标节点电压",
            devices=[target_bus], device_type="bus",
        ))
        dag.add_task(Task(
            id="t2", description="查询邻近节点状态",
            devices=[target_bus], device_type="bus",
        ))
        dag.add_task(Task(
            id="t3", description="验证全网约束",
            dependencies=["t1", "t2"],
            devices=[target_bus], device_type="bus",
        ))
        return dag
