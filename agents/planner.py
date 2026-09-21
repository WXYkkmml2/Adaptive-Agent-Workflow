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
from llm.client import LLMClient, LLMServiceUnavailable
from llm.prompts import PLANNER_SYSTEM, PLANNER_USER
from config.settings import (
    BUS_VOLTAGE_MIN, BUS_VOLTAGE_MAX, LINE_LOADING_MAX,
)

logger = logging.getLogger(__name__)


class Planner:
    """
    S1 规划器。

    输入：自然语言调度指令
    输出：PlanResult，包含 G（任务DAG）、C、D0、H
    """

    def __init__(self, network: PowerNetwork, llm: LLMClient, depth_mode: str = "adaptive"):
        self.network = network
        self.llm = llm
        if depth_mode not in ("adaptive", "fixed"):
            raise ValueError("depth_mode 必须是 adaptive 或 fixed")
        self.depth_mode = depth_mode

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
        if self.network.case == "case39":
            return self._plan_regional(instruction)
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
        logger.info(f"  结构代理指标 C = {certainty:.4f}（非 token/注意力确定性）")

        # ---- 5. 确定树深度 H ----
        tree_depth = 3 if self.depth_mode == "fixed" else self._compute_tree_depth(d0, certainty)
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
        if self.network.case == "case39":
            return self._parse_targets(instruction)
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

    def _parse_targets(self, instruction):
        """Resolve each permitted region's lowest voltage buses, including ties."""
        from grid.goal import Goal, goal_status
        goal = Goal.from_instruction(instruction)
        targets = {}
        for zone, violations in goal_status(self.network.net, goal)["violations_by_zone"].items():
            if int(zone) in goal.forbidden_regions:
                continue
            low = [v for v in violations if v["type"] == "voltage_low"]
            if low:
                minimum = min(v["value"] for v in low)
                targets[int(zone)] = [v["bus_id"] for v in low if abs(v["value"] - minimum) < 1e-5]
        return targets

    def _plan_regional(self, instruction):
        import json
        from grid.goal import Goal, goal_status
        from grid.topology import regional_scope, regional_tree_depth
        from llm.prompts import CASE39_PLANNER_SYSTEM
        from config.settings import CASE39_TEMPERATURE, CASE39_MAX_TOKENS
        targets = self._parse_targets(instruction)
        if not targets:
            raise ValueError("No regional low-voltage targets")
        goal = Goal.from_instruction(instruction)
        scope = regional_scope(self.network.net, [b for ids in targets.values() for b in ids], goal.forbidden_regions)
        response = self.llm.complete_json(CASE39_PLANNER_SYSTEM, json.dumps({
            "instruction": instruction, "targets_by_region": targets,
            "current_goal": goal_status(self.network.net, goal),
            "d0": [{k: v for k, v in i.items() if k in ("bus_id", "d0", "topology_depth_b")} for i in scope["d0_info"]]
        }, ensure_ascii=False), temperature=CASE39_TEMPERATURE, max_tokens=CASE39_MAX_TOKENS, source="planner")
        if not isinstance(response, dict):
            raise ValueError("Planner response must be an object")
        if response.get("error") == "LLM_ERROR":
            if response.get("retryable"):
                raise LLMServiceUnavailable(response.get("message", "service unavailable"))
            raise ValueError(response.get("message", "Invalid planner response"))
        dag = TaskDAG()
        items = response.get("tasks")
        if not isinstance(items, list) or not items:
            raise ValueError("LLM did not produce a DAG")
        for item in items:
            if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not item["id"]:
                raise ValueError("Planner task requires explicit id")
            if not isinstance(item.get("devices"), list) or not item["devices"]:
                raise ValueError("Planner task requires explicit devices")
            kind = item.get("device_type")
            if kind not in ("bus", "gen", "line", "trafo"):
                raise ValueError("Invalid task device_type")
            if any(isinstance(i, bool) or not isinstance(i, int) or i not in self.network.net[kind].index for i in item["devices"]):
                raise ValueError("Invalid task device index")
            if (not isinstance(item.get("dependencies"), list)
                    or any(not isinstance(d, str) for d in item["dependencies"])):
                raise ValueError("Task requires dependencies array of IDs")
            if not isinstance(item.get("description"), str) or not item["description"].strip():
                raise ValueError("Task requires description")
            dag.add_task(Task(id=item["id"], description=item.get("description", ""),
                              devices=item["devices"], device_type=kind, dependencies=item["dependencies"]))
        if not dag.tasks:
            raise ValueError("LLM did not produce a DAG")
        dag.validate()
        return {"instruction": instruction, "target_bus": None, "targets_by_region": targets,
                "dag": dag, "scope": scope, "d0_info": {"d0": max(i["d0"] for i in scope["d0_info"])},
                "certainty": self._compute_certainty(dag), "tree_depth": regional_tree_depth(scope, self.network.net)}

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
            max_tokens=2048,
        )
        if response.get("error") == "LLM_ERROR":
            if response.get("retryable"):
                raise LLMServiceUnavailable(response.get("message", "LLM 服务暂时不可用"))
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

            devices = t_data.get("devices", [target_bus])
            if not isinstance(devices, list):
                devices = [target_bus]
            devices = [target_bus if device == target_bus + 1 and device not in self.network.net.bus.index
                       else device for device in devices]
            task = Task(
                id=task_id,
                description=t_data.get("description", ""),
                dependencies=t_data.get("dependencies", []),
                devices=devices,
                device_type=t_data.get("device_type", "bus"),
                voltage_level=t_data.get("voltage_level", "MV"),
            )
            dag.add_task(task)

        if not dag.tasks:
            raise ValueError("LLM 未返回有效任务图；请检查规划提示词或模型响应")

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
        """由 D0 决定深度；C 暂只记录，不参与增层。"""
        return d0_to_h0(d0)
