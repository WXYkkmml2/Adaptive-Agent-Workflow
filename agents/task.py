"""
任务与任务 DAG 的数据结构。
"G 是 S2 需要分配的任务单元的集合，其中存在一定的逻辑，
 如 t1 完成了 → t2 可以跟进。G 本质上可以看作图，
 每个小任务是节点，节点连接是依赖关系。"
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class TaskStatus(Enum):
    PENDING = "pending"       # 未激活：前置依赖未满足
    ACTIVE = "active"         # 执行中
    COMPLETED = "completed"   # 已完成
    FAILED = "failed"         # 失败


@dataclass
class Task:
    """一个任务单元（G 中的节点）。"""
    id: str
    description: str
    dependencies: list = field(default_factory=list)   # 前置依赖的任务 ID
    devices: list = field(default_factory=list)         # 涉及的设备 ID
    device_type: str = ""                               # 主要设备类型
    voltage_level: str = "MV"                           # 电压等级
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[dict] = None                       # 执行结果

    # 编排层拆解后产生的设备级指令（到达最后编排层时填充）
    device_instructions: list = field(default_factory=list)


@dataclass
class TaskDAG:
    """
    任务有向无环图。

    根智能体用这个结构管理任务状态：
    "根智能体维护一个任务状态表（未激活、执行中、已完成、失败），
     检查 G 中哪些任务的前置依赖已经全部完成，
     对这些任务分别实例化编排层智能体。"
    """
    tasks: dict = field(default_factory=dict)  # id → Task

    def add_task(self, task: Task):
        if task.id in self.tasks:
            raise ValueError(f"重复任务 ID: {task.id}")
        self.tasks[task.id] = task

    def validate(self) -> None:
        """验证依赖存在且任务图无环。"""
        state = {}

        def visit(task_id):
            if state.get(task_id) == 1:
                raise ValueError(f"任务依赖存在环: {task_id}")
            if state.get(task_id) == 2:
                return
            state[task_id] = 1
            for dep_id in self.tasks[task_id].dependencies:
                if dep_id not in self.tasks:
                    raise ValueError(f"任务 {task_id} 依赖不存在: {dep_id}")
                visit(dep_id)
            state[task_id] = 2

        for task_id in self.tasks:
            visit(task_id)

    def get_ready_tasks(self) -> list:
        """
        返回所有前置依赖已完成的 PENDING 任务。
        这些任务可以被根智能体同时激活（并行）。
        
        "t1 和 t5 之间没有依赖关系，那可以初始时，
         同时激活两个编排智能体分别处理 t1 和 t5"
        """
        self.validate()
        ready = []
        for task in self.tasks.values():
            if task.status != TaskStatus.PENDING:
                continue
            deps_met = all(
                self.tasks[dep_id].status == TaskStatus.COMPLETED
                for dep_id in task.dependencies
            )
            if deps_met:
                ready.append(task)
        return ready

    def update_status(self, task_id: str, status: TaskStatus, result=None):
        if task_id in self.tasks:
            self.tasks[task_id].status = status
            if result is not None:
                self.tasks[task_id].result = result

    def all_completed(self) -> bool:
        return all(
            t.status == TaskStatus.COMPLETED for t in self.tasks.values()
        )

    def has_failed(self) -> bool:
        return any(
            t.status == TaskStatus.FAILED for t in self.tasks.values()
        )

    def get_task_results(self) -> dict:
        """汇总所有已完成任务的结果，供后续任务使用。"""
        return {
            tid: t.result
            for tid, t in self.tasks.items()
            if t.status == TaskStatus.COMPLETED and t.result is not None
        }

    def summary(self) -> str:
        lines = []
        for tid, t in self.tasks.items():
            deps = ",".join(t.dependencies) if t.dependencies else "无"
            lines.append(
                f"  {tid}: {t.description}  "
                f"[依赖: {deps}] [{t.status.value}]"
            )
        return "\n".join(lines)
