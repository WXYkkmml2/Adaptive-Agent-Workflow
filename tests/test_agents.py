"""测试 S2 + S3 智能体系统。"""

import pytest
import logging
from grid.network import PowerNetwork
from llm.client import MockLLMClient
from agents.planner import Planner
from agents.root_agent import RootAgent
from agents.task import TaskDAG, Task, TaskStatus
from agents.permission import Permission
from agents.anti_example import AntiExampleStore

# 让测试输出智能体日志
logging.basicConfig(level=logging.WARNING)


@pytest.fixture
def setup():
    """准备完整的测试环境。"""
    network = PowerNetwork()
    llm = MockLLMClient()
    planner = Planner(network, llm)
    plan = planner.plan("Bus 14 电压过低，请恢复")
    return network, llm, plan


def test_root_agent_completes(setup):
    """根智能体应该能跑完整个 DAG。"""
    network, llm, plan = setup
    root = RootAgent(
        network=network,
        dag=plan["dag"],
        llm=llm,
        tree_depth=plan["tree_depth"],
        d0_info=plan["d0_info"],
        certainty=plan["certainty"],
    )
    result = root.execute()
    assert result["success"] is True
    assert all(s == "completed" for s in result["task_status"].values())


def test_dag_respects_dependencies():
    """DAG 应该按依赖顺序释放任务。"""
    dag = TaskDAG()
    dag.add_task(Task(id="t1", description="first"))
    dag.add_task(Task(id="t2", description="second", dependencies=["t1"]))
    dag.add_task(Task(id="t3", description="third", dependencies=["t2"]))

    # 初始只有 t1 就绪
    ready = dag.get_ready_tasks()
    assert len(ready) == 1
    assert ready[0].id == "t1"

    # t1 完成后 t2 就绪
    dag.update_status("t1", TaskStatus.COMPLETED)
    ready = dag.get_ready_tasks()
    assert len(ready) == 1
    assert ready[0].id == "t2"


def test_parallel_tasks():
    """无依赖的任务应该同时就绪。"""
    dag = TaskDAG()
    dag.add_task(Task(id="t1", description="a"))
    dag.add_task(Task(id="t2", description="b"))
    dag.add_task(Task(id="t3", description="c", dependencies=["t1", "t2"]))

    ready = dag.get_ready_tasks()
    assert len(ready) == 2  # t1 和 t2 同时就绪


def test_permission_intersection():
    """子权限应该是父权限的子集。"""
    parent = Permission(
        regions={"ieee14"},
        voltage_levels={"HV", "MV"},
        device_types={"bus", "gen", "line"},
    )
    child_required = Permission(
        regions={"ieee14"},
        voltage_levels={"MV", "LV"},
        device_types={"bus", "gen", "trafo"},
    )
    result = parent.intersect(child_required)
    # MV 是交集
    assert result.voltage_levels == {"MV"}
    # bus 和 gen 是交集，trafo 被父权限排除
    assert result.device_types == {"bus", "gen"}


def test_permission_narrowing():
    """权限只能继承或收窄，不能超出父级。"""
    parent = Permission(
        regions={"ieee14"},
        voltage_levels={"MV"},
        device_types={"bus"},
    )
    child_required = Permission(
        regions={"ieee14"},
        voltage_levels={"HV"},  # 父没有 HV
        device_types={"bus", "gen"},  # 父没有 gen
    )
    result = parent.intersect(child_required)
    assert result.voltage_levels == set()   # 交集为空
    assert result.device_types == {"bus"}   # 只保留 bus


def test_anti_example_empty_store():
    """空反例库匹配应返回空列表，不影响正常流程。"""
    store = AntiExampleStore()
    matched = store.match("查询电压", 0.5, 3)
    assert matched == []


def test_execution_log(setup):
    """根智能体应记录执行日志。"""
    network, llm, plan = setup
    root = RootAgent(
        network=network,
        dag=plan["dag"],
        llm=llm,
        tree_depth=plan["tree_depth"],
        d0_info=plan["d0_info"],
        certainty=plan["certainty"],
    )
    result = root.execute()
    assert len(root.execution_log) > 0
    assert all("task_id" in entry for entry in root.execution_log)