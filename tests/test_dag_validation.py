import pytest
from agents.task import Task, TaskDAG


def test_missing_dependency_rejected():
    dag = TaskDAG()
    dag.add_task(Task(id="a", description="a", dependencies=["missing"]))
    with pytest.raises(ValueError, match="依赖不存在"):
        dag.get_ready_tasks()


def test_cycle_rejected():
    dag = TaskDAG()
    dag.add_task(Task(id="a", description="a", dependencies=["b"]))
    dag.add_task(Task(id="b", description="b", dependencies=["a"]))
    with pytest.raises(ValueError, match="存在环"):
        dag.validate()


def test_valid_dag():
    dag = TaskDAG()
    dag.add_task(Task(id="a", description="a"))
    dag.add_task(Task(id="b", description="b", dependencies=["a"]))
    dag.validate()
    assert [task.id for task in dag.get_ready_tasks()] == ["a"]
