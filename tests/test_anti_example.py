from agents.anti_example import AntiExampleStore, FailureCase


def test_failure_case_matches_same_task():
    store = AntiExampleStore()
    store.add_case(FailureCase(
        case_id="f1", task_description="调高母线电压",
        coupling_strength=0.5, topology_depth=2, certainty=0.7,
        tool_sequence=["simulate_action"], failure_type="constraint",
    ))
    matches = store.match("调高母线电压", 0.5, 2)
    assert len(matches) == 1
    assert matches[0].case_id == "f1"
