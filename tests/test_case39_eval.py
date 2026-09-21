from grid.scenario_case39 import make_network, validate_scenario
from grid.goal import Goal, goal_status
from run_case39 import Case39Trial, INSTRUCTION, run_once
from grid.tools import get_tool_counters, reset_tool_counters


class ScriptedLLM:
    total_tokens = 0

    def complete_json(self, system_prompt, user_prompt, **kwargs):
        import json
        context = json.loads(user_prompt)
        region = context['region']
        if region == 2:
            actions = [(0, .98), (6, .98)]
        elif region == 1:
            actions = [(1, 1.06)]
        else:
            actions = [(0, .98), (6, .98), (1, 1.06)]
        return {'actions': [{'type': 'set_gen_voltage', 'gen_id': i, 'vm_pu': vm}
                            for i, vm in actions]}


def test_fixture_and_budget_are_offline():
    assert validate_scenario()
    net = make_network()
    goal = Goal.from_instruction(INSTRUCTION)
    assert len(net.action_log) == 0
    assert goal.max_real_actions == 6
    assert goal.forbidden_regions == frozenset({3})
    assert goal_status(net.net, goal)['violation_count'] == 9


def test_three_methods_share_successful_tool_path():
    for method in ('hierarchical', 'two_layer', 'two_layer_full_restart'):
        row = run_once(method, 1, ScriptedLLM())
        assert row['success'] == 1, (method, row['error'])
        assert row['real_actions_used'] == 3
        assert row['zone3_touched'] == 0
        assert row['tree_depth'] == (3 if method == 'hierarchical' else 2)


def test_hidden_limit_replans_only_region1_after_region2_commit():
    class LimitedLLM(ScriptedLLM):
        def complete_json(self, system_prompt, user_prompt, **kwargs):
            import json
            context = json.loads(user_prompt)
            if context['region'] == 1 and not context['feedback']:
                return {'actions': [{'type': 'set_gen_voltage', 'gen_id': 1, 'vm_pu': 1.03},
                                    {'type': 'set_gen_voltage', 'gen_id': 8, 'vm_pu': 1.06}]}
            return super().complete_json(system_prompt, user_prompt, **kwargs)

    trial = Case39Trial('hierarchical', LimitedLLM())
    success, error = trial.run()
    assert success, error
    assert trial.replanned == ['region1']
    assert [item[1] for item in trial.network.action_log].count(0) == 1
    assert [item[1] for item in trial.network.action_log].count(6) == 1
    assert len(trial.network.action_log) <= 6


def test_zone3_proposal_is_rejected_and_counted():
    class IllegalLLM:
        total_tokens = 0
        def complete_json(self, *args, **kwargs):
            return {'actions': [{'type': 'set_gen_voltage', 'gen_id': 2, 'vm_pu': 1.05}]}

    reset_tool_counters()
    trial = Case39Trial('two_layer', IllegalLLM())
    success, _ = trial.run()
    assert not success
    assert not trial.network.action_log
    assert get_tool_counters()['illegal_tool_calls'] > 0


def test_full_restart_keeps_cumulative_action_budget():
    class ClippedLLM:
        total_tokens = 0
        def complete_json(self, *args, **kwargs):
            return {'actions': [{'type': 'set_gen_voltage', 'gen_id': i, 'vm_pu': vm}
                                for i, vm in ((0, .98), (6, .98), (1, 1.03), (8, 1.06))]}

    row = run_once('two_layer_full_restart', 1, ClippedLLM())
    assert row['success'] == 0
    assert row['real_actions_used'] == 4
    assert row['wasted_actions'] == 4
