import copy
import csv
import inspect
import json
from unittest.mock import patch

import pytest
from grid.scenario_case39 import make_network, validate_scenario
from grid.goal import Goal, goal_status
from run_case39 import Case39Trial, INSTRUCTION, run_once, METHODS, main, method_order
from grid.tools import get_tool_counters, reset_tool_counters


class ScriptedLLM:
    """Offline protocol peer; all dispatch choices originate here, never in production."""
    model = 'offline-test'
    base_url = 'https://offline.invalid'

    def __init__(self, trap=False, dependencies=False, single=False):
        self.total_tokens = self.llm_calls = 0
        self.tokens_by_source = {}
        self.prompts = []
        self.trap, self.dependencies, self.single = trap, dependencies, single

    def check_connection(self):
        pass

    def complete_json(self, system_prompt, user_prompt, **kwargs):
        self.llm_calls += 1
        self.total_tokens += 10
        source = kwargs['source']
        bucket = self.tokens_by_source.setdefault(source, {'total_tokens': 0})
        bucket['total_tokens'] += 10
        context = json.loads(user_prompt)
        self.prompts.append((system_prompt, context, kwargs))
        if source == 'planner':
            if self.single:
                return {'tasks': [{'id': 'whole', 'description': '恢复所有区域', 'devices': [29, 36, 7],
                                   'device_type': 'bus', 'dependencies': []}]}
            return {'tasks': [
                {'id': 'region2', 'description': '恢复区域2', 'devices': [29, 36], 'device_type': 'bus', 'dependencies': []},
                {'id': 'region1', 'description': '恢复区域1', 'devices': [7], 'device_type': 'bus',
                 'dependencies': ['region2'] if self.dependencies else []}]}
        task = context['instruction']['task']
        region = 2 if task == '恢复区域2' else 1 if task == '恢复区域1' else None
        actions = [(0, .98), (6, .98)] if region == 2 else [(1, 1.06)] if region == 1 else [(0, .98), (6, .98), (1, 1.06)]
        if self.trap and not context['feedback']:
            actions = ([(0, .98), (6, .98)] if region != 1 else []) + [(1, 1.03), (8, 1.06)] if region != 2 else actions
        return {'actions': [{'type': 'set_gen_voltage', 'gen_id': i, 'vm_pu': vm} for i, vm in actions]}


def test_fixture_and_budget_are_offline():
    assert validate_scenario()
    net = make_network()
    goal = Goal.from_instruction(INSTRUCTION)
    assert len(net.action_log) == 0
    assert goal.max_real_actions == 6
    assert goal.forbidden_regions == frozenset({3})
    assert goal_status(net.net, goal)['violation_count'] == 9


def test_three_methods_share_successful_tool_path():
    for method in METHODS:
        row = run_once(method, 1, ScriptedLLM())
        assert row['success'] == 1, (method, row['error'])
        assert row['real_actions_used'] == 3
        assert row['zone3_touched'] == 0
        assert row['tree_depth'] == (3 if method == 'hierarchical' else 2)
        assert row['total_tokens'] == row['llm_calls'] * 10
        assert sum(v['total_tokens'] for v in json.loads(row['tokens_by_source']).values()) == row['total_tokens']


def test_hidden_limit_replans_only_region1_after_region2_commit():
    llm = ScriptedLLM(trap=True)
    trial = Case39Trial('hierarchical', llm)
    success, error = trial.run()
    assert success, error
    assert trial.replanned == ['region1']
    assert [item[1] for item in trial.network.action_log].count(0) == 1
    assert [item[1] for item in trial.network.action_log].count(6) == 1
    leaves = [c for _, c, k in llm.prompts if k['source'] != 'planner']
    assert [c['instruction']['task'] for c in leaves] == ['恢复区域2', '恢复区域1', '恢复区域1']
    assert len(trial.network.action_log) <= 6
    assert len(trial.clipped_actions) == 1
    assert trial.first_attempt_deviation
    assert trial.attempts == 2


def test_zone3_proposal_is_rejected_and_counted():
    class IllegalLLM(ScriptedLLM):
        def complete_json(self, *args, **kwargs):
            return {'actions': [{'type': 'set_gen_voltage', 'gen_id': 2, 'vm_pu': 1.05}]}
    reset_tool_counters()
    trial = Case39Trial('two_layer', IllegalLLM())
    success, _ = trial.run()
    assert not success
    assert not trial.network.action_log
    assert get_tool_counters()['illegal_tool_calls'] == 3
    assert trial.scope_hit == 3


def test_full_restart_keeps_cumulative_action_budget():
    class ClippedLLM(ScriptedLLM):
        def complete_json(self, *args, **kwargs):
            return {'actions': [{'type': 'set_gen_voltage', 'gen_id': i, 'vm_pu': vm}
                                for i, vm in ((0, .98), (6, .98), (1, 1.03), (8, 1.06))]}
    row = run_once('two_layer_full_restart', 1, ClippedLLM())
    assert row['success'] == 0
    assert row['real_actions_used'] == 4
    assert row['wasted_actions'] == 4


@pytest.mark.parametrize('method', METHODS)
def test_real_class_chain_and_common_kernel(method, monkeypatch):
    from agents.planner import Planner
    from agents.root_agent import RootAgent
    from agents.orchestration_agent import OrchestrationAgent
    from agents.execution_agent import ExecutionAgent
    counts = {}
    for cls in (Planner, RootAgent, OrchestrationAgent, ExecutionAgent):
        original = cls.__init__
        def wrapped(self, *args, _original=original, _name=cls.__name__, **kwargs):
            counts[_name] = counts.get(_name, 0) + 1
            _original(self, *args, **kwargs)
        monkeypatch.setattr(cls, '__init__', wrapped)
    import grid.tools as tools
    from grid.network import PowerNetwork
    import grid.dispatch_kernel as kernel_module
    callers = {'simulation': [], 'real': [], 'goal': []}
    for owner, name, key in ((tools, 'simulate_action', 'simulation'), (PowerNetwork, 'set_gen_voltage', 'real'),
                             (kernel_module, 'goal_status', 'goal')):
        original = getattr(owner, name)
        def spy(*args, _original=original, _key=key, **kwargs):
            callers[_key].append(inspect.currentframe().f_back.f_globals['__name__'])
            return _original(*args, **kwargs)
        monkeypatch.setattr(owner, name, spy)
    trial = Case39Trial(method, ScriptedLLM())
    assert trial.run()[0]
    assert bool(counts.get('Planner')) == (method == 'hierarchical')
    assert bool(counts.get('RootAgent')) == (method == 'hierarchical')
    assert counts['OrchestrationAgent'] and counts['ExecutionAgent'] == 3
    for values in callers.values():
        assert values and set(values) == {'grid.dispatch_kernel'}


def test_hidden_limit_preview_and_prompt_isolation():
    from grid.tools import simulate_action
    net = make_network()
    actions = [{'type': 'set_gen_voltage', 'gen_id': i, 'vm_pu': vm}
               for i, vm in ((0, .98), (6, .98), (1, 1.03), (8, 1.06))]
    assert simulate_action(net.net, actions, Goal.from_instruction(INSTRUCTION))['goal_met']
    for action in actions:
        net.set_gen_voltage(action['gen_id'], action['vm_pu'])
    assert net.net.gen.at[8, 'vm_pu'] == 1.03
    assert not goal_status(net.net, Goal.from_instruction(INSTRUCTION))['goal_met']
    llm = ScriptedLLM()
    assert Case39Trial('hierarchical', llm).run()[0]
    prompts = json.dumps(llm.prompts, ensure_ascii=False)
    for private in ('vm_max', 'device_limits', 'witness', '见证', '1.03'):
        assert private not in prompts
    # Changing private limits cannot change any initial LLM-visible catalog.
    from agents.permission import Permission
    a = Case39Trial('two_layer', ScriptedLLM())
    initial_catalog = a.catalog(Permission.root_permission())
    a.network.device_limits = {8: {'vm_max': .99}}
    assert a.catalog(Permission.root_permission()) == initial_catalog


@pytest.mark.parametrize('method,expected_vm', [('two_layer', 'within_goal_range'), ('two_layer_full_restart', .92)])
def test_restart_state_and_budget_observations(method, expected_vm):
    llm = ScriptedLLM(trap=True)
    trial = Case39Trial(method, llm)
    trial.run()
    second = llm.prompts[1][1]
    assert second['used_actions'] == 4
    assert second['budget'] == 6
    assert next(g['vm_pu'] for g in second['catalog']['generators'] if g['gen_id'] == 0) == expected_vm


def test_catalog_scope_and_identical_leaf_fields():
    contexts = []
    for method in METHODS:
        llm = ScriptedLLM()
        trial = Case39Trial(method, llm)
        assert trial.run()[0]
        leaves = [(s, c, k) for s, c, k in llm.prompts if k['source'] != 'planner']
        for system, context, kwargs in leaves:
            gens = context['catalog']['generators']
            forbidden = [g for g in gens if int(trial.network.net.bus.at[g['bus_id'], 'zone']) == 3]
            assert bool(forbidden) == (method != 'hierarchical')
            assert set(context) == {'instruction', 'current_goal', 'used_actions', 'budget', 'feedback', 'catalog'}
            contexts.append((system, kwargs))
    assert all(c == contexts[0] for c in contexts)


def test_region_targets_and_dependency_partition():
    from agents.planner import Planner
    from agents.regional import partition_dag
    network = make_network()
    planner = Planner(network, ScriptedLLM(dependencies=True))
    targets = planner._parse_target(INSTRUCTION)
    assert set(targets) == {1, 2}
    assert targets != 13
    plan = planner.plan(INSTRUCTION)
    assert len(partition_dag(plan['dag'], network.net)) == 1


def test_invalid_dag_is_not_replaced():
    class NoDag(ScriptedLLM):
        def complete_json(self, *args, **kwargs):
            return {'actions': []}
    row = run_once('hierarchical', 1, NoDag())
    assert not row['success']
    assert row['dag_task_count'] == 0
    assert row['real_actions_used'] == 0
    assert 'DAG' in row['error']


def test_shared_client_usage_is_per_trial():
    client = ScriptedLLM()
    first = run_once('two_layer', 1, client)
    second = run_once('two_layer', 2, client)
    assert first['total_tokens'] == second['total_tokens'] == 10
    assert first['llm_calls'] == second['llm_calls'] == 1


def test_protocol_resume_rotation_and_infrastructure(tmp_path, monkeypatch):
    import run_case39
    from llm.client import LLMServiceUnavailable
    monkeypatch.setattr(run_case39, 'RealLLMClient', lambda **kwargs: ScriptedLLM())
    original = run_case39.run_once
    def interrupted(method, *args, **kwargs):
        if method == 'two_layer':
            raise LLMServiceUnavailable('HTTP 429')
        return original(method, *args, **kwargs)
    monkeypatch.setattr(run_case39, 'run_once', interrupted)
    output = tmp_path / 'smoke.csv'
    args = ['--smoke', '--output', str(output)]
    assert main(args) == 2
    with output.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1 and rows[0]['formal'] == '0'
    with pytest.raises(SystemExit):
        main(args)
    monkeypatch.setattr(run_case39, 'run_once', original)
    assert main(args + ['--resume']) == 0
    with output.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 3
    assert method_order(2) == ('two_layer', 'two_layer_full_restart', 'hierarchical')
    assert method_order(3) == ('two_layer_full_restart', 'hierarchical', 'two_layer')
    with pytest.raises(SystemExit):
        main(args + ['--resume', '--max-actions', '7'])


def test_wilson_and_trap_subset():
    from summarize_case39 import summarize, wilson
    assert wilson(0, 0) is None
    assert wilson(10, 10)[0] == pytest.approx(.7224672)
    row = run_once('two_layer', 1, ScriptedLLM())
    results = summarize([row])
    assert results[0]['n'] == 1 and results[0]['success_rate'] == 1
    assert results[1]['n'] == 0 and results[1]['success_rate'] is None


def test_dependency_failure_waits_and_rebuilds_only_failed_path():
    from agents.task import TaskStatus
    llm = ScriptedLLM(trap=True, dependencies=True)
    trial = Case39Trial('hierarchical', llm)
    assert trial.run()[0]
    assert trial.agent.partitions == [['region2', 'region1']]
    assert trial.replanned == ['region1']
    assert all(t.status == TaskStatus.COMPLETED for t in trial.agent.dag.tasks.values())
    # Reverse dependency: failed predecessor blocks the region2 physical actions.
    class Reverse(ScriptedLLM):
        def complete_json(self, system_prompt, user_prompt, **kwargs):
            response = super().complete_json(system_prompt, user_prompt, **kwargs)
            if kwargs['source'] == 'planner':
                response['tasks'][0]['dependencies'] = ['region1']
                response['tasks'][1]['dependencies'] = []
            return response
    trial = Case39Trial('hierarchical', Reverse(trap=True))
    assert trial.run()[0]
    assert trial.replanned == ['region1']
    ids = [item[1] for item in trial.network.action_log]
    assert ids[:2] == [1, 8]
    assert ids.count(0) == ids.count(6) == 1


def test_ablation_switches_and_depth_are_used(monkeypatch):
    from agents.orchestration_agent import OrchestrationAgent
    from agents import regional
    seen = []
    original = OrchestrationAgent.__init__
    def spy(self, *args, **kwargs):
        original(self, *args, **kwargs)
        seen.append((self.current_depth, self.max_depth))
    monkeypatch.setattr(OrchestrationAgent, '__init__', spy)
    monkeypatch.setattr('grid.topology.regional_tree_depth', lambda scope, net: 4)
    row = run_once('hierarchical_no_shrink', 1, ScriptedLLM())
    assert row['success'] and row['tree_depth'] == 4
    assert (1, 4) in seen and (2, 4) in seen
    assert not json.loads(row['method_config'])['permission_shrink']
    row = run_once('hierarchical_full_restart', 1, ScriptedLLM(trap=True))
    assert json.loads(row['method_config'])['replan_mode'] == 'full'
    assert row['real_actions_used'] == row['wasted_actions'] == 4


def test_trap_prompts_never_include_private_number_or_witness():
    for method in METHODS:
        client = ScriptedLLM(trap=True)
        Case39Trial(method, client).run()
        prompts = json.dumps(client.prompts, ensure_ascii=False)
        assert '1.03' not in prompts
        assert 'vm_max' not in prompts and 'witness' not in prompts


def test_common_gate_compares_adjacent_prefixes(monkeypatch):
    from grid.dispatch_kernel import DispatchKernel
    from agents.permission import Permission
    import grid.tools as tools
    trial = Case39Trial('two_layer', ScriptedLLM())
    statuses = iter([
        {'violations': [{'type': 'voltage_low', 'bus_id': 0, 'value': .90}]},
        {'violations': [{'type': 'voltage_low', 'bus_id': 0, 'value': .94}]},
        {'violations': [{'type': 'voltage_low', 'bus_id': 0, 'value': .92}]},
    ])
    monkeypatch.setattr(trial.kernel, 'status', lambda *args: next(statuses))
    monkeypatch.setattr(tools, 'simulate_action', lambda *args: {'success': True, 'net_copy': object(), 'goal_met': True})
    actions = [{'type': 'set_gen_voltage', 'gen_id': 0, 'vm_pu': .98}] * 2
    assert not trial.joint_gate(actions)['goal_met']


@pytest.mark.parametrize('code', [429, 500, 501, 503, 599])
def test_fixed_client_service_errors_and_request_limits(code, monkeypatch):
    import io
    import urllib.error
    from llm.client import RealLLMClient, LLMServiceUnavailable
    monkeypatch.setenv('LLM_API_KEY', 'secret-offline')
    monkeypatch.setenv('LLM_BASE_URL', 'https://offline.invalid')
    def failed(request, **kwargs):
        raise urllib.error.HTTPError(request.full_url, code, 'unavailable', {}, io.BytesIO(b'secret-offline'))
    monkeypatch.setattr('urllib.request.urlopen', failed)
    client = RealLLMClient(fixed_limits=True)
    with pytest.raises(LLMServiceUnavailable):
        client.check_connection()
    response = client.complete_json('system', 'user', max_tokens=2048, source='planner')
    assert response['retryable']
    assert client.llm_calls == 1
    assert 'secret-offline' not in response['message']


def test_fixed_client_never_expands_token_budget(monkeypatch):
    from tests.test_llm_client import FakeResponse
    from llm.client import RealLLMClient
    monkeypatch.setenv('LLM_API_KEY', 'offline')
    monkeypatch.setenv('LLM_BASE_URL', 'https://offline.invalid')
    payloads = []
    def truncated(request, **kwargs):
        payloads.append(json.loads(request.data))
        return FakeResponse({'choices': [{'message': {'content': ''}, 'finish_reason': 'length'}],
                             'usage': {'prompt_tokens': 5, 'completion_tokens': 2048, 'total_tokens': 2053}})
    monkeypatch.setattr('urllib.request.urlopen', truncated)
    client = RealLLMClient(fixed_limits=True)
    result = client.complete_json('system', 'user', max_tokens=2048, source='planner')
    assert result['error'] == 'LLM_ERROR' and not result['retryable']
    assert len(payloads) == 1 and payloads[0]['max_tokens'] == 2048
    assert client.tokens_by_source['planner']['total_tokens'] == 2053


def test_network_error_is_infrastructure(monkeypatch):
    from llm.client import RealLLMClient, LLMServiceUnavailable
    monkeypatch.setenv('LLM_API_KEY', 'offline')
    monkeypatch.setenv('LLM_BASE_URL', 'https://offline.invalid')
    def reset(*args, **kwargs):
        raise ConnectionResetError('peer reset')
    monkeypatch.setattr('urllib.request.urlopen', reset)
    client = RealLLMClient(fixed_limits=True)
    with pytest.raises(LLMServiceUnavailable):
        client.check_connection()
    assert client.complete_json('', '')['retryable']


def test_typed_devices_define_region_not_index_coincidence():
    from agents.permission import Permission
    from agents.task import Task
    net = make_network().net
    assert Permission.task_buses(Task('g', '恢复发电机', devices=[0], device_type='gen'), net) == {29}
    assert Permission.from_task(Task('g', '恢复发电机', devices=[0], device_type='gen'), net).regions == {'2'}


def test_duplicate_metric_counts_calls_only_after_retry():
    from agents.permission import Permission
    trial = Case39Trial('two_layer', ScriptedLLM())
    action = {'type': 'set_gen_voltage', 'gen_id': 0, 'vm_pu': .98}
    trial.kernel.attempts = 1
    trial.kernel._record_call('set_gen_voltage', action)
    trial.kernel._record_call('set_gen_voltage', action)
    assert trial.duplicates == 0
    trial.kernel.attempts = 2
    trial.kernel._record_call('set_gen_voltage', dict(reversed(list(action.items()))))
    assert trial.duplicates == 1
    trial.kernel._record_call('simulate_action', {'action': [action]})
    assert trial.duplicates == 1
