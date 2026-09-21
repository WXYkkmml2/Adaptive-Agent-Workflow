import ssl
import pytest
from llm.ssl_utils import get_ssl_context
from llm.client import LLMServiceUnavailable
from run_case39 import run_once


def test_system_ca_context_keeps_verification(monkeypatch):
    monkeypatch.setenv('SSL_CERT_FILE', '/etc/ssl/cert.pem')
    context = get_ssl_context()
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname


def test_retryable_network_failure_does_not_count_as_trial_failure():
    class FailedLLM:
        total_tokens = 0
        def complete_json(self, *args, **kwargs):
            return {'error': 'LLM_ERROR', 'message': 'TLS failure', 'retryable': True}

    with pytest.raises(LLMServiceUnavailable, match='TLS failure'):
        run_once('two_layer', 1, FailedLLM())
