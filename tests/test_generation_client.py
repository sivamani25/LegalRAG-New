import httpx
import pytest

from legalrag.generation.client import OpenRouterClient, OpenRouterError


CONFIG = {
    "generation": {
        "provider": "openrouter",
        "model": "cohere/north-mini-code:free",
        "fallback_model": "poolside/laguna-xs-2.1:free",
        "temperature": 0.2,
        "max_tokens": 1024,
    }
}


class FakeHTTP:
    def __init__(self, responses):
        self.responses = list(responses)
        self.payloads = []

    def post(self, url, *, headers, json):
        self.payloads.append(json)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def response(status, body):
    request = httpx.Request(
        "POST", "https://openrouter.ai/api/v1/chat/completions"
    )
    return httpx.Response(status, json=body, request=request)


def success(text):
    return response(200, {
        "choices": [{"message": {"content": text}}]
    })


def test_primary_model_success():
    fake = FakeHTTP([success("Grounded answer.")])
    client = OpenRouterClient(CONFIG, api_key="test-key", http_client=fake)

    answer = client.generate([{"role": "user", "content": "Test question"}])

    assert answer == "Grounded answer."
    assert fake.payloads[0]["model"] == "cohere/north-mini-code:free"
    assert len(fake.payloads) == 1


def test_fallback_used_when_primary_fails():
    fake = FakeHTTP([
        response(503, {"error": {"message": "Unavailable"}}),
        success("Fallback answer."),
    ])
    client = OpenRouterClient(CONFIG, api_key="test-key", http_client=fake)

    answer = client.generate([{"role": "user", "content": "Test question"}])

    assert answer == "Fallback answer."
    assert fake.payloads[0]["model"] == "cohere/north-mini-code:free"
    assert fake.payloads[1]["model"] == "poolside/laguna-xs-2.1:free"


def test_missing_api_key_fails_without_request():
    fake = FakeHTTP([])
    client = OpenRouterClient(CONFIG, api_key="", http_client=fake)

    with pytest.raises(OpenRouterError, match="OPENROUTER_API_KEY"):
        client.generate([{"role": "user", "content": "Test question"}])

    assert fake.payloads == []


def test_both_models_failing_raises_error():
    fake = FakeHTTP([
        response(503, {"error": {"message": "Primary unavailable"}}),
        response(503, {"error": {"message": "Fallback unavailable"}}),
    ])
    client = OpenRouterClient(CONFIG, api_key="test-key", http_client=fake)

    with pytest.raises(OpenRouterError, match="all configured models"):
        client.generate([{"role": "user", "content": "Test question"}])

    assert len(fake.payloads) == 2


def test_empty_primary_content_uses_fallback():
    fake = FakeHTTP([
        response(200, {
            "choices": [{
                "finish_reason": "length",
                "message": {"content": None},
            }]
        }),
        success("Valid fallback answer."),
    ])
    client = OpenRouterClient(CONFIG, api_key="test-key", http_client=fake)

    answer = client.generate([{"role": "user", "content": "Test question"}])

    assert answer == "Valid fallback answer."
    assert len(fake.payloads) == 2
    assert fake.payloads[0]["model"] == "cohere/north-mini-code:free"
    assert fake.payloads[1]["model"] == "poolside/laguna-xs-2.1:free"


def test_empty_content_from_both_models_raises_error():
    empty = response(200, {
        "choices": [{"finish_reason": "length", "message": {"content": None}}]
    })
    fake = FakeHTTP([empty, empty])
    client = OpenRouterClient(CONFIG, api_key="test-key", http_client=fake)

    with pytest.raises(OpenRouterError, match="all configured models"):
        client.generate([{"role": "user", "content": "Test question"}])

    assert len(fake.payloads) == 2

    
def test_empty_response_reports_finish_reason():
    empty = response(200, {
        "choices": [{
            "finish_reason": "length",
            "message": {"content": None},
        }]
    })
    fake = FakeHTTP([empty, empty])
    client = OpenRouterClient(CONFIG, api_key="test-key", http_client=fake)

    with pytest.raises(OpenRouterError) as exc_info:
        client.generate([{"role": "user", "content": "Test question"}])

    assert "finish_reason=length" in str(exc_info.value)
    assert len(fake.payloads) == 2


def test_reasoning_settings_are_sent():
    fake = FakeHTTP([success("Grounded answer.")])
    client = OpenRouterClient(CONFIG, api_key="test-key", http_client=fake)

    client.generate([{"role": "user", "content": "Test question"}])

    payload = fake.payloads[0]
    assert payload["max_completion_tokens"] == 1024
    assert payload["reasoning"] == {
        "max_tokens": 512,
        "exclude": True,
    }

