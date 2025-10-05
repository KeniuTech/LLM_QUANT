"""Test cases for LLM client."""
import pytest
import responses

from app.llm.client import LLMError, call_endpoint_with_messages
from app.utils.config import LLMEndpoint


@responses.activate
def test_openai_chat():
    """Test OpenAI chat completion."""
    # Mock successful response
    responses.add(
        responses.POST,
        "https://api.openai.com/v1/chat/completions",
        json={
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Test response"
                }
            }]
        },
        status=200
    )

    endpoint = LLMEndpoint(
        provider="openai",
        model="gpt-3.5-turbo",
        api_key="test_key"
    )

    messages = [{"role": "user", "content": "Test prompt"}]
    response = call_endpoint_with_messages(endpoint, messages)
    assert response["choices"][0]["message"]["content"] == "Test response"


@responses.activate
def test_ollama_chat():
    """Test Ollama chat completion."""
    # Mock successful response
    responses.add(
        responses.POST,
        "http://localhost:11434/api/chat",
        json={
            "message": {
                "role": "assistant",
                "content": "Test response"
            }
        },
        status=200
    )

    endpoint = LLMEndpoint(
        provider="ollama",
        model="llama2"
    )

    messages = [{"role": "user", "content": "Test prompt"}]
    response = call_endpoint_with_messages(endpoint, messages)
    assert response["message"]["content"] == "Test response"


@responses.activate
def test_error_handling():
    """Test error handling."""
    # Mock error response
    responses.add(
        responses.POST,
        "https://api.openai.com/v1/chat/completions",
        json={"error": "Test error"},
        status=400
    )

    endpoint = LLMEndpoint(
        provider="openai",
        model="gpt-3.5-turbo",
        api_key="test_key"
    )

    messages = [{"role": "user", "content": "Test prompt"}]
    with pytest.raises(LLMError):
        call_endpoint_with_messages(endpoint, messages)


def test_endpoint_resolution():
    """Test endpoint configuration resolution."""
    # Default Ollama endpoint
    endpoint = LLMEndpoint(provider="ollama")
    assert endpoint.model == "llama2"  # Default model
    assert endpoint.temperature == 0.2  # Default temperature

    # Custom OpenAI endpoint
    endpoint = LLMEndpoint(
        provider="openai",
        model="gpt-4",
        temperature=0.5,
        timeout=60
    )
    assert endpoint.model == "gpt-4"
    assert endpoint.temperature == 0.5
    assert endpoint.timeout == 60

    # Invalid temperature
    endpoint = LLMEndpoint(temperature=3.0)
    assert endpoint.temperature == 2.0  # Clamped to max

    # Invalid timeout
    endpoint = LLMEndpoint(timeout=1.0)
    assert endpoint.timeout == 5.0  # Clamped to min
