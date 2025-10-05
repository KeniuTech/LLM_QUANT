"""Test cases for LLM context management."""
import time

import pytest

from app.llm.context import Context, ContextConfig, ContextManager, DataAccessConfig, Message


def test_data_access_config():
    """Test data access configuration and validation."""
    config = DataAccessConfig(
        allowed_tables={"daily", "daily_basic"},
        max_history_days=365,
        max_batch_size=1000
    )

    # Valid request
    errors = config.validate_request("daily", "20251001", "20251005")
    assert not errors

    # Invalid table
    errors = config.validate_request("invalid", "20251001")
    assert len(errors) == 1
    assert "not allowed" in errors[0]

    # Invalid date format
    errors = config.validate_request("daily", "invalid")
    assert len(errors) == 1
    assert "Invalid date format" in errors[0]

    # Date range too long
    errors = config.validate_request("daily", "20251001", "20261001")
    assert len(errors) == 1
    assert "exceeds max" in errors[0]

    # End date before start
    errors = config.validate_request("daily", "20251005", "20251001")
    assert len(errors) == 1
    assert "before start date" in errors[0]


def test_context_config():
    """Test context configuration defaults."""
    config = ContextConfig()
    assert config.max_total_tokens > 0
    assert config.max_messages > 0
    assert config.include_system is True
    assert config.include_functions is True


def test_message():
    """Test message functionality."""
    # Basic message
    msg = Message(role="user", content="Hello")
    assert msg.role == "user"
    assert msg.content == "Hello"
    assert msg.name is None
    assert msg.function_call is None
    assert msg.timestamp <= time.time()

    # Function message
    func_msg = Message(
        role="function",
        content="Result",
        name="test_func",
        function_call={"name": "test_func", "arguments": "{}"}
    )
    assert func_msg.name == "test_func"
    assert func_msg.function_call is not None

    # Dict conversion
    msg_dict = msg.to_dict()
    assert msg_dict["role"] == "user"
    assert msg_dict["content"] == "Hello"
    assert "name" not in msg_dict
    assert "function_call" not in msg_dict

    func_dict = func_msg.to_dict()
    assert func_dict["name"] == "test_func"
    assert func_dict["function_call"] is not None

    # Token estimation
    assert msg.estimated_tokens > 0
    assert func_msg.estimated_tokens > msg.estimated_tokens


def test_context():
    """Test context management."""
    config = ContextConfig(max_total_tokens=100, max_messages=3)
    context = Context(config=config)

    # Add messages
    msg1 = Message(role="system", content="System message")
    msg2 = Message(role="user", content="User message")
    msg3 = Message(role="assistant", content="Assistant message")
    msg4 = Message(role="user", content="Another message")

    context.add_message(msg1)
    context.add_message(msg2)
    context.add_message(msg3)
    assert len(context.messages) == 3

    # Test max messages
    context.add_message(msg4)
    assert len(context.messages) == 3
    assert msg4 in context.messages  # Newest message kept

    # Get messages
    all_msgs = context.get_messages()
    assert len(all_msgs) == 3

    no_system = context.get_messages(include_system=False)
    assert len(no_system) == 2
    assert all(m["role"] != "system" for m in no_system)

    # Clear context
    context.clear(keep_system=True)
    assert len(context.messages) == 1
    assert context.messages[0].role == "system"

    context.clear(keep_system=False)
    assert len(context.messages) == 0


def test_context_manager():
    """Test context manager functionality."""
    ContextManager.clear_all()

    # Create context
    context = ContextManager.create_context("test")
    assert ContextManager.get_context("test") == context

    # Duplicate context
    with pytest.raises(ValueError):
        ContextManager.create_context("test")

    # Custom config
    config = ContextConfig(max_total_tokens=200)
    custom = ContextManager.create_context("custom", config)
    assert custom.config.max_total_tokens == 200

    # Remove context
    ContextManager.remove_context("test")
    assert ContextManager.get_context("test") is None

    # Clear all
    ContextManager.clear_all()
    assert ContextManager.get_context("custom") is None
