import enum

import pytest

from pai import tools


@pytest.fixture(autouse=True)
def managed_tool_registry():
    """
    Fixture to ensure the tool registry is clean for each test and restored afterwards.
    """
    original_registry = tools.TOOL_REGISTRY.copy()
    tools.TOOL_REGISTRY.clear()
    yield
    tools.TOOL_REGISTRY.clear()
    tools.TOOL_REGISTRY.update(original_registry)


class SampleEnum(enum.Enum):
    A = "a"
    B = "b"


@tools.tool
def sample_tool(name: str, value: int = 10, option: SampleEnum = SampleEnum.A) -> dict:
    """A sample tool for testing.

    Args:
        name (str): The name parameter.
        value (int): The value parameter.
        option (SampleEnum): An enum choice.
    """
    return {"name": name, "value": value, "option": option.value}


def test_tool_decorator_registers_and_creates_schema():
    """Verify that the @tool decorator correctly registers the tool and builds its schema."""
    assert "sample_tool" in tools.TOOL_REGISTRY
    schema = tools.get_tool_schemas()[0]

    expected_schema = {
        "type": "function",
        "function": {
            "name": "sample_tool",
            "description": "A sample tool for testing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "The name parameter."},
                    "value": {"type": "integer", "description": "The value parameter."},
                    "option": {
                        "type": "string",
                        "description": "An enum choice.",
                        "enum": ["a", "b"],
                    },
                },
                "required": ["name"],
            },
        },
    }
    assert schema == expected_schema


def test_execute_tool_success_with_defaults():
    """Test successful execution of a tool using default values."""
    result = tools.execute_tool("sample_tool", {"name": "test"})
    assert result == {"name": "test", "value": 10, "option": "a"}


def test_execute_tool_success_with_all_args():
    """Test successful execution with all arguments provided, including an enum."""
    result = tools.execute_tool(
        "sample_tool", {"name": "full", "value": 99, "option": "b"}
    )
    assert result == {"name": "full", "value": 99, "option": "b"}


def test_execute_tool_not_found():
    """Test that executing a non-existent tool returns a specific error."""
    result = tools.execute_tool("non_existent_tool", {})
    assert "Error: Tool 'non_existent_tool' not found." in result


def test_execute_tool_invalid_enum_value():
    """Test that executing a tool with an invalid enum value returns a ValueError."""
    result = tools.execute_tool("sample_tool", {"name": "test", "option": "c"})
    assert "Error: Invalid argument value provided" in result
    assert "'c' is not a valid SampleEnum" in str(result)


def test_get_tool_manifest():
    """Verify that the text-based tool manifest is generated correctly."""
    manifest = tools.get_tool_manifest()
    assert "- Name: sample_tool" in manifest
    assert "Description: A sample tool for testing." in manifest
    assert "- name (string): The name parameter." in manifest
    assert "- value (integer): The value parameter." in manifest
    assert "- option (string (enum: a, b)): An enum choice." in manifest
