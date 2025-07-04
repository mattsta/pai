import enum

import pytest

from pai import tools


@pytest.fixture(autouse=True)
def managed_tool_registry():
    """
    Fixture to ensure the tool registry is clean for each test and restored afterwards.
    It runs after the module is loaded (and sample_tool is registered), and resets
    the registry state after each test to ensure isolation.
    """
    original_registry = tools.TOOL_REGISTRY.copy()
    yield
    # Reset registry to its post-import state after each test.
    # This removes any tools that might have been registered *during* a test.
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
    # To make the test robust against dict ordering, find the specific schema.
    schemas = tools.get_tool_schemas()
    schema = next((s for s in schemas if s["function"]["name"] == "sample_tool"), None)
    assert schema is not None

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


async def test_execute_tool_success_with_defaults():
    """Test successful execution of a tool using default values."""
    result = await tools.execute_tool("sample_tool", {"name": "test"})
    assert result == {"name": "test", "value": 10, "option": "a"}


async def test_execute_tool_success_with_all_args():
    """Test successful execution with all arguments provided, including an enum."""
    result = await tools.execute_tool(
        "sample_tool", {"name": "full", "value": 99, "option": "b"}
    )
    assert result == {"name": "full", "value": 99, "option": "b"}


async def test_execute_tool_not_found():
    """Test that executing a non-existent tool raises ToolNotFound."""
    with pytest.raises(tools.ToolNotFound, match="Tool 'non_existent_tool' not found"):
        await tools.execute_tool("non_existent_tool", {})


async def test_execute_tool_invalid_enum_value():
    """Test that an invalid enum value raises ToolArgumentError."""
    with pytest.raises(tools.ToolArgumentError, match="'c' is not a valid SampleEnum"):
        await tools.execute_tool("sample_tool", {"name": "test", "option": "c"})


async def test_execute_tool_missing_required_arg():
    """Test that a missing required argument raises ToolArgumentError."""
    with pytest.raises(tools.ToolArgumentError, match="Missing or invalid arguments"):
        # The 'name' argument is required and has no default value.
        await tools.execute_tool("sample_tool", {})


def test_get_tool_manifest():
    """Verify that the text-based tool manifest is generated correctly."""
    manifest = tools.get_tool_manifest()
    assert "- Name: sample_tool" in manifest
    assert "Description: A sample tool for testing." in manifest
    assert "- name (string): The name parameter." in manifest
    assert "- value (integer): The value parameter." in manifest
    assert "- option (string (enum: a, b)): An enum choice." in manifest
