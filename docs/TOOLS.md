# Polyglot AI: Tool System Guide

This guide explains how to create, use, and extend the tool system in Polyglot AI. Tools are Python functions that the AI model can call to get information or perform actions, allowing it to interact with the local environment.

## What is a Tool?

A tool is a Python function decorated with the `@tool` decorator from `pai/tools.py`. This decorator does two things:

1.  **JSON Schema Generation:** It inspects the function's signature (parameter names, type hints) and docstring to automatically generate a JSON Schema. This schema is what's sent to the AI model, telling it what the function does, what parameters it expects, and what their types are.
2.  **Registration:** It registers the function in a global `TOOL_REGISTRY`, making it available for execution when the model requests it.

## How to Create a Tool

Creating a tool is simple. Just write a Python function with type hints and a clear docstring, then add the `@tool` decorator.

**Rules for creating effective tools:**
*   **Use Type Hints:** The decorator uses type hints to build the schema. Supported types are `str`, `int`, `float`, `bool`, and `enum.Enum`.
*   **Write a Good Docstring:** The first line of the docstring becomes the tool's `description` for the model. The `Args:` section is parsed to provide descriptions for each parameter. Be clear and direct so the model knows when to use your tool.
*   **Return JSON-serializable Data:** It's best practice for tools to return strings, especially JSON strings, as this is a format models handle well.

### Example: A Simple Calculator Tool

Let's say you want to create a tool in a new file `custom_tools/calculator.py`:

```python
# in custom_tools/calculator.py
import enum
from pai.tools import tool

class MathOperation(enum.Enum):
    """The supported mathematical operations."""
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    DIVIDE = "divide"

@tool
def calculate(a: float, b: float, operation: MathOperation) -> str:
    """Performs a basic arithmetic calculation.

    Args:
        a (float): The first number.
        b (float): The second number.
        operation (MathOperation): The operation to perform.
    """
    op = operation
    if op == MathOperation.ADD:
        result = a + b
    elif op == MathOperation.SUBTRACT:
        result = a - b
    elif op == MathOperation.MULTIPLY:
        result = a * b
    elif op == MathOperation.DIVIDE:
        if b == 0:
            return "Error: Cannot divide by zero."
        result = a / b
    else:
        return f"Error: Unknown operation {op}"

    return f"The result is {result}"
```
This example demonstrates a function with multiple arguments and an `Enum` type hint, which will be presented to the model as a list of valid string choices.

## Dynamic Tool Loading

You don't need to edit the core `pai` codebase to add new tools. You can place your tool files in a separate directory and tell Polyglot AI to load them at startup.

1.  **Create a Directory:** Create a directory for your custom tools (e.g., `custom_tools/`).
2.  **Add Tool Files:** Place your Python files containing `@tool`-decorated functions inside this directory.
3.  **Configure `polyglot.toml`:** Open your `polyglot.toml` file and add a `[tool_config]` section pointing to your directory.

    ```toml
    # In polyglot.toml

    [tool_config]
    # A list of directories to scan for custom tool modules.
    # Paths can be relative to your project's root or absolute.
    directories = ["custom_tools"]
    ```

When you next run `pai`, it will automatically find and load the `calculate` tool from `custom_tools/calculator.py`, making it available to the model. You'll see a confirmation message in the console at startup.
