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

## Prerequisites for Some Tools

Some of the advanced developer tools rely on popular, best-in-class command-line programs. Please install them using your system's package manager to enable the full toolset.

- **`fd`**: For `find_files`. A simple, fast, and user-friendly alternative to `find`.
- **`ripgrep` (`rg`)**: For `search_code`. A line-oriented search tool that respects your `.gitignore` and is very fast.

**Installation (macOS via Homebrew):**
```sh
brew install fd ripgrep
```

**Installation (Debian/Ubuntu via apt):**
```sh
sudo apt-get update && sudo apt-get install fd-find ripgrep
# On Debian/Ubuntu, fd may be installed as `fdfind`. A symlink `ln -s $(which fdfind) ~/.local/bin/fd` may be needed.
```

## Default Tool Showcase

Polyglot AI comes with a set of useful default tools located in the `custom_tools/` directory. These serve as powerful examples and can be used for productive tasks. They are loaded automatically if `directories = ["custom_tools"]` is present in your `polyglot.toml`.

### Data Conversion Tools (`data_converter.py`)

These tools are for converting between common structured data formats.

*   `json_to_yaml(json_string: str)`: Converts a JSON string to YAML.
*   `yaml_to_json(yaml_string: str)`: Converts a YAML string to JSON.
*   `csv_to_json(csv_data: str)`: Converts CSV data into a JSON array of objects.

**Example Usage:** `"take the following CSV data and convert it to JSON: \n\nid,name\n1,alice\n2,bob"`

### Code Generation Tools (`code_generator.py`)

Tools for generating code and other structured text.

*   `generate_python_class(class_name: str, fields: str)`: Generates a Python dataclass definition.

**Example Usage:** `"generate a python dataclass named 'Book' with fields: title: str, author: str, and published_year: int"`

### Filesystem Tools (`file_system.py`)

These tools allow the AI to interact with your local filesystem. For security, they are "jailed" to the project's current working directory by default.

*   `read_file(path: str)`: Reads the content of a file.
*   `list_directory(path: str = ".")`: Lists files and folders in a directory.
*   `write_file(path: str, content: str = "")`: Writes content to a file, creating it if needed or overwriting it.
*   `append_to_file(path: str, content: str)`: Appends text to an existing file.
*   `delete_file(path: str)`: Deletes a single file.
*   `delete_directory(path: str)`: Recursively deletes a directory and all its contents. Use with caution.

**Example Usage:** `"write a file named 'hello.txt' with 'hello world', then read it, then delete it."`

### Developer Tools (`developer_tools.py`)

A set of powerful tools for code navigation and exploration, which depend on `fd` and `ripgrep`. If these are not installed, the tools will return an error.

*   `find_files(pattern: str, search_path: str = ".")`: Finds files by name using a glob pattern. Returns a JSON list of paths.
*   `search_code(pattern: str, search_path: str = ".")`: Searches for a regex pattern within files and returns a structured JSON output of matches.
*   `read_file_lines(path: str, start_line: int, end_line: int)`: Reads a specific range of lines from a file, which is useful for inspecting the context around a `search_code` result.

**Example Usage:** `"find all python files, then search for 'ProtocolContext' in the 'pai' directory, then show me lines 1-10 of 'pai/protocols/base_adapter.py'"`

### HTTP Client Tools (`http_client.py`)

A structured tool for making HTTP requests to web APIs. This is a safer and more powerful alternative to a simple URL fetcher.

*   `make_http_request(url: str, method: HttpMethod, ...)`: Makes an HTTP request and returns a JSON object with the status code, headers, and body.

**Example Usage:** `"make a GET request to https://api.github.com/zen to get a random github zen quote"`

### Code Editing Tools (`code_editor.py`)

This powerful tool allows the AI to edit files by submitting a script of changes. It is the foundation for agentic software development tasks.

*   `apply_search_replace(edit_script: str)`: Parses and applies one or more `SEARCH/REPLACE` blocks.

**`SEARCH/REPLACE` Block Rules:**
The `edit_script` must contain one or more blocks with the following strict format:

´´´´[language]
path/to/file.ext
<<<<<<< SEARCH
content_to_find
