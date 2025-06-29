# Polyglot AI: Tool System Guide

This guide explains how to **create and extend** the tool system in Polyglot AI. For a practical guide on **how to use tools** in a chat session, see the [Tool Usage Tutorial](./TOOL_TUTORIAL.md).

Tools are Python functions that the AI model can call to get information or perform actions, allowing it to interact with the local environment.

## What is a Tool?

A tool is a Python function decorated with the `@tool` decorator from `pai/tools.py`. This decorator does two things:

1.  **JSON Schema Generation:** It inspects the function's signature (parameter names, type hints) and docstring to automatically generate a JSON Schema. This schema is what's sent to the AI model, telling it what the function does, what parameters it expects, and what their types are.
2.  **Registration:** It registers the function in a global `TOOL_REGISTRY`, making it available for execution when the model requests it.

## How to Create a Tool

Creating a tool is simple. Just write a Python function with type hints and a clear docstring, then add the `@tool` decorator.

**Rules for creating effective tools:**
*   **Use Type Hints:** The decorator uses type hints to build the schema. Supported types are `str`, `int`, `float`, `bool`, and `enum.Enum`.
*   **Write a Good Docstring:** The first line of the docstring becomes the tool's `description` for the model. The `Args:` section is parsed to provide descriptions for each parameter. Be clear and direct so the model knows when to use your tool.
*   **Return Standardized JSON:** All tools **must** return a JSON string with a `status` field. This provides a reliable structure for the AI to parse.
    - **On success:** `{"status": "success", "result": ...}`
    - **On failure:** `{"status": "failure", "reason": "..."}`

### Example: A Simple Calculator Tool

Let's say you want to create a tool in a new file `custom_tools/calculator.py`:

```python
# in custom_tools/calculator.py
import enum
import json
from pai.tools import tool

class MathOperation(enum.Enum):
    ADD = "add"
    SUBTRACT = "subtract"

@tool
def calculate(a: float, b: float, operation: MathOperation) -> str:
    """Performs a basic arithmetic calculation.

    Args:
        a (float): The first number.
        b (float): The second number.
        operation (MathOperation): The operation to perform.
    """
    result = {}
    try:
        if operation == MathOperation.ADD:
            calc_result = a + b
        elif operation == MathOperation.SUBTRACT:
            calc_result = a - b
        else:
            raise ValueError(f"Unknown operation {operation}")
        result = {"status": "success", "result": f"The result is {calc_result}"}
    except Exception as e:
        result = {"status": "failure", "reason": str(e)}
    
    return json.dumps(result, indent=2)
```
This example demonstrates a function that returns the required standardized JSON output.

## Dynamic Tool Loading

You don't need to edit the core `pai` codebase to add new tools. You can place your tool files in a separate directory and tell Polyglot AI to load them at startup.

1.  **Create a Directory:** Create a directory for your custom tools (e.g., `my_tools/`). It must contain an `__init__.py` file to be recognized as a Python package.
2.  **Add Tool Files:** Place your Python files containing `@tool`-decorated functions inside this directory.
3.  **Configure `pai.toml`:** Open your `pai.toml` file and add a `[tool_config]` section pointing to your directory.

    ```toml
    # In pai.toml

    [tool_config]
    # A list of directories to scan for custom tool modules.
    # Paths can be relative to your project's root or absolute.
    directories = ["custom_tools", "my_tools"]
    ```

When you next run `pai`, it will automatically find and load the tools from all specified directories. You'll see a confirmation message in the console at startup.

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

Polyglot AI comes with a set of useful default tools located in the `custom_tools/` directory.

### Code Editor Tool (`code_editor.py`)

This is one of the most powerful tools, allowing the AI to edit code by providing `SEARCH/REPLACE` blocks.

-   `apply_search_replace(edit_script: str)`: Parses and applies one or more `SEARCH/REPLACE` blocks. The edit script must follow a strict format:

    ||````[language]
    ||path/to/file.ext
    ||<<<<<<< SEARCH
    ||content_to_find
    ||=======
    ||content_to_replace_with
    ||>>>>>>> REPLACE
    ||````

    - The `[language]` is optional (e.g., `python`).
    - The `path/to/file.ext` must be the full, correct path relative to the project root.
    - The `<<<<<<< SEARCH` block must contain text that **exactly** matches a section of the target file.
    - To create a new file, the `SEARCH` block must be empty.

**Example Usage:** `"Correct a typo in README.md by changing 'hte' to 'the'."`

### Filesystem Tools (`file_system.py`)

These tools allow the AI to interact with your local filesystem. For security, they are "jailed" to the project's current working directory by default.

-   `read_file(path: str)`: Reads the content of a file.
-   `list_directory(path: str = ".")`: Lists files and folders in a directory.
-   `write_file(path: str, content: str = "")`: Writes content to a file, creating it if needed or overwriting it.
-   `append_to_file(path: str, content: str)`: Appends text to an existing file.
-   `delete_file(path: str)`: Deletes a single file.
-   `delete_directory(path: str)`: Recursively deletes a directory and all its contents. Use with caution.

### Developer Tools (`developer_tools.py`)

Tools for code exploration and inspection, powered by `fd` and `ripgrep`.

-   `find_files(pattern: str, search_path: str = ".")`: Finds files by a glob pattern.
-   `search_code(pattern: str, search_path: str = ".")`: Searches for a regex pattern in files.
-   `read_file_lines(path: str, start_line: int, end_line: int)`: Reads a specific range of lines from a file.

### Data Conversion Tools (`data_converter.py`)

These tools are for converting between common structured data formats.

*   `json_to_yaml(json_string: str)`: Converts a JSON string to YAML.
*   `yaml_to_json(yaml_string: str)`: Converts a YAML string to JSON.
*   `csv_to_json(csv_data: str)`: Converts CSV data into a JSON array of objects.

**Example Usage:** `"take the following CSV data and convert it to JSON: \n\nid,name\n1,alice\n2,bob"`

### Code Generation Tools (`code_generator.py`)

-   `generate_python_class(class_name: str, fields: str)`: Generates a Python dataclass definition.

**Example Usage:** `"generate a python dataclass named 'Book' with fields: title: str, author: str, and published_year: int"`

### HTTP Client Tool (`http_client.py`)

-   `make_http_request(url: str, ...)`: Makes an HTTP request to an external API. Supports `GET`, `POST`, `PUT`, `DELETE`, `PATCH`, and `HEAD`.

**Example Usage:** `"make a GET request to https://api.github.com/zen"`
