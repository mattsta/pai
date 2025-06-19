import json
import shutil
import subprocess
from pai.tools import tool
from .file_system import is_safe_path, WORKSPACE

# --- Prerequisite Note ---
# The tools in this file depend on external command-line utilities.
# For these tools to work, you must have them installed and available in your PATH.
#
# - `find_files` requires 'fd' (https://github.com/sharkdp/fd)
# - `search_code` requires 'ripgrep' (https://github.com/BurntSushi/ripgrep)
#
# You can usually install them with your system's package manager, e.g.:
# `brew install fd ripgrep` or `apt-get install fd-find ripgrep`


def _run_command(command: list[str], check_tool_name: str, working_dir: str) -> str:
    """A helper to safely run shell commands and check for presence first."""
    tool_path = shutil.which(check_tool_name)
    if not tool_path:
        return f"Error: The '{check_tool_name}' command is not installed or not in your PATH. Please install it to use this tool."

    try:
        result = subprocess.run(
            [tool_path] + command,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,  # We check the return code manually
            cwd=working_dir,
        )
        if result.returncode != 0 and result.returncode != 1:
            # rg returns 1 for no matches, which is not an error here.
            return f"Error: Command failed with exit code {result.returncode}.\nSTDERR:\n{result.stderr}"
        return result.stdout
    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after 15 seconds."
    except Exception as e:
        return f"An unexpected error occurred: {e}"


@tool
def find_files(pattern: str, search_path: str = ".") -> str:
    """Finds files by name/pattern using 'fd'. Returns a JSON list of paths.

    Args:
        pattern (str): A glob pattern to search for (e.g., "*.py", "README.md").
        search_path (str): The directory to start searching from, relative to the project root.
    """
    if not is_safe_path(search_path):
        return "Error: Search path is outside the allowed workspace."

    command = ["--json", "--glob", pattern, search_path]
    return _run_command(command, "fd", str(WORKSPACE))


@tool
def search_code(pattern: str, search_path: str = ".") -> str:
    """Searches for a regex pattern in files using 'ripgrep' (rg).

    Returns a JSON stream of matches, including line numbers and content.

    Args:
        pattern (str): The regex pattern to search for.
        search_path (str): The file or directory to search within.
    """
    if not is_safe_path(search_path):
        return "Error: Search path is outside the allowed workspace."

    command = ["--json", pattern, search_path]
    stdout = _run_command(command, "rg", str(WORKSPACE))

    # rg --json returns a newline-delimited stream. We parse it into a single JSON array.
    if stdout.startswith("Error:"):
        return stdout

    matches = []
    for line in stdout.strip().split("\n"):
        if line:
            try:
                matches.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # Ignore non-JSON lines if any
    return json.dumps(matches, indent=2)


@tool
def read_file_lines(path: str, start_line: int, end_line: int) -> str:
    """Reads and returns a specific range of lines from a text file.

    Lines are 1-indexed. Both start and end lines are inclusive.

    Args:
        path (str): The path to the file.
        start_line (int): The starting line number to read (1-indexed).
        end_line (int): The ending line number to read (inclusive).
    """
    if not is_safe_path(path):
        return "Error: Path is outside the allowed workspace."
    if start_line <= 0 or end_line < start_line:
        return "Error: Invalid line range. Start must be > 0 and end >= start."

    try:
        with open(WORKSPACE / path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        start_index = start_line - 1
        end_index = end_line

        if start_index >= len(lines):
            return "Error: Start line is after the end of the file."

        selected_lines = lines[start_index:end_index]

        # Prepend line numbers to the output for context
        output_lines = [
            f"{i + start_line}: {line.rstrip()}"
            for i, line in enumerate(selected_lines)
        ]

        if not output_lines:
            return f"Note: No lines found in range {start_line}-{end_line} for file '{path}'."

        return "\n".join(output_lines)

    except FileNotFoundError:
        return f"Error: File not found at '{path}'"
    except Exception as e:
        return f"Error reading file '{path}': {e}"
