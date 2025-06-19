import subprocess
from pai.tools import tool

# --- DANGER ZONE ---
# The `execute_shell` tool is extremely powerful and dangerous.
# It allows the AI to run arbitrary commands on your system.
# It is DISABLED by default. To enable it, you must change
# `SHELL_ACCESS_ENABLED` to True below.
#
# DO NOT enable this unless you fully understand the risks and are running
# in a sandboxed environment. You are responsible for any actions taken
# by the AI using this tool.

SHELL_ACCESS_ENABLED = False


@tool
def execute_shell(command: str) -> str:
    """Executes a shell command and returns its output.

    THIS IS A DANGEROUS TOOL. It is disabled by default for security.
    The AI can run any command on your computer, which could have unintended
    consequences, including deleting files or exposing sensitive information.
    Only enable this if you know what you are doing.

    Args:
        command (str): The shell command to execute.
    """
    if not SHELL_ACCESS_ENABLED:
        return "Error: Shell access is disabled for security reasons. You must edit custom_tools/shell.py to enable it."

    try:
        # Use a timeout to prevent long-running commands from hanging.
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,  # 30-second timeout
            check=True,  # Raise CalledProcessError if return code is non-zero
        )
        output = result.stdout
        if not output.strip():
            return f"Command '{command}' executed successfully with no output."
        # Combine stdout and stderr for a more complete picture
        full_output = f"STDOUT:\n{result.stdout}"
        if result.stderr:
            full_output += f"\nSTDERR:\n{result.stderr}"
        return full_output
    except subprocess.TimeoutExpired:
        return f"Error: Command '{command}' timed out after 30 seconds."
    except subprocess.CalledProcessError as e:
        error_output = f"STDERR:\n{e.stderr}" if e.stderr else "No stderr output."
        return (
            f"Error: Command '{command}' failed with exit code {e.returncode}.\n{error_output}"
        )
    except Exception as e:
        return f"An unexpected error occurred: {e}"
