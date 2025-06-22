# Polyglot AI: Feature Summary

This document summarizes the core features and capabilities of the Polyglot AI framework.

### Core Framework

*   **Asynchronous Core:** The entire application pipeline, from the command-line interface to the network requests, is fully asynchronous using `httpx` and `asyncio`, ensuring a responsive, non-blocking user interface.
*   **Robust Configuration:** The framework uses `typer` for CLI argument parsing and `Pydantic` for config file validation, providing strong type safety, clear error messages, and a clean system for managing all application settings.
*   **Profile Management:** A `--profile <name>` flag allows loading preset configurations (endpoint, model, temperature, etc.) from a `[profiles]` section in `polyglot.toml`. CLI flags will always override profile settings for maximum flexibility.
*   **Manual Session Management:** Interactive chat sessions can be manually saved to and loaded from disk using the `/save <name>` and `/load <name>` commands.
*   **Plugin System:** The provider system is a formal plugin architecture. New providers can be added by creating and installing separate Python packages that register a `ProtocolAdapter` using Python's `entry_points` mechanism, making the framework truly extensible without modifying the core code.

### Advanced Agent & Tool Features

*   **Agentic Control:** A `--confirm` flag and `/confirm on|off` command require user confirmation before executing any tool, providing a critical safety layer for agentic workflows.
*   **Parallel Multi-Tool Calls:** The framework supports models that can request multiple tool calls in a single turn, running them concurrently with `asyncio.gather` to improve performance.
*   **Multi-Model Arena:** The application supports an "arena" mode where two or more models can converse with each other and an optional "judge" model can provide a verdict.
*   **Arena Interjection:** The multi-model arena can be paused and resumed. Users can interject with their own messages using the `/say` command while the arena is paused, allowing them to steer the conversation.

### Usability and User Experience

*   **Rich Output Formatting:** Final model output is rendered as Markdown in the terminal using the `rich` library. HTML session logs also render Markdown and feature syntax-highlighted code blocks.
*   **Cost Estimation:** The framework tracks and displays a running estimate of the session's API cost (for supported providers) in the toolbar and `/stats` command.
*   **Command Autocompletion:** A fuzzy completer is integrated for slash commands. Typing part of a command (e.g., `/swi`) and pressing `Tab` shows a list of possible completions.

### Future Enhancements

*   **Provider Expansion:** Adding support for more API providers, such as Google Gemini and Together AI.
*   **Advanced Prompt Management:** Designing a system to manage multiple, stackable `system` prompts.
