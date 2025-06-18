# Polyglot AI: Future Features Roadmap

This document outlines potential features and improvements for the future development of Polyglot AI.

### Core Framework Enhancements

*   **Async Support:** <ins>Completed.</ins> The entire application pipeline, from the command-line interface to the network requests, has been refactored to be fully asynchronous using `httpx` and `asyncio`. This has resolved UI blocking issues and improved responsiveness.
*   **Pydantic Configuration:** Replace `argparse.Namespace` and manual config management with `Pydantic Settings` for automatic environment variable loading, type validation, and cleaner code.
*   **Profile Management:** Allow users to define and save profiles in a local config file (e.g., `~/.polyglot/config.yaml`) that store provider, model, and parameter presets (e.g., `polyglot --profile research_claude`).
*   **Manual Session Management:** While sessions are now logged automatically, add commands to manually save the current conversation state to a named file (`/save <filename>`) and load a previous session to continue it (`/load <filename>`).

### Advanced Agent & Tool Features

*   **Advanced Agentic Control:** Introduce commands to control the agent loop, such as setting max iterations (`/set max_loops 5`) or requiring user confirmation before executing a tool (`/confirm_tools on`).
*   **Multi-Tool Calls:** Improve support for models that can request multiple tool calls in a single turn, running them in parallel where possible.
*   **Dynamic Tool Loading:** Implement a mechanism to load tools from a specified directory, allowing users to add their own tools without modifying the core `tools.py` file.

### Usability and User Experience

*   **Rich Output Formatting:** Integrate the `rich` library to render Markdown, tables, and syntax-highlighted code blocks from the model's output directly in the terminal.
*   **Cost Estimation:** For providers that supply token usage data, provide a running estimate of the session's API cost.
*   **Plugin System:** Formalize the provider system into a true plugin architecture where external packages can register new providers.

### Provider Expansion

*   **Anthropic Provider:** Add a provider for the Claude family of models.
*   **Google Gemini Provider:** Add a provider for Google's Gemini models.
*   **Ollama Provider:** Add a provider to connect to locally-running models via the Ollama server, enabling fully local and private interaction.
*   **Together AI Provider:** Add a provider for the Together AI platform to gain access to a wide variety of open-source models.
