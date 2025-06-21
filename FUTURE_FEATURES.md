# Polyglot AI: Future Features Roadmap

This document outlines potential features and improvements for the future development of Polyglot AI.

### Core Framework Enhancements

*   **Async Support:** <ins>Completed.</ins> The entire application pipeline, from the command-line interface to the network requests, has been refactored to be fully asynchronous using `httpx` and `asyncio`. This has resolved UI blocking issues and improved responsiveness.
*   **Pydantic Configuration:** <ins>Completed.</ins> Replaced `argparse` with `typer` for CLI argument parsing and `Pydantic` for config file validation. This provides robust type safety, better error messages, and cleaner code for managing all application settings.
*   **Profile Management:** Allow users to define and save profiles in a local config file (e.g., `~/.polyglot/config.yaml`) that store provider, model, and parameter presets (e.g., `polyglot --profile research_claude`).
*   **Manual Session Management:** While sessions are now logged automatically, add commands to manually save the current conversation state to a named file (`/save <filename>`) and load a previous session to continue it (`/load <filename>`).

### Advanced Agent & Tool Features

*   **Advanced Agentic Control:** Introduce commands to control the agent loop, such as setting max iterations (`/set max_loops 5`) or requiring user confirmation before executing a tool (`/confirm_tools on`).
*   **Multi-Tool Calls:** Improve support for models that can request multiple tool calls in a single turn, running them in parallel where possible.
*   **Arena Interjection:** <ins>Completed.</ins> The multi-model arena can now be paused and resumed using `/pause` and `/resume`. The user can interject with their own messages using the `/say` command while the arena is paused, allowing them to steer the conversation.
*   **Arena Judge:** <ins>Completed.</ins> Arenas can now include an optional third `judge` participant. The judge is given the full conversation history at the end of the dialogue and provides a summary and verdict, which is also logged.

### Usability and User Experience

*   **Rich Output Formatting:** <ins>Completed.</ins> Integrated the `rich` library to render final model output as Markdown in the terminal. HTML logs now also render Markdown and syntax-highlighted code blocks using `marked.js` and `highlight.js`.
*   **Cost Estimation:** For providers that supply token usage data, provide a running estimate of the session's API cost.
*   **Plugin System:** Formalize the provider system into a true plugin architecture where external packages can register new providers.

### Provider Expansion

*   **Anthropic Provider:** <ins>Completed.</ins> Added a provider for the Claude family of models using the Messages API.
*   **Google Gemini Provider:** Add a provider for Google's Gemini models.
*   **Ollama Provider:** Add a provider to connect to locally-running models via the Ollama server, enabling fully local and private interaction.
*   **Together AI Provider:** Add a provider for the Together AI platform to gain access to a wide variety of open-source models.
