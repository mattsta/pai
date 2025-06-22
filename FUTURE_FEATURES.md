# Polyglot AI: Future Features Roadmap

This document outlines potential features and improvements for the future development of Polyglot AI.

### Core Framework Enhancements

*   **Async Support:** <ins>Completed.</ins> The entire application pipeline, from the command-line interface to the network requests, has been refactored to be fully asynchronous using `httpx` and `asyncio`. This has resolved UI blocking issues and improved responsiveness.
*   **Pydantic Configuration:** <ins>Completed.</ins> Replaced `argparse` with `typer` for CLI argument parsing and `Pydantic` for config file validation. This provides robust type safety, better error messages, and cleaner code for managing all application settings.
*   **Profile Management:** Allow users to define and save profiles in a local config file (e.g., `~/.polyglot/config.yaml`) that store provider, model, and parameter presets (e.g., `polyglot --profile research_claude`).
*   **Manual Session Management:** <ins>Completed.</ins> Added `/save <name>` and `/load <name>` commands to manually save and restore interactive chat sessions.

### Advanced Agent & Tool Features

*   **Advanced Agentic Control:** <ins>Completed.</ins> Introduced a command (`/confirm on|off`) and a flag (`--confirm`) to require user confirmation before executing any tool. This provides a critical safety layer for agentic workflows.
*   **Multi-Tool Calls:** Improve support for models that can request multiple tool calls in a single turn, running them in parallel where possible.
*   **Arena Interjection:** <ins>Completed.</ins> The multi-model arena can now be paused and resumed using `/pause` and `/resume`. The user can interject with their own messages using the `/say` command while the arena is paused, allowing them to steer the conversation.
*   **Arena Judge:** <ins>Completed.</ins> Arenas can now include an optional third `judge` participant. The judge is given the full conversation history at the end of the dialogue and provides a summary and verdict, which is also logged.

### Usability and User Experience

*   **Rich Output Formatting:** <ins>Completed.</ins> Integrated the `rich` library to render final model output as Markdown in the terminal. HTML logs now also render Markdown and syntax-highlighted code blocks using `marked.js` and `highlight.js`.
*   **Cost Estimation:** <ins>Completed.</ins> The framework now tracks and displays a running estimate of the session's API cost for supported providers (OpenAI, Anthropic) in the toolbar and `/stats` command.
*   **Command Autocompletion:** <ins>Completed.</ins> Integrated a fuzzy completer for slash commands. Typing part of a command (e.g., `/swi`) and pressing `Tab` will now show a list of possible completions.
*   **Plugin System:** After completing the "Next Steps for Refactoring," formalize the provider system into a true plugin architecture where external packages can register new providers. The proposed refactors (especially Orchestrator Extraction and State Management) are critical prerequisites for a clean plugin API.

### Provider Expansion

*   **Anthropic Provider:** <ins>Completed.</ins> Added a provider for the Claude family of models using the Messages API.
*   **Google Gemini Provider:** Add a provider for Google's Gemini models.
*   **Ollama Provider:** <ins>Completed.</ins> Added a provider to connect to locally-running models via the Ollama server, enabling fully local and private interaction.
*   **Together AI Provider:** Add a provider for the Together AI platform to gain access to a wide variety of open-source models.
