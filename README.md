# 🪶 Polyglot AI Framework

**A Universal Command-Line Interface for Interacting with Any AI Provider.**

Polyglot AI is an interactive, provider-agnostic CLI designed for developers, researchers, and AI enthusiasts. It provides a single, unified interface to test, debug, and converse with a multitude of AI models from different providers like Featherless and OpenAI, with a plug-and-play architecture to easily add more.

![Demo Screenshot](https://i.imgur.com/g88t8D4.png) <!-- A placeholder for a future screenshot -->

### Key Features

*   **Provider Agnostic:** Seamlessly switch between different AI providers (`Featherless`, `OpenAI`, etc.) in a single session.
*   **Interactive Chat:** A rich, terminal-based chat experience with persistent and searchable command history (up/down arrows for navigation, prefix search, and `Ctrl+R` for reverse search), streaming, helper commands, and a live status toolbar.
*   **Powerful Debugging:** A first-class, verbose debug mode to inspect raw API traffic, essential for development and research.
*   **Tool & Function Calling:** An integrated system for allowing models to use local Python functions to answer questions (e.g., for real-time data or actions).
*   **Agentic Looping:** The framework supports basic agentic loops where the model can use tools iteratively to solve complex problems.
*   **Automatic Session Logging:** Every interactive session is automatically saved to a timestamped folder in the `sessions/` directory. Each turn is saved as a structured JSON file, and the entire conversation is rendered into multiple browseable HTML formats.
*   **Extensible by Design:** Adding a new provider is as simple as creating a new class and registering it.

### Getting Started

#### 1. Prerequisites

*   Python 3.12+
*   See `pyproject.toml` for a full list of dependencies.

#### 2. Project Structure

Clone or download the repository. The core structure is:

```
/pai/
|-- pai.py                # The main script and orchestrator
|-- protocols/
|   |-- base_adapter.py
|   |-- openai_chat_adapter.py
|   |-- ...
|-- templates/
|   |-- conversation.html
|   |-- gptwink_format.html
|-- tools.py
/sessions/                # Auto-generated for session logs
/docs/                    # Detailed documentation
polyglot.toml             # Endpoint configuration
pyproject.toml            # Project dependencies and definition
```

#### 3. API Key Configuration

Polyglot AI loads API keys from environment variables. Set them in your shell or `.env` file:

```bash
export FEATHERLESS_API_KEY="sk-your-featherless-key"
export OPENAI_API_KEY="sk-your-openai-key"
```

#### 4. Running the Framework

**Install the package in editable mode:**
```bash
pip install -e .
```
This will install the `pai` command.

**Start in Interactive Mode (Default: OpenAI)**
```bash
pai --chat
```

**Start with a Different Endpoint (e.g., Featherless)**
```bash
pai --chat --endpoint featherless --model featherless/claude-3-opus
```

**Run a Single, Non-Interactive Prompt**
```bash
# This will stream the response by default.
pai --endpoint openai --prompt "Explain quantum computing in one sentence."

# Use --no-stream to get the full response at once.
pai --endpoint openai --prompt "Explain quantum computing in one sentence." --no-stream
```

**Enable Tool Calling**
```bash
pai --chat --endpoint openai --model gpt-4o --tools
```
Inside the interactive session, you can then ask: `What is the weather like in Paris?`

### Interactive Commands

Once in interactive mode, use `/` commands to control the session:

*   `/help`: Shows the list of available commands.
*   `/stats`: Displays statistics for the current session.
*   `/switch <name>`: Switch to a different endpoint (e.g., `/switch openai`).
*   `/model <name>`: Change the model (e.g., `/model gpt-4o-mini`).
*   `/system <prompt>`: Set a new system prompt for the chat.
*   `/prompt <name>`: Load a system prompt from the `prompts/` directory.
*   `/tools`: Toggle tool-calling on or off.
*   `/debug`: Toggle raw protocol debugging on or off.
*   `/clear`: Clear the current conversation history.
*   `/quit`: Exit the application.

For more details on session logging, see `docs/LOGGING.md`.
```

---

