# 🪶 Polyglot AI Framework

**A Universal Command-Line Interface for Interacting with Any AI Provider.**

Polyglot AI is an interactive, provider-agnostic CLI designed for developers, researchers, and AI enthusiasts. It provides a single, unified interface to test, debug, and converse with a multitude of AI models from different providers with a plug-and-play architecture to easily add more.


### Sample Output

```
$ uv run pai --chat --endpoint featherless --model THUDM/GLM-4-32B-0414 --tools

🪶 Polyglot AI: A Universal CLI for the OpenAI API Format 🪶
🔎 Loading protocol adapters from entry point group 'polyglot_ai.protocols'...
  ✅ Loaded adapter 'anthropic' from 'pai.protocols.anthropic_adapter'
  ✅ Loaded adapter 'legacy_completion' from 'pai.protocols.legacy_completion_adapter'
  ✅ Loaded adapter 'ollama' from 'pai.protocols.ollama_adapter'
  ✅ Loaded adapter 'openai_chat' from 'pai.protocols.openai_chat_adapter'
🛠️  --tools flag detected. Loading tools...
🔎 Loading custom tools from: custom_tools
  ✅ Loaded custom tool module: data_converter.py
  ✅ Loaded custom tool module: http_client.py
  ✅ Loaded custom tool module: developer_tools.py
  ✅ Loaded custom tool module: code_editor.py
  ✅ Loaded custom tool module: file_system.py
  ✅ Loaded custom tool module: code_generator.py
✅ Switched to endpoint: featherless
🎯 Chat Mode | Endpoint: featherless | Model: THUDM/GLM-4-32B-0414
💾 Session logs will be saved to: sessions/2025-06-28_09-34-42-interactive
Type '/help' for commands, '/quit' to exit.
------------------------------------------------------------

👤 (Chat) User: hello how are you today
╭─ 🤖 Assistant ──────────────────────────────────────────────────────────────────────────────────────╮
│ Hello! I'm doing well, thank you for asking. As an AI, I don't have feelings, but I'm functioning   │
│ optimally. How about you? Is there anything I can assist you with today?                            │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────╯

👤 (Chat) User: /mode
🧹 History cleared.
✅ Switched to Completion mode.

👤 (Completion) User: /tokens 30
✅ Max tokens set to: 30

👤 (Completion) User: i really want to dance but
╭─ 🤖 Assistant ──────────────────────────────────────────────────────────────────────────────────────╮
│ i have no idea how to so i just like... stand there awkwardly when music comes on or try to wiggle  │
│ a bit and it looks stupid                                                                           │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

### Key Features

*   **Universal Provider Support:** Seamlessly switch between different AI providers and profiles in a single session using `/switch` and `/profile`.
*   **Advanced Interactive TUI:** A rich, terminal-based chat experience built on `prompt-toolkit`, featuring persistent history, multiline input, and a live status toolbar with default smooth-stream rendering that provides real-time feedback on cost, performance, and agent status.
*   **Deep Introspection & Debugging:** A first-class, verbose debug mode (`--debug`) to inspect raw API traffic, and a powerful `/stats` command to see detailed performance metrics for every request.
*   **Customizable Pricing Engine:** While default pricing is fetched automatically, you can provide a custom YAML or TOML file to override costs, define pricing for local models, and even specify complex tiered or time-based pricing rules. See the [Custom Pricing Guide](./docs/PRICING.md) for details.
*   **Powerful Agentic Tool-Use:** An extensible system allowing models to use local Python functions as tools. Supports native tool-calling APIs (OpenAI, etc.) and provides a legacy agent mode for models that lack this capability.
    *   **To create tools:** See the [Tool System Guide](./docs/TOOLS.md).
    *   **For a walkthrough:** Check out the [Tool Usage Tutorial](./docs/TOOL_TUTORIAL.md).
*   **Automatic Session Logging:** Every interactive session is automatically saved to a timestamped folder in `sessions/`. Each turn is saved as structured JSON, and the entire conversation is rendered into multiple browseable HTML formats.
*   **Extensible by Design:** Add new providers via a simple plugin system. Add new tools by dropping Python files into a directory. No core code modification needed.
*   **Multi-Model Arena:** Pit models against each other in a conversational arena, with an optional judge model to provide a final verdict.

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
pai.toml                  # Endpoint configuration
pyproject.toml            # Project dependencies and definition
```

#### 3. API Key Configuration

Polyglot AI loads API keys from environment variables. Set them in your shell or `.env` file:

```bash
export OPENAI_API_KEY="sk-your-openai-key"
```

#### 4. Running the Framework

**Install the package in editable mode:**
```bash
uv pip install -e .
```
This installs the `pai` command and its dependencies in editable mode, allowing you to make changes to the source code that are immediately reflected.

**Start in Interactive Mode (Default: OpenAI)**
```bash
uv run pai --chat
```

**Using a Profile**
You can define preset configurations in `pai.toml` and use them with the `--profile` flag. This is great for switching between common setups.
```bash
uv run pai --profile research_haiku --chat
```

**Run a Single, Non-Interactive Prompt**
```bash
# This will stream the response by default.
uv run pai --endpoint openai --prompt "Explain quantum computing in one sentence."

# Use --no-stream to get the full response at once.
uv run pai --endpoint openai --prompt "Explain quantum computing in one sentence." --no-stream
```

**Using Tools**
To load and enable tools, you must start `pai` with the `--tools` flag. This gives the AI the *capability* to see and use tools.
```bash
uv run pai --chat --endpoint openai --model gpt-4o --tools
```
For complex tasks, you can then switch into "agent mode" to give the AI a better reasoning framework for using those tools.
```
/agent
```
Now you can ask it to perform tasks that require tools: `Refactor the 'get_current_weather' function in 'pai/tools.py' to handle a 'kelvin' unit.`

### Interactive Commands

Once in interactive mode, use `/` commands to control the session:

*   `/help`: Shows this list of commands.
*   `/stats`: Displays performance and cost statistics for the current session.
*   `/quit` or `/q`: Exits the application.

**Provider & Model Controls:**
*   `/endpoints`: Lists all available provider endpoints from your config file.
*   `/switch <name>`: Switches to a different provider endpoint (e.g., `/switch anthropic`).
*   `/model <name>`: Changes the model for the current session (e.g., `/model gpt-4o-mini`).
*   `/temp <value>`: Changes the generation temperature (e.g., `/temp 0.9`).
*   `/tokens <num>`: Changes the maximum number of tokens for the response (e.g., `/tokens 4000`).
*   `/timeout <seconds>`: Changes the network request timeout (e.g., `/timeout 120`).

**Agent & Tool Controls:**
*   `/agent`: Enables agent mode by loading the `code_editor` system prompt. Requires starting with `--tools`.
*   `/legacy_agent`: Enables agent mode for models that don't support native tool-calling.
*   `/tools`: Toggles the tool-use capability on or off for the current session.
*   `/confirm on|off`: Toggles whether the agent must ask for confirmation before executing a tool.

**Chat & History Management:**
*   `/mode`: Toggles between `chat` and `completion` modes. Clears history.
*   `/system <text>`: Replaces the entire system prompt stack with new text.
*   `/system add <text>`: Adds a new system prompt to the top of the stack.
*   `/system pop`: Removes the most recent system prompt from the stack.
*   `/system show`: Shows all system prompts currently in the stack.
*   `/system clear`: Clears all system prompts.
*   `/prompts`: Lists all available, loadable system prompts from the `prompts/` directory.
*   `/prompt <name>`: Loads a prompt from the `prompts/` directory and adds it to the system prompt stack.
*   `/clear`: Clears the current conversation history.
*   `/history`: Shows the raw message history for the current conversation.
*   `/save <name>`: Saves the current chat session to a file in `saved_sessions/`.
*   `/load <name>`: Loads a chat session from a file.

**Multi-Model Arena:**
*   `/arena <name> [turns]`: Starts a multi-model arena conversation defined in `pai.toml`.
*   `/pause`: Pauses the arena conversation after the current model's turn.
*   `/resume`: Resumes a paused arena.
*   `/say <message>`: While paused, interjects with a message to steer the conversation.

**UI & Debugging:**
*   `/multiline`: Toggles multi-line input mode (use `Esc+Enter` to submit).
*   `/stream`: Toggles response streaming on or off.
*   `/rich`: Toggles rich Markdown rendering for final output.
*   `/smooth`: Toggles the adaptive smooth streaming mode (ON by default).
*   `/verbose`: Toggles verbose logging of request parameters.
*   `/debug`: Toggles raw protocol-level debugging for network streams.

For more details on session logging, see [`docs/LOGGING.md`](docs/LOGGING.md).
