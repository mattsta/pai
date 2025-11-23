# 🪶 Polyglot AI

**A Universal Command-Line Interface for Interacting with Any AI Provider.**

Polyglot AI is an interactive, provider-agnostic CLI for developers, researchers, and AI enthusiasts. It provides a single, unified interface to test, debug, and converse with AI models from any provider, featuring a plug-and-play architecture for easy extension.


### Sample Output

```
$ uv run pai --chat --endpoint featherless --model THUDM/GLM-4-32B-0414 --tools

🪶 Polyglot AI: A Universal CLI for Any AI Provider 🪶
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
*   **MCP (Model Context Protocol) Support:** Connect to external MCP servers to extend tool capabilities with filesystem access, web browsing, database queries, and more. See [MCP Integration Guide](#mcp-integration) below.
*   **Automatic Session Logging:** Every interactive session is automatically saved to a timestamped folder in `logs/`. Each turn is saved as structured JSON, and the entire conversation—including partial responses from cancelled turns—is rendered into multiple browseable HTML formats.
*   **Extensible by Design:** Add new providers via a simple plugin system. Add new tools by dropping Python files into a directory. No core code modification needed.
*   **Multi-Model Arena:** Pit models against each other in a conversational arena, with an optional judge model to provide a final verdict.

### Getting Started

#### 1. Prerequisites

*   Python 3.12+
*   See `pyproject.toml` for a full list of dependencies.

#### 2. Project Structure

The project is organized to separate concerns, making it modular and easy to navigate.

```
/pai/                     # Core application source code
|-- pai.py                # Main application entrypoint and TUI
|-- client.py             # Central client controller
|-- models.py             # Core data models (Conversation, Turn, etc.)
|-- protocols/            # Provider-specific communication logic (adapters)
|-- orchestration/        # Business logic for different modes (chat, agent, arena)
|-- mcp/                  # MCP (Model Context Protocol) client and configuration
|-- commands.py           # Implementation of all /slash commands
|-- tools.py              # Core tool system and @tool decorator
/custom_tools/            # Default directory for user-extendable tools
/prompts/                 # Default directory for system prompts
/docs/                    # Detailed documentation and guides
/logs/                    # Auto-generated session logs (git-ignored)
/session_snapshots/       # User-saved session snapshots (git-ignored)
pai.toml                  # Main configuration for endpoints, profiles, tools, MCP servers
pyproject.toml            # Project definition and dependencies
```

#### 3. API Key Configuration

Polyglot AI loads API keys from environment variables. Set them in your shell or `.env` file:

```bash
export OPENAI_API_KEY="sk-your-openai-key"
```

#### 4. Running the Framework

**Install dependencies and run:**
```bash
pip install uv -U
uv sync -U
```
This prepares a `pai` command and its dependencies locally.

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
*   `/models [term] [refresh]`: Lists and filters models (add 'refresh' to bypass cache).
*   `/info [model_id]`: Shows detailed model info (params, memory, etc.). Defaults to the current model.
*   `/temp <value>`: Changes the generation temperature (e.g., `/temp 0.9`).
*   `/tokens <num>`: Changes the maximum number of tokens for the response (e.g., `/tokens 4000`).
*   `/timeout <seconds>`: Changes the network request timeout (e.g., `/timeout 120`).

**Agent & Tool Controls:**
*   `/agent`: Enables agent mode by loading the `code_editor` system prompt. Requires starting with `--tools`.
*   `/legacy_agent`: Enables agent mode for models that don't support native tool-calling.
*   `/tools`: Toggles the tool-use capability on or off for the current session. (Requires starting `pai` with `--tools`).
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
*   `/save <name>`: Saves the current chat session snapshot to a file in `session_snapshots/`.
*   `/load <name>`: Loads a chat session snapshot from a file.

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

---

## MCP Integration

Polyglot AI supports the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/), Anthropic's open standard for connecting AI models to external tools and data sources. MCP enables your AI assistant to interact with filesystems, databases, web browsers, and custom services.

### Quick Start with MCP

1. **Configure MCP servers in `pai.toml`:**

```toml
[mcp]
enabled = true

[mcp.servers.filesystem]
command = ["npx", "-y", "@modelcontextprotocol/server-filesystem", "/home/user/projects"]
auto_connect = true
timeout = 30.0

[mcp.servers.brave-search]
command = ["npx", "-y", "@anthropic/brave-mcp-server"]
env = { BRAVE_API_KEY = "your-api-key-here" }
auto_connect = true
```

2. **Start PAI with tools enabled:**

```bash
uv run pai --chat --tools
```

MCP servers will automatically connect on startup. You'll see:
```
🔌 MCP support enabled. Loading MCP servers...
  📦 Configured 2 MCP server(s)
  ✅ MCP server 'filesystem' connected (11 tools)
  ✅ MCP server 'brave-search' connected (2 tools)
```

3. **Use MCP tools naturally in conversation:**

```
👤 User: List all Python files in the projects directory

🤖 Assistant: I'll use the filesystem tools to list Python files...
   [Calls mcp__filesystem__list_directory with path="/home/user/projects"]

   Found 15 Python files:
   - main.py
   - utils.py
   ...
```

### MCP Commands

| Command | Description |
|---------|-------------|
| `/mcp status` | Show connection status of all MCP servers |
| `/mcp list` | List all available MCP tools with descriptions |
| `/mcp connect <server>` | Manually connect to a specific MCP server |
| `/mcp disconnect <server>` | Disconnect from an MCP server |

### MCP Configuration Reference

Each MCP server is configured as a subsection under `[mcp.servers]`:

```toml
[mcp.servers.my-server]
# Required: Command and arguments to start the server
command = ["node", "/path/to/server.js", "--arg1", "value"]

# Optional: Environment variables passed to the server
env = { API_KEY = "secret", DEBUG = "true" }

# Optional: Enable/disable this server (default: true)
enabled = true

# Optional: Auto-connect when PAI starts with --tools (default: true)
auto_connect = true

# Optional: Connection timeout in seconds (default: 30.0)
timeout = 30.0
```

### Popular MCP Servers

Here are some commonly used MCP servers you can configure:

| Server | Install Command | Description |
|--------|----------------|-------------|
| **Filesystem** | `npx -y @modelcontextprotocol/server-filesystem /path` | Read/write files, list directories |
| **Brave Search** | `npx -y @anthropic/brave-mcp-server` | Web search via Brave API |
| **GitHub** | `npx -y @anthropic/mcp-server-github` | GitHub API access |
| **SQLite** | `npx -y @anthropic/mcp-server-sqlite --db-path /path/db.sqlite` | SQLite database queries |
| **Postgres** | `npx -y @anthropic/mcp-server-postgres` | PostgreSQL database access |

### Example: Full MCP Configuration

```toml
# pai.toml

[mcp]
enabled = true

# Local filesystem access (restricted to specific directory)
[mcp.servers.filesystem]
command = ["npx", "-y", "@modelcontextprotocol/server-filesystem", "/home/user/workspace"]
auto_connect = true

# Web search capability
[mcp.servers.brave]
command = ["npx", "-y", "@anthropic/brave-mcp-server"]
env = { BRAVE_API_KEY = "${BRAVE_API_KEY}" }
auto_connect = true

# Database access (connect manually when needed)
[mcp.servers.database]
command = ["npx", "-y", "@anthropic/mcp-server-postgres"]
env = { POSTGRES_CONNECTION_STRING = "postgresql://user:pass@localhost/mydb" }
auto_connect = false

# Custom local server
[mcp.servers.custom]
command = ["python", "-m", "my_mcp_server"]
env = { CONFIG_PATH = "./config.json" }
timeout = 60.0
```

### How MCP Tools Work

1. **Tool Discovery**: When an MCP server connects, PAI discovers its available tools
2. **Tool Naming**: MCP tools are prefixed with `mcp__<server>__<tool>` (e.g., `mcp__filesystem__read_file`)
3. **Seamless Integration**: MCP tools appear alongside native tools in the AI's available functions
4. **Automatic Routing**: PAI automatically routes tool calls to the appropriate MCP server

### Troubleshooting MCP

**Server won't connect:**
- Check that the command exists and is executable
- Verify Node.js/npx is installed for npm-based servers
- Check server logs: `uv run pai --debug --chat --tools`

**Tools not appearing:**
- Verify the server is connected: `/mcp status`
- Check if server discovered tools: `/mcp list`

**Environment variables not working:**
- Use `env = { VAR = "${VAR}" }` to reference shell environment variables
- Or set them directly: `env = { VAR = "value" }`

---
