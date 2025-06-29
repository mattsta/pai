# Polyglot AI: A Practical Guide to Agentic Tool Use

This tutorial provides a step-by-step guide on how to use tools in an interactive `pai` session. For a guide on how to *create* tools, see the [Tool System Guide](./TOOLS.md).

## The Core Concepts: Capability vs. Behavior

Using tools effectively requires understanding two key concepts:

1.  **Tool Capability:** This is the AI's *ability* to see and execute tools. You enable this capability by starting `pai` with the `--tools` flag. This is the master switch; without it, no tools are loaded, and the AI is unaware of them.
2.  **Agentic Behavior:** This is the AI's set of instructions on *how* to behave. You guide its behavior by setting a system prompt. Prompts like the one loaded with `/agent` instruct the AI to think step-by-step, explore problems, use tools to achieve goals, and verify its work.

You need both to accomplish complex tasks: the capability to act, and the instructions on how to act.

## Step 1: Start a Session with Tool Capability

To load all available tools and enable the tool-use capability, you **must** start `pai` with the `--tools` flag.

```bash
# This loads tools from custom_tools/ and enables the /tools command
uv run pai --chat --endpoint openai --model gpt-4o --tools
```

## Step 2: Your First Tool Call (Simple Task)

The easiest way to trigger a tool is to ask a question that directly maps to a tool's description.

**User Prompt:**
> What is the weather in Paris?

**What Happens Next (The Agent Loop):**
1.  **Model Recognizes Intent:** Your prompt is sent to the AI. The AI also receives the list of available tools and their descriptions, including `get_current_weather`. The model sees that "what is the weather" is a very close match for the tool's purpose.
2.  **Model Requests Tool Call:** The model doesn't answer you directly. Instead, it sends back a special `tool_calls` message to `pai`, asking to run `get_current_weather(location="Paris")`.
3.  **`pai` Executes the Tool:** The `pai` framework executes the local Python function and captures its output as a JSON string (e.g., `'{"status":"success","result":{"location":"Paris", ...}}'`).
4.  **`pai` Reports Back to Model:** `pai` sends the raw JSON string result back to the model in a new message. The conversation history now includes your prompt, the model's request to call the tool, and the tool's string output.
5.  **Model Generates Final Answer:** The model, now equipped with the weather data, parses the JSON string and generates a final, user-friendly response based on the tool's output.

*   **Tool Result:** A JSON string from the tool: `'{"status": "success", "result": {"location": "Paris", "temperature": "22", "condition": "Sunny"}}'`

**AI's Final Response:**
> The current weather in Paris is 22°C and Sunny.

## Step 3: A Complex Task (Agentic Behavior)

For more complex tasks, you need to guide the AI with a system prompt that encourages agentic behavior.

**1. Enable Agent Mode:**
```
/agent
```
This loads the `prompts/code_editor.md` system prompt, which instructs the model to think step-by-step and use tools to accomplish its goals.

**2. Give the AI a Task:**
> Please find the `get_current_weather` function and tell me what file it is in.

**What Happens Next (Multi-Step Agent Loop):**
The AI will now follow a multi-step reasoning process:

*   **Thought:** "The user wants me to find a function. The `search_code` tool is perfect for this."
*   **Action:** It calls `search_code(pattern="def get_current_weather")`.
*   **Observation:** The tool returns a JSON object showing a match in `pai/tools.py`.
*   **Thought:** "I have found the file. Now I can answer the user's question."
*   **Final Answer:** It formulates a final response based on the tool's output.

**AI's Final Response:**
> The `get_current_weather` function is located in the file `pai/tools.py`.

This multi-step process—exploring, investigating, and acting—is the foundation of agentic behavior in Polyglot AI. By giving the AI the right tools and guiding it, you can accomplish complex tasks.

## Step 4: Using Legacy Agent Mode

What if your model doesn't support native tool calling (like OpenAI's API)? Polyglot AI provides `/legacy_agent` mode for this.

In this mode, `pai` includes a text-based list of available tools directly in the system prompt. It instructs the model to respond with a special XML tag (`<tool_call>...`) when it wants to use a tool. `pai` then parses this text, runs the tool, and feeds the result back to the model.

**How to use it:**
1.  Start with `--tools` to load the tools: `uv run pai --chat --endpoint <your-endpoint> --tools`
2.  Type `/legacy_agent`. This loads a special prompt and disables native tool calling.
3.  Give the AI a task.

The AI will follow the same thought process, but instead of the API handling tool calls, the agent's reasoning and `pai`'s parsing logic manage the entire loop via plain text. This brings agentic capabilities to a much wider range of models.
