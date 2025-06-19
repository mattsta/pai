# Polyglot AI: A Practical Guide to Using Tools

This tutorial provides a step-by-step guide on how to use tools in an interactive `pai` session. For a guide on how to *create* your own tools, see the [Tool System Guide](./TOOLS.md).

Using tools effectively allows you to turn the AI into a powerful agent that can interact with your local file system, browse the web, and even edit code.

## Step 1: Start a Session with Tools Enabled

To use tools, you need two things:
1. A capable model that is good at following instructions (e.g., `gpt-4o`, `claude-3-5-sonnet`).
2. The `--tools` flag when launching `pai`.

Here is the recommended command to start a tool-enabled session:
```bash
pai --chat --endpoint openai --model gpt-4o --tools
```

## Step 2: Your First Tool Call (Single-Turn)

The easiest way to trigger a tool is to ask a question that directly maps to a tool's description.

**User Prompt:**
> What is the weather in Paris?

**What Happens Next (The Agent Loop):**
1.  **Model Recognizes Intent:** Your prompt is sent to the AI. The AI also receives the list of available tools and their descriptions, including `get_current_weather`. The model sees that "what is the weather" is a very close match for the tool's purpose.
2.  **Model Requests Tool Call:** The model doesn't answer you directly. Instead, it sends back a special `tool_calls` message to `pai`, asking to run `get_current_weather(location="Paris")`.
3.  **`pai` Executes the Tool:** The `pai` framework executes the local Python function and captures its output (e.g., a JSON string: `{"location": "Paris", "temperature": "22", ...}`).
4.  **`pai` Reports Back to Model:** `pai` sends the tool's output back to the model in a new message. The conversation history now includes your prompt, the model's request to call the tool, and the tool's result.
5.  **Model Generates Final Answer:** The model, now equipped with the weather data, generates a final, user-friendly response based on the tool's output.

**AI's Final Response:**
> The weather in Paris is 22°C and Sunny.

## Step 3: A Complex Task (Multi-Turn Agentic Behavior)

For more complex tasks, you need to guide the AI step-by-step. Let's try to fix a typo in `README.md`.

**User Prompt 1: Set the Context**
First, let's load the `code_editor` system prompt to tell the AI to act like a software engineer.
> /prompt code_editor

**AI Response:**
> 🤖 System prompt loaded from 'code_editor'. History cleared.

**User Prompt 2: Explore**
Now, let's start the task. Assume there is a typo in `README.md` where `uv sync -U` is written as `uv cync -U`.

> I need to fix a typo in the project's README file. First, can you confirm the file `README.md` exists?

**AI's Thought Process:** "The user wants to find a file. I have a `find_files` tool that seems perfect for this."
*   **Tool Call:** `find_files(pattern="README.md")`
*   **Tool Result:** `["README.md"]`

**AI's Response to User:**
> Yes, I found the file `README.md`. What should I look for?

**User Prompt 3: Investigate**
> Great. Search for the line containing `uv cync -U` in that file.

**AI's Thought Process:** "The user wants to search inside a file. The `search_code` tool is the right choice."
*   **Tool Call:** `search_code(pattern="uv cync -U", search_path="README.md")`
*   **Tool Result:** A JSON object showing the match, file path, and line number.

**AI's Response to User:**
> Okay, I found a match. The text `uv cync -U` appears on line 60 of `README.md`.

**User Prompt 4: Execute the Edit**
> Perfect. Please correct it to `uv sync -U`.

**AI's Thought Process:** "The user wants to modify a file. The `apply_search_replace` tool is designed for this. I need to construct the correct `SEARCH/REPLACE` block."

*   **Tool Call:** `apply_search_replace(edit_script=...)` with the following script:

```
´´´´markdown
README.md
<<<<<<< SEARCH
uv cync -U
=======
uv sync -U
>>>>>>> REPLACE
´´´´
```
*   **Tool Result:** A JSON object: `{"file_path": "README.md", "status": "success", ...}`

**AI's Final Response:**
> I have applied the edit to `README.md`. The tool reported success. The typo should now be fixed.

This multi-step process—exploring, investigating, and acting—is the foundation of agentic behavior in Polyglot AI. By
giving the AI the right tools and guiding it, you can accomplish complex tasks.
