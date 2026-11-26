# Agent Prototyping App

A workspace for building, configuring, and experimenting with AI agents. It brings together LLM configuration, agent instructions, MCP tool integrations, multi-step agent execution, and conversation management in a single browser-based application.

<img width="1726" height="866" alt="Screenshot 2026-11-26 at 8 53 50 AM" src="https://github.com/user-attachments/assets/3a35e629-cdb3-4663-9ddf-2ef293c89390" />

Built with **Python, FastAPI, SQLite, HTML, CSS, and vanilla JavaScript**.

## Core Capabilities

- **Project-based workspace:** Organize work into projects, each containing independent chat sessions.
- **Multiple LLM providers:** Use locally hosted Ollama models or external services supporting the OpenAI-compatible Chat Completions API.
- **Agent configuration:** Customize system instructions, model parameters, reasoning options, and agent execution limits.
- **Agent workflow:** Support planning, model-directed execution, tool invocation, and completion verification.
- **MCP integrations:** Connect multiple Model Context Protocol servers and make their tools available to the agent.
- **Tool discovery:** Inspect available MCP tools, descriptions, input parameters, required fields, and JSON schemas.
- **Reusable instructions:** Manage prompts and import local skills to extend the agent's knowledge and behavior.
- **Execution visibility:** Inspect model interactions, tool calls, results, token usage, and execution timing.
- **Local persistence:** Store application configuration, projects, conversations, and MCP server definitions locally.
- **Customizable interface:** Switch between dark and light themes.

## Workspace

The application provides a three-panel workspace:

- **Projects and conversations:** Create and organize projects, start new conversations, and manage existing sessions.
- **Chat workspace:** Interact with the configured agent, review responses, and inspect execution details.
- **Configuration panel:** Manage the active model, generation parameters, agent settings, prompts, skills, and MCP servers.

Conversations are maintained independently, allowing different tasks and experiments to be organized without mixing their histories.

## LLM Providers and Models

The application supports two provider categories.

### Ollama

Connect to a local Ollama service and use models installed on that service.

- Configure the Ollama base URL.
- Discover and select installed models.
- Monitor Ollama service availability.
- Start or stop the local Ollama service through the application when supported by the environment.
- Enable model reasoning for models that expose thinking output.

### OpenAI-Compatible Providers

Connect to external or self-hosted services exposing an OpenAI-compatible Chat Completions endpoint.

Configure the service endpoint, model name, and API credentials.

### Model Parameters

Configure the generation parameters exposed by the selected provider, including:

- Temperature
- Top P
- Top K
- Context length
- Repeat penalty
- Maximum output tokens

<img width="376" height="370" alt="Screenshot 2026-11-26 at 8 54 42 AM" src="https://github.com/user-attachments/assets/c0d26e8a-88d3-416e-864e-7315e8df94a2" />


The application also supports conversation-history limits, allowing the agent to use all previous messages or a selected number of recent messages as context.

Model reasoning can be enabled separately. Its availability and output depend on the selected model and provider.

## Agent Architecture

The application provides a generic, model-driven agent workflow. Rather than implementing fixed business processes or hardcoded tool sequences, it allows the model to determine how a task should be approached and executed.

When the agent architecture is enabled, a run can involve three main stages.

### 1. Planning

The model analyzes the user's request and produces a task plan containing objectives, acceptance criteria, and dependencies.

The plan provides a structured representation of the requested task and a basis for subsequent execution and verification.

### 2. Execution

The model performs the task using its available capabilities.

It can:

- Generate a direct response when no external action is needed.
- Select and invoke available MCP tools.
- Issue multiple tool calls in the same model response.
- Examine tool results and determine whether additional actions are required.
- Continue execution until it has sufficient information to produce a candidate answer.

The model controls tool selection, arguments, ordering, and subsequent actions. The application executes native structured tool calls and returns their results to the model.

Tool arguments are validated against the input schemas advertised by the corresponding MCP servers. Ordinary assistant text containing JSON or proposed tool invocations is not interpreted as an executable tool call.

### 3. Verification

The model evaluates the candidate result against the original request, task objectives, execution history, and available evidence.

Verification checks whether the required objectives have been satisfied and whether the result is adequately supported.

If required work remains incomplete, feedback is returned to the execution stage so the model can continue working.

The application enforces bounded verification attempts and does not treat an invalid or unsupported verification response as successful completion.

Planning and verification can be enabled or disabled through Agent Settings.

## Agent Settings and Execution Limits

The Agent Settings panel exposes the controls that govern the agent's execution architecture and resource limits.

Available settings include:

| Setting | Default |
|---|---:|
| Planning and verification | Enabled |
| Maximum total LLM calls per run | 30 |
| Maximum execution iterations | 12 |
| Maximum native tool calls | 20 |
| Maximum planning attempts | 2 |
| Maximum verification cycles | 2 |
| Maximum verifier attempts per run | 4 |
| Maximum verifier attempts per cycle | 2 |
| Maximum consecutive tool errors | 4 |

These limits are enforced by the application across the complete agent run. They prevent uncontrolled execution loops and bound the number of model and tool invocations.

<img width="388" height="573" alt="Screenshot 2026-11-26 at 8 55 36 AM" src="https://github.com/user-attachments/assets/3eedfdf0-9e3b-4608-823d-fbe936b9844b" />

Settings are validated and persisted locally. Changes apply to subsequent agent runs.

## Prompts and Skills

The application supports reusable instructions that can be selected for individual conversations.

### Prompt Library

Manage custom prompts through the application:

- Create prompts.
- Edit existing prompts.
- Enable or disable prompts.
- Delete prompts.
- Select the prompts to use in a conversation.

Custom prompts are opt-in and are not automatically selected when created.

<img width="389" height="732" alt="Screenshot 2026-11-26 at 8 56 18 AM" src="https://github.com/user-attachments/assets/11125fc4-1719-43f9-baf2-72cda04ea7aa" />

The application's baseline system instruction is maintained separately from the prompt library.

### Local Skills

Import local skill directories to provide the agent with additional instructions and supporting resources.

A skill directory can contain:

- `SKILL.md` — the skill's primary instructions.
- `references/` — supporting documentation.
- `prompts/` — additional prompt resources.

Skills can be selected for the relevant conversations, allowing different agent capabilities and instructions to be used for different tasks.

## MCP Servers and Tools

The application integrates with external capabilities through the **Model Context Protocol (MCP)**.

Multiple MCP servers can be configured independently. Their advertised tools become available to the agent according to the server connection and enablement settings.

### Supported Transports

- **STDIO:** Communicate with a locally launched MCP server through standard input and output.
- **Streamable HTTP:** Connect to an MCP server through its HTTP endpoint.
- **Auto Detect:** Detect the transport from the server's startup information.

Remote HTTP connections can use optional custom HTTP headers when authentication or other endpoint-specific configuration is required.

### Server Configuration

Each MCP server can be configured with:

- Display name.
- Launch command.
- Command-line arguments.
- Working directory.
- Transport type.
- HTTP endpoint.
- Environment variables.
- HTTP headers, where applicable.
- Optional example usage prompt.

The application supports both locally managed MCP processes and remote MCP endpoints.

<img width="814" height="788" alt="Screenshot 2026-11-26 at 8 57 00 AM" src="https://github.com/user-attachments/assets/242744e1-1cb2-4e91-bcf6-4dbab6d3a315" />

### Server Lifecycle and Availability

MCP server process management and agent availability are independent controls.

- **Start:** Launch a locally managed server and establish its MCP connection.
- **Stop:** Stop the locally managed server process.
- **Enable/Disable:** Control whether the server's tools are included in the agent's available tool catalog.

Disabling a server does not automatically stop its process.

### MCP Tools Catalog

Each MCP server provides a Tools action that opens a dedicated dialog displaying its available tools.

The catalog presents:

- Tool names.
- Tool descriptions.
- Input parameters.
- Required fields.
- Expandable input JSON schemas.

<img width="853" height="806" alt="Screenshot 2026-11-26 at 8 57 41 AM" src="https://github.com/user-attachments/assets/fc1b6196-a14c-40e5-8531-a25cbdd2b4e7" />


The application discovers tools from connected servers and refreshes the available tool definitions before agent requests.

### Example Usage

An optional example prompt can be associated with an MCP server. Its Try action sends the example directly into the current conversation, providing a convenient way to experiment with the server's capabilities.

### Python MCP Server Environments

For locally launched Python MCP servers, the application can create an independent virtual environment for each server:

`~/.agent-prototyping-app/mcp-venvs/<server-id>/`

On startup, dependencies can be installed from the server project's `requirements.txt`, or the project can be installed in editable mode when `pyproject.toml` or `setup.py` is available.

If the server imports `fastmcp` without declaring it in a dependency manifest, the application can install the package automatically.

The working directory should point to the MCP project folder when using relative script paths.

Command-line arguments are entered as ordinary terminal arguments, while environment variables use one `KEY=value` entry per line.

## Execution Inspection and Monitoring

The application provides visibility into the agent's execution process without requiring users to inspect backend logs.

Each assistant turn can expose an expandable execution trace showing the sequence of model interactions and tool operations.

Depending on provider capabilities, the trace can include:

- Planning, execution, and verification stages.
- Model requests and responses.
- Selected MCP servers and exposed tool definitions.
- Native structured tool calls and their arguments.
- Tool execution status and returned results.
- Token consumption.
- Model and tool execution durations.
- Ollama-specific generation and evaluation timing information.

Model reasoning can also be displayed when enabled and supported by the selected provider.

These inspection capabilities help users understand agent behavior, troubleshoot integrations, and evaluate how the agent approaches a task.

## Local Storage and Data

The application uses SQLite for persistent local storage.

By default, application data is stored under:

`~/.agent-prototyping-app/`

This directory contains application settings, model configuration, credentials, projects, conversations, and MCP server configuration.

The application is designed for local use and binds to `127.0.0.1` by default.

API credentials and other sensitive configuration values are stored locally. Users should protect their application data directory and local operating-system account.

## Running the Application

### Requirements

- Python 3.11 or later.
- macOS or Linux.
- Ollama or an OpenAI-compatible provider, depending on the models being used.
- MCP servers for any external tools the agent needs.

### Start

```bash
chmod +x run.sh
./run.sh
```

The launcher creates a project-local Python virtual environment, installs the required dependencies, starts the application, and opens the browser when ready.

### Process Management

```bash
./run.sh start
./run.sh status
./run.sh stop
```

### Default URL

<http://127.0.0.1:8750>

The application's virtual environment is created locally on first launch and is not included in the source package.

## Appearance

The application uses a dark graphite interface with blue accents by default. A light theme is also available through Settings.

---

**Agent Prototyping App — Version 1.5.9**
