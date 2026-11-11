# Agent Prototyping App

**Version 1.0 Beta** — a local-first agent prototyping workspace built with Python, FastAPI, SQLite, HTML, CSS, and vanilla JavaScript.

## Features

- Project-based chat sessions with rename and delete actions.
- Ollama and external OpenAI-compatible chat-completions models.
- Editable system prompt and model parameters.
- Markdown `skills.md` files loaded as agent instructions.
- MCP servers launched as child processes using stdio transport.
- MCP server lifecycle is controlled explicitly with Start and Stop. The Enabled checkbox only controls whether the agent may use that server’s tools; it never starts or stops the process.
- MCP tools can be inspected with their descriptions and input schemas; individual servers can be started, stopped, enabled, disabled, or deleted.
- Optional example usage per MCP server. Its **Try** control sends the example directly into the current chat.
- Tool calls are shown in expandable conversation entries with arguments, status, and results.
- Optional model reasoning display, when the selected model/provider exposes reasoning output (including Ollama thinking models).
- Dark and light themes, About dialog, and graceful application shutdown.
- SQLite persistence under `~/.agent-prototyping-app/` by default.

## Run

Python 3.11+ is required. On macOS/Linux:

```bash
chmod +x run.sh
./run.sh
```

The launcher creates a project-local `.venv`, installs dependencies into it, starts the application, and opens the browser when ready.

```bash
./run.sh start
./run.sh status
./run.sh stop
```

Default URL: <http://127.0.0.1:8750>

Each Python MCP server receives its own virtual environment under `~/.agent-prototyping-app/mcp-venvs/<server-id>/`. On first start, the app creates that environment and automatically installs dependencies from the server folder's `requirements.txt`, or installs an editable project from `pyproject.toml` / `setup.py`. If the source imports `fastmcp` and no manifest declares it, the app installs the `fastmcp` package automatically. The MCP log shows environment creation, pip output, and any dependency-installation errors. The application's own `.venv` is separate and is created by `run.sh` on first launch; it is intentionally not bundled in the ZIP. MCP command-line arguments and environment variables are optional plain-text fields. Enter arguments as you would in a terminal (for example, `server.py --port 8080`). Enter environment variables one per line in `KEY=value` format. Leave either field empty when it is not needed; no JSON editing is required.

## Adding an MCP server

1. Select **Add skill / MCP server** in the right panel, or open **Settings → Manage skills & MCP servers**.
2. Select MCP server and enter a display name, launch command, optional working directory, and optional arguments.
3. Optionally enter command-line arguments as ordinary text, environment variables one per line (`KEY=value`), and an example usage prompt. Both configuration fields may be left empty.
4. Select **Add configuration**. The application saves the MCP configuration without starting the process.
5. Use the separate **Start** control to launch the server, initialize the MCP session, and discover tools. Use **Stop** to stop the process.
6. The Enabled checkbox only adds or removes that server’s tools from the agent tool catalog; the process remains in its current running/stopped state.
7. Use **Try** beside a server to send its example usage into the chat immediately.

Examples of command/arguments combinations:

- `python` with `server.py`
- `node` with `./build/index.js`
- `npx` with `-y @modelcontextprotocol/server-filesystem /path/to/folder`
- `/absolute/path/to/executable` with `[]`

The working directory should point to the MCP project folder when using a relative script name such as `python app.py`; this is also where the app looks for `requirements.txt`, `pyproject.toml`, or `setup.py`.

## Notes

- Model reasoning is opt-in from the right-hand Model Reasoning section. Ollama receives `think: true`; actual reasoning output depends on model support. OpenAI-compatible providers display reasoning only if the endpoint returns a reasoning field.
- External models must expose an OpenAI-compatible `/chat/completions` endpoint.
- API credentials are stored locally in the application's SQLite settings database. Protect the local user account and application data directory accordingly.
- The app binds to `127.0.0.1` and is intended for local use.


## Appearance

The default appearance is dark graphite with blue gradient accents. Switch to the light theme from Settings if preferred.


### Ollama model selection and status
- The active model can be selected from the right-side model panel or from Settings.
- When Ollama is reachable, the app loads installed model names from `/api/tags`.
- The app refreshes Ollama availability every three seconds so chat availability and the status indicator follow the live service state.
- Start/Stop controls locate the Ollama executable on common macOS installation paths and verify the service state after sending a stop signal.


## Version 1.1.0 updates

- Dark workspace panels now use high-contrast blue surfaces and brighter text, including the left project/session list, chat messages/composer, and right-side agent prompt preview.
- Ollama status is checked directly against the configured `/api/tags` endpoint with proxy bypass for local services. The selected model is available in both Settings and the right-side Active Model selector, and the selector is refreshed when Settings opens.
- Ollama process inspection is more tolerant of macOS command-line quoting and restricted process visibility; the UI reports externally managed services without presenting a successful HTTP response as an API failure.
- MCP servers that announce a Streamable HTTP endpoint (such as FastMCP `http://0.0.0.0:8060/mcp`) are detected and connected over HTTP. The app now initializes the MCP session, discovers tools, invokes tools over HTTP, and keeps the live server process running. Existing stdio MCP servers continue to use stdio.


### v1.1.0 MCP behavior
- Saving an MCP configuration closes the dialog immediately and starts the server asynchronously in the background.
- FastMCP Streamable HTTP startup detection waits for the advertised endpoint, retries initialization, and disables inherited HTTP proxy settings for local MCP requests.
- A live HTTP MCP process is not terminated solely because its initial handshake fails; inspect Server logs or retry tool discovery.

## MCP transport configuration (v1.1.0)

Each MCP server can be configured independently with one of three transports:

- **Auto detect**: retain automatic transport detection from server startup output.
- **STDIO**: use the MCP JSON-RPC stdio protocol.
- **Streamable HTTP**: connect directly to the configured MCP endpoint URL (for example `http://127.0.0.1:8000/mcp`). The URL is an example only; configure the actual endpoint exposed by your server.

Transport and endpoint are saved per MCP server in the local SQLite configuration. They are not hardcoded to a particular server, hostname, port, or HR MCP implementation. Editing a server applies its changes and starts/reconnects it in the background so the dialog closes without waiting for dependency installation or protocol initialization.


### 1.1.5
- Replaced browser confirmation for conversation and project deletion with themed in-app confirmation dialogs.
- Moved new-conversation creation to a plus action beside each project name.
- Added independent expand/collapse controls for dynamically rendered projects.


### 1.1.6
- Refined project action icons: consistent sizing, alignment, hover/focus feedback, and a clean SVG plus icon.
