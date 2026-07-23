## 📘 MODEL CONTEXT PROTOCOL (MCP) — COMPLETE STUDY NOTES

**A Professional Reference for Interviews, Revision, and Building MCP Projects**
Compiled from playlist study + official MCP documentation, FastMCP docs, and current (2026) ecosystem research.

---

## 📑 TABLE OF CONTENTS

1. Introduction to MCP
2. Why MCP Is Needed (The M×N Problem)
3. MCP Architecture
4. The Complete MCP Lifecycle
5. Connecting MCP to Claude Desktop
6. Building Local MCP Servers (FastMCP)
7. Building Remote MCP Servers
8. Testing MCP Servers (MCP Inspector)
9. Building MCP Clients
10. Security, Auth & Best Practices
11. Production, Deployment & Performance
12. Terminology / Glossary
13. Cheat Sheet
14. Interview Questions
15. Common Mistakes & Troubleshooting Guide
16. FAQ
17. Key Takeaways & Revision Notes

---

## 1. 🧩 INTRODUCTION TO MCP

### 1.1 What is MCP?

Model Context Protocol (MCP) is an open, vendor-neutral standard, originally released by Anthropic in November 2024, that defines a common way for AI applications ("hosts," like Claude Desktop, Claude Code, ChatGPT, Cursor, VS Code) to discover and use external tools, data sources, and prompt templates exposed by lightweight servers.

Analogy: MCP is often called "USB-C for AI." Just as USB-C lets any device plug into any USB-C-compliant cable regardless of manufacturer, MCP lets any AI model plug into any compliant tool/data server without needing a bespoke integration.

### 1.2 Definition (Formal)

MCP is a protocol built on JSON-RPC 2.0 that standardizes communication between:
- A Host (the AI application, e.g., Claude Desktop)
- A Client (embedded inside the host, manages one 1:1 connection to a server)
- A Server (exposes Tools, Resources, and Prompts to the client)

### 1.3 Why It Exists

Before MCP, every AI application that wanted to talk to an external system (GitHub, Slack, Postgres, a filesystem, etc.) needed a custom, one-off integration. This didn't scale as the number of models and the number of tools both grew. MCP decouples "who built the model" from "who built the tool."

### 1.4 Real-World Applications & Use Cases

| Use Case | Example |
|---|---|
| Developer tooling | Claude Code / Cursor reading & editing a codebase via a filesystem/git MCP server |
| Data access | Querying a PostgreSQL/MySQL database in natural language |
| SaaS integration | Reading/writing Slack messages, Notion pages, Jira tickets |
| Enterprise agents | Agentic workflows that call internal company APIs safely |
| Personal automation | Managing Google Drive, Gmail, Calendar |
| Search & retrieval | Web search, document/RAG-style retrieval servers |

By 2026, MCP is supported not only by Anthropic's products but also by OpenAI (ChatGPT, Agents SDK), Google DeepMind (Gemini), Microsoft (Copilot Studio), and thousands of community-built servers, making it the de facto standard for agentic AI tool connectivity.

Note: MCP does not replace REST APIs or GraphQL. It is a standardized wrapper that makes existing APIs consumable by LLMs in a predictable, discoverable way.

---

## 2. ❓ WHY MCP IS NEEDED — THE M×N PROBLEM

### 2.1 The Problem With Traditional Integrations

Without a shared standard:
- M = number of AI applications/models
- N = number of tools/data sources

Every model needing every tool requires a custom connector → M × N integrations to build and maintain.

Without MCP (M×N problem):
Model A → custom → Tool 1
Model A → custom → Tool 2
Model B → custom → Tool 1
Model B → custom → Tool 2
Model C → custom → Tool 1
Model C → custom → Tool 2
3 models × 2 tools = 6 bespoke integrations

With MCP (M+N problem):
Model A ┐
Model B ┼→ [MCP Protocol] → Tool 1 (MCP Server)
Model C ┘               └→ Tool 2 (MCP Server)
3 models + 2 tools = 5 total integration points (each side only integrates with MCP, once)

### 2.2 Key Benefits Over Custom APIs

| Benefit | Explanation |
|---|---|
| Standardization | One protocol, one schema shape (JSON-RPC 2.0), for every tool |
| Scalability | Adding a new tool doesn't require touching every model integration |
| Reduced maintenance | Tool authors maintain one server; model vendors maintain one client |
| Vendor independence | A server built for Claude also works with ChatGPT, Cursor, etc. |
| Discoverability | Clients can query a server at runtime to find out what it can do (no hardcoding) |
| Composable ecosystem | Community can publish reusable servers (GitHub, Slack, Postgres...) |

Warning: MCP is not a replacement for good API design on the backend. A poorly designed underlying API, wrapped in MCP, is still a poorly designed API — MCP just standardizes how the LLM talks to it.

---

## 3. 🏗️ MCP ARCHITECTURE

### 3.1 Core Components

| Component | Role |
|---|---|
| Host | The end-user AI application (Claude Desktop, Claude Code, an IDE, a custom agent app). Owns the UI and the LLM. |
| Client | Lives inside the Host. Maintains a 1:1 stateful connection to exactly one Server. A Host can run multiple Clients (one per Server). |
| Server | A (usually lightweight) program that exposes capabilities — Tools, Resources, Prompts — over the protocol. |
| Tools | Model-invoked functions (actions) — e.g., create_github_issue, send_email. |
| Resources | Read-only, addressable data — e.g., a file's contents, a DB schema, a doc. Identified by a URI. |
| Prompts | Reusable prompt templates with parameters, often exposed as slash-commands in a host UI. |
| Registry / Discovery | The mechanism (tools/list, resources/list, prompts/list) by which a client learns what a server offers — no hardcoding needed. |

### 3.2 Architecture Diagram (text form)

HOST (Claude Desktop / Claude Code / Custom App)
  MCP Client 1 —JSON-RPC 2.0→ MCP Server 1 (Filesystem) → Local Files
  MCP Client 2 —JSON-RPC 2.0→ MCP Server 2 (GitHub) → GitHub API
  MCP Client 3 —JSON-RPC 2.0→ MCP Server 3 (Postgres) → Database

Flow in words: Host → Client (1 per server) → JSON-RPC 2.0 message over a Transport → Server → underlying system (file, API, DB) → Tool/Resource result → back up the chain → inserted into the model's context.

### 3.3 Communication Flow (Request → Response)

1. Host wants to call a tool (because the LLM decided to, or a Resource needs loading).
2. Client sends a JSON-RPC request object (method, params, id) to the Server over the transport.
3. Server executes the corresponding function.
4. Server sends back a JSON-RPC response object (result or error, matching id).
5. Client hands the result back to the Host, which inserts it into the model's context window.

### 3.4 JSON-RPC 2.0 — The Wire Format

MCP is transport-agnostic but always uses JSON-RPC 2.0 as its message format.

Example Request (tool call):
{
  "jsonrpc": "2.0",
  "id": 7,
  "method": "tools/call",
  "params": { "name": "get_weather", "arguments": { "city": "Delhi" } }
}

Example Response (success):
{
  "jsonrpc": "2.0",
  "id": 7,
  "result": { "content": [{ "type": "text", "text": "It is 34°C and sunny in Delhi." }], "isError": false }
}

Example Response (error):
{
  "jsonrpc": "2.0",
  "id": 7,
  "error": { "code": -32602, "message": "Invalid params: 'city' is required" }
}

### 3.5 Transport Layer

As of the current (2025-11-25) spec, two transports are first-class, and one is deprecated:

| Transport | Status | Use Case |
|---|---|---|
| stdio | Current | Local process communication over stdin/stdout. Zero network, zero auth overhead. Best for Claude Desktop, local dev tools. |
| Streamable HTTP | Current | Single /mcp HTTP endpoint; supports plain JSON or a streamed text/event-stream response. Best for remote/cloud/multi-user servers. |
| HTTP+SSE (legacy) | Deprecated (since spec 2025-03-26) | Original remote transport — two separate endpoints (a long-lived GET /sse stream + POST /messages). Kept only for backward compatibility. |

Important 2026 update: The MCP spec is undergoing its largest revision yet. A release candidate (2026-07-28) introduces a fully stateless protocol core, an Extensions framework, a Tasks extension (for long-running work), MCP Apps (interactive UIs rendered inside a host, built on mcp-ui), tighter OAuth/OIDC-aligned authorization, and a formal deprecation policy. The final spec is scheduled to ship July 28, 2026. Always check modelcontextprotocol.io for the current version before building.

### 3.6 Local vs Remote MCP Servers

| | Local Server (stdio) | Remote Server (Streamable HTTP) |
|---|---|---|
| Runs where | As a subprocess on the user's machine | On a remote host / cloud, behind a URL |
| Auth | None needed (process boundary = trust boundary) | OAuth 2.1 / API keys typically required |
| Multi-user | No — one client per process | Yes — many clients can connect |
| Latency | ~0ms (in-process pipe) | Network latency applies |
| Setup in Claude Desktop | Add command + args to config JSON | Add as a Custom Connector in Settings, or bridge via mcp-remote |
| Best for | Filesystem access, local scripts, personal automation | SaaS integrations (Slack, GitHub, Notion), shared team tools |

Common gotcha (2026): Claude Desktop's claude_desktop_config.json schema historically only validates stdio server entries. Pasting a raw url/type: "http" block (copied from a "remote MCP" doc) into that file will often be silently dropped or crash the app. For remote servers, either register them as a Custom Connector through Settings, or wrap the URL using the mcp-remote stdio-bridge package.

### 3.7 How Claude Desktop Connects to MCP

1. Claude Desktop reads claude_desktop_config.json on startup.
2. For every entry under "mcpServers", it spawns a subprocess using the given command + args (stdio transport).
3. It performs the initialization handshake (see Lifecycle, below) with each server.
4. Discovered Tools/Resources/Prompts become available to the model and to the user (e.g., as the tools icon or / slash commands in the UI).
5. You must fully quit and reopen Claude Desktop after editing the config — it does not hot-reload.

---

## 4. 🔄 THE COMPLETE MCP LIFECYCLE

Stages: 1. Connect (spawn/dial) → 2. Initialize handshake → 3. Capability negotiation → 4. Discovery (list tools/resources) → 5. Invoke tools/resources → 6. Response / error handling → 7. Session teardown / shutdown

### Stage-by-Stage Explanation

1. Client Initialization / Connection
The Host launches (stdio) or dials (HTTP) the Server. A transport-level channel now exists, but no protocol messages have been exchanged yet.

2. Handshake (initialize)
The Client sends an initialize request containing its protocol version and its own capabilities. The Server responds with its protocol version, capabilities, and server info (name/version).

Example:
// Client → Server
{
  "jsonrpc": "2.0", "id": 1, "method": "initialize",
  "params": {
    "protocolVersion": "2025-11-25",
    "capabilities": { "roots": {}, "sampling": {} },
    "clientInfo": { "name": "claude-desktop", "version": "1.0" }
  }
}

3. Capability Negotiation
Both sides now know what the other supports (e.g., does the server support resources/subscribe? Does the client support sampling?). The Client then sends an initialized notification to confirm it's ready.

4. Tool / Resource / Prompt Discovery
The Client calls tools/list, resources/list, and prompts/list to fetch the full catalog of what the Server offers, including JSON Schemas for each tool's parameters.

5. Tool Invocation
When the LLM decides to use a tool, the Client sends tools/call with the tool name and arguments. Resources are fetched via resources/read. Prompts are fetched via prompts/get.

6. Response Generation & Error Handling
The Server executes the underlying logic and returns a result (with isError: true/false) or a JSON-RPC-level error object (for protocol-level failures like unknown method or bad params).

7. Session Lifecycle & Shutdown
For stdio: the session ends when the process is killed (host closes / user quits app). For HTTP: sessions may be tracked via a session ID header and can be explicitly terminated. A clean shutdown releases file handles, DB connections, or other resources the server was holding.

Tip: During discovery, a good tool docstring and precise JSON Schema are what let the LLM correctly decide when and how to call the tool — this is the single highest-leverage thing to get right when authoring a server.

---

## 5. 🔌 CONNECTING MCP TO CLAUDE DESKTOP

There are two common methods:

### Method 1 — Edit the Config JSON Directly

Location of the config file:
- macOS: ~/Library/Application Support/Claude/claude_desktop_config.json
- Windows: %APPDATA%\Claude\claude_desktop_config.json

Example config (stdio, Python server via uv):
{
  "mcpServers": {
    "notes-server": {
      "command": "uv",
      "args": ["--directory", "/absolute/path/to/notes-server", "run", "server.py"]
    }
  }
}

Steps:
1. Open the file (create it if it doesn't exist).
2. Add an entry under "mcpServers" with a unique key (server name).
3. Save the file.
4. Fully quit Claude Desktop (not just close the window) and reopen it.
5. Verify the server loaded — check the tools icon in the chat box.

### Method 2 — Install via FastMCP CLI + uv

uv is a fast, modern Python package/project manager (written in Rust) that has become the de facto standard for MCP projects because of its speed and built-in virtual-env handling.

# 1. Install uv (macOS/Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell, as Administrator)
powershell -c "irm https://install.python-uv.org | iex"

# 2. Initialize a new project
uv init notes-server
cd notes-server

# 3. Create & sync a virtual environment
uv venv
uv sync

# 4. Add FastMCP as a dependency
uv add fastmcp
# or, if you want the CLI/inspector tools too:
uv add "mcp[cli]"

# 5. Install your server directly into Claude Desktop's config
uv run fastmcp install claude-desktop server.py

The fastmcp install claude-desktop command automatically writes the correct entry into claude_desktop_config.json for you — no manual JSON editing required. This is the recommended path for beginners because it avoids path/quoting mistakes.

Best Practice: Prefer Method 2 (fastmcp install) during learning/development — it removes an entire class of "my config JSON is malformed" bugs. Use Method 1 (manual JSON) when you need fine control (custom env vars, non-Python servers, remote connector URLs).

---

## 6. 🛠️ BUILDING LOCAL MCP SERVERS (FastMCP)

### 6.1 What is FastMCP?

FastMCP is a Pythonic, decorator-based framework for building MCP servers and clients with minimal boilerplate. FastMCP 1.0 was folded into the official mcp Python SDK; the actively developed FastMCP 2/3 project (by Jeremiah Lowin / Prefect) is now the community standard, reportedly powering roughly 70% of all MCP servers across languages. FastMCP 3.0 (Jan 2026) added component versioning, authorization controls, and OpenTelemetry integration.

### 6.2 Project Setup

uv init weather-server
cd weather-server
uv venv
uv add fastmcp

Recommended folder structure:

<img width="432" height="244" alt="image" src="https://github.com/user-attachments/assets/9ecf249c-0571-4a58-9354-cfb010f49c56" />

### 6.3 Defining & Registering Tools

# server.py
from fastmcp import FastMCP

mcp = FastMCP("Weather Server")

@mcp.tool
def get_alerts(state: str) -> str:
    """Get active weather alerts for a US state.
    Args:
        state: Two-letter US state code, e.g. 'CA', 'NY'
    """
    return f"No active alerts for {state}."

@mcp.tool
def get_forecast(city: str, days: int = 3) -> str:
    """Get an N-day weather forecast for a city."""
    return f"{days}-day forecast for {city}: sunny throughout."

if __name__ == "__main__":
    mcp.run()  # defaults to stdio transport

Note: FastMCP auto-generates the JSON Schema for each tool from the type hints, and surfaces the docstring to the model as the tool description. Precise type hints + a clear docstring = better tool-selection accuracy by the LLM.

### 6.4 Defining Resources

@mcp.resource("config://app-version")
def get_app_version() -> str:
    """Static resource: current app version."""
    return "1.4.2"

@mcp.resource("notes://{topic}")
def get_note(topic: str) -> str:
    """Dynamic resource: fetch a note by topic name."""
    return read_note_from_disk(topic)

### 6.5 Defining Prompts

@mcp.prompt
def summarize_notes(topic: str, style: str = "concise") -> str:
    """Reusable prompt template for summarizing notes on a topic."""
    return f"Summarize my {topic} notes in a {style} style, using bullet points."

### 6.6 Running & Testing the Server

# Run directly
uv run server.py

# Run with the MCP Inspector (interactive debug UI)
uv run mcp dev server.py

---

## 7. 🌐 BUILDING REMOTE MCP SERVERS

### 7.1 Switching to Streamable HTTP

from fastmcp import FastMCP

mcp = FastMCP("Remote Notes Server")

@mcp.tool
def add_note(text: str) -> str:
    """Add a note to the remote store."""
    save_note(text)
    return "Note saved."

if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8000)

This exposes a single HTTP endpoint (conventionally /mcp) that accepts JSON-RPC requests via POST, and can respond as plain application/json or stream progress as text/event-stream, depending on the request.

### 7.2 Remote Communication & Deployment Concepts

| Concept | Explanation |
|---|---|
| Statelessness (2026 direction) | The upcoming spec revision moves the protocol core to be stateless, so it scales cleanly on ordinary HTTP infrastructure (load balancers, autoscaling) without sticky sessions. |
| Authentication | Remote servers typically require OAuth 2.1 (authorization code + PKCE) or API keys; never rely on the transport alone for security. |
| Session IDs | Streamable HTTP servers may issue a session identifier (header) to correlate a stream of requests from one client. |
| Reverse proxy / gateway | In production, put the server behind Nginx/Caddy/an API gateway for TLS termination, rate limiting, and logging. |
| Containerization | Package the server as a Docker image for reproducible deployment (Fly.io, Render, AWS ECS, Cloud Run, etc.). |

### 7.3 Bridging for Clients That Only Speak stdio

Some hosts (notably older Claude Desktop builds) only validate stdio entries in their config. To use a remote HTTP server from such a host:

npx mcp-remote https://your-server.example.com/mcp

mcp-remote runs locally as a stdio process and forwards everything to the remote URL — effectively a protocol bridge.

### 7.4 Testing a Remote Server

uv run mcp dev server.py --transport streamable-http
# or point the Inspector directly at a running URL
npx @modelcontextprotocol/inspector http://localhost:8000/mcp

---

## 8. 🔍 TESTING MCP SERVERS — THE MCP INSPECTOR

### 8.1 What is the Inspector?

The MCP Inspector is the official interactive debugging UI/tool for MCP servers — think of it as "Postman for MCP." It lets you exercise a server without needing an LLM in the loop.

npx @modelcontextprotocol/inspector
# or, for a Python/uv project:
uv run mcp dev server.py

### 8.2 Features

- Lists all discovered Tools, Resources, and Prompts with their schemas.
- Auto-generates a form UI for each tool's parameters so you can invoke it manually.
- Shows the raw JSON-RPC traffic (requests and responses) for debugging.
- Live log/console panel for anything your server prints to stderr.
- Works with both stdio and HTTP-based servers.

### 8.3 Debugging Workflow

1. Start the Inspector against your server.
2. Open the Tools tab → confirm your tool appears with the correct name/description/params.
3. Fill in sample parameters → click Run Tool → inspect the live output.
4. If something's wrong, check the Logs panel and the raw JSON-RPC pane simultaneously.
5. Fix your code, Ctrl-C, restart the Inspector, and repeat.

### 8.4 Common Issues Found via Inspector

| Symptom | Likely Cause |
|---|---|
| Tool doesn't appear at all | Function not decorated with @mcp.tool, or server crashed on startup |
| Tool appears but arguments are wrong type | Missing/incorrect type hints |
| "Tool not found" error on call | Name mismatch between tools/list and tools/call |
| Server hangs | Blocking synchronous call inside an async tool; missing await |
| Works in Inspector, fails in Claude Desktop | Absolute path issues, wrong working directory, or missing env vars in the desktop config |

---

## 9. 🧑‍💻 BUILDING MCP CLIENTS

### 9.1 Client Architecture

An MCP Client is the piece embedded in a Host that:
1. Establishes a transport connection to one server.
2. Performs the initialize handshake.
3. Exposes methods to list and call tools/resources/prompts.
4. Feeds results back into the Host's LLM loop.

### 9.2 Connecting to a Local (stdio) Server

import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

server_params = StdioServerParameters(command="uv", args=["run", "server.py"])

async def main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print("Available tools:", [t.name for t in tools.tools])
            result = await session.call_tool("get_forecast", arguments={"city": "Delhi", "days": 3})
            print(result.content)

asyncio.run(main())

### 9.3 Connecting to a Remote (HTTP) Server

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

async def main():
    async with streamablehttp_client("http://localhost:8000/mcp") as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("add_note", {"text": "Learned MCP today!"})
            print(result.content)

### 9.4 Complete Client Workflow

Connect → initialize() → list_tools()/list_resources()/list_prompts() → call_tool(name, args) → handle result/error → close session

Tip: In real agent frameworks (LangChain, LangGraph, custom loops), the client's call_tool is typically wrapped so the LLM's function-calling output maps directly onto an MCP tools/call — this is exactly how Claude Desktop, Claude Code, and LangChain's MCP adapters work under the hood.

---

## 10. 🔐 SECURITY, AUTH & BEST PRACTICES

| Practice | Why |
|---|---|
| Store secrets in environment variables, never in tool schemas/code | Prevents leaking API keys to the model or to logs |
| Validate & sanitize all tool inputs | LLMs can pass unexpected/malicious values |
| Use least privilege for any DB/API credentials the server holds | Limits blast radius if the server is compromised |
| Prefer OAuth 2.1 + PKCE for remote servers | Aligns with the 2026 spec's authorization hardening |
| Never expose destructive tools (e.g., DROP TABLE) without explicit confirmation flows | LLM tool-calls can be probabilistic; guard rails matter |
| Log tool invocations (without leaking secrets) | Needed for audits & debugging in production |
| Pin dependency versions (uv.lock) | Reproducible builds; avoids "worked yesterday" bugs |
| Use HTTPS/TLS for all remote transports | Protects JSON-RPC payloads in transit |

---

## 11. 🚀 PRODUCTION, DEPLOYMENT & PERFORMANCE

- Use uv in CI/CD for fast, reproducible installs (uv sync --frozen).
- Containerize servers with Docker; keep images slim (multi-stage builds).
- Observability: FastMCP 3.0 ships OpenTelemetry integration — wire this into your existing tracing stack.
- Stateless design: favor stateless tool handlers so servers can be horizontally scaled behind a load balancer (this is also the direction of the 2026-07-28 spec).
- Timeouts & retries: wrap outbound calls (to DBs/APIs) with sane timeouts; surface failures as MCP isError: true results rather than crashing the process.
- Versioning: version your tools/schemas so you can evolve a server without breaking existing clients (component versioning, new in FastMCP 3.0).
- Rate limiting: put a gateway in front of remote servers shared across many users.

---

## 12. 📖 GLOSSARY / TERMINOLOGY

| Term | Meaning |
|---|---|
| Host | The AI application the user interacts with |
| Client | The connector living inside the Host, 1:1 with a Server |
| Server | Program exposing Tools/Resources/Prompts |
| Tool | An invokable, model-callable function (an "action") |
| Resource | Read-only, URI-addressable data |
| Prompt | Reusable, parameterized instruction template |
| JSON-RPC 2.0 | The message format MCP uses for all requests/responses |
| Transport | The channel carrying JSON-RPC bytes: stdio / Streamable HTTP / (legacy) SSE |
| Handshake | The initialize / initialized exchange at connection start |
| Capability negotiation | Both sides declaring what protocol features they support |
| Discovery | list calls (tools/list, etc.) that reveal a server's offerings |
| FastMCP | Pythonic decorator-based framework for building MCP servers/clients |
| uv | Fast Rust-based Python package/project manager |
| MCP Inspector | Official GUI tool for testing/debugging MCP servers |
| mcp-remote | Node helper that bridges a remote HTTP MCP server into a local stdio interface |
| MCP Apps | 2026 extension standardizing interactive UIs rendered by a host from server output |
| SEP | Spec Enhancement Proposal — how changes to the MCP spec get proposed |

---

## 13. ⚡ CHEAT SHEET

Core commands:
uv init myserver && cd myserver     # scaffold project
uv venv && uv sync                  # create + sync venv
uv add fastmcp                      # add FastMCP
uv run server.py                    # run the server (stdio)
uv run mcp dev server.py            # run with Inspector attached
npx @modelcontextprotocol/inspector # standalone inspector
npx mcp-remote <url>                # bridge remote HTTP server to stdio

Minimal server:
from fastmcp import FastMCP
mcp = FastMCP("Demo")

@mcp.tool
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

if __name__ == "__main__":
    mcp.run()

Comparison Tables:

STDIO vs Streamable HTTP vs SSE (legacy)
| | stdio | Streamable HTTP | SSE (legacy) |
|---|---|---|---|
| Scope | Local only | Remote/multi-user | Remote (deprecated) |
| Auth | None (process boundary) | OAuth 2.1 / API key | OAuth (older model) |
| Status | Current | Current | Deprecated since 2025-03-26 |
| Endpoint(s) | n/a (stdin/stdout) | Single /mcp | Two (/sse + /messages) |

Tool vs Resource vs Prompt
| | Tool | Resource | Prompt |
|---|---|---|---|
| Analogy | POST endpoint | GET endpoint | Reusable template |
| Triggered by | Model decision | Host/model pull | User (slash command) or model |
| Side effects | Yes (can act) | No (read-only) | No |

Local vs Remote Server
| | Local | Remote |
|---|---|---|
| Transport | stdio | Streamable HTTP |
| Auth needed | No | Usually yes |
| Multi-user | No | Yes |

MCP vs REST APIs vs Function Calling
| | MCP | Plain REST API | Native Function Calling |
|---|---|---|---|
| Standardized across models? | Yes | No (custom per integration) | No (Model/vendor-specific schema) |
| Discovery at runtime? | Yes (list methods) | No (needs docs/OpenAPI) | No (must hardcode schema) |
| Built for LLMs specifically? | Yes | General purpose | Yes, but not portable |

uv vs pip
| | uv | pip |
|---|---|---|
| Language | Rust (fast) | Python |
| Lockfile | uv.lock (built-in) | Needs pip-tools/poetry |
| Venv management | Built-in (uv venv) | Separate (venv/virtualenv) |
| Speed | Very fast (parallel resolver) | Slower |

---

## 14. 🎯 INTERVIEW QUESTIONS

1. What is MCP and why was it created?
→ A JSON-RPC-2.0-based open protocol standardizing how AI hosts connect to external tools/data, solving the M×N integration problem.

2. Explain the difference between a Tool, a Resource, and a Prompt.
→ Tool = callable action with side effects; Resource = read-only addressable data; Prompt = reusable parameterized template.

3. What transports does MCP support, and which is deprecated?
→ stdio (local) and Streamable HTTP (remote) are current; legacy HTTP+SSE is deprecated since spec 2025-03-26.

4. Walk through the MCP lifecycle from connection to shutdown.
→ Connect → initialize handshake → capability negotiation → discovery (list calls) → invocation (call) → response/error handling → session teardown.

5. Why is stdio not suitable for multi-user/remote scenarios?
→ It's a 1:1 process-to-process pipe with no built-in auth or network layer — only one client can talk to one server process.

6. What role does JSON Schema play in MCP tools?
→ It's auto-derived (in FastMCP) from type hints and tells the client/model exactly what arguments a tool expects, enabling validation and correct model tool-selection.

7. How would you debug a tool that isn't showing up in Claude Desktop?
→ Test first with the MCP Inspector to isolate server-side issues; then check the desktop config JSON for path/transport mismatches; confirm the app was fully restarted.

8. What is mcp-remote used for?
→ A Node-based stdio bridge that lets stdio-only clients (like some Claude Desktop versions) talk to a remote Streamable HTTP MCP server.

9. Why is uv preferred over pip for MCP projects?
→ Much faster dependency resolution, built-in virtual-env + lockfile management, and it's become the de facto tool used by FastMCP's own install/CLI commands.

10. What are MCP Apps?
→ A 2026 official extension (built on mcp-ui) that lets servers return interactive UI (dashboards, forms) rendered directly inside the host, beyond plain text/structured data.

11. Does MCP replace REST APIs or native function calling?
→ No — it standardizes how an LLM accesses tools/data; the underlying REST/GraphQL APIs still exist and MCP servers typically wrap them.

12. What security practices matter most when building an MCP server?
→ Env-var secrets, input validation, least-privilege credentials, OAuth 2.1 for remote servers, and guarding destructive tools behind confirmation.

---

## 15. 🐛 COMMON MISTAKES & TROUBLESHOOTING GUIDE

| Mistake | Fix |
|---|---|
| Hardcoding absolute local file paths in resources (file:///Users/you/...) | Use relative/parameterized paths or environment-based config |
| Forgetting to fully quit & restart Claude Desktop after config edits | Config is read only at startup — always fully quit, not just close window |
| Pasting an HTTP url/type: "http" block into claude_desktop_config.json | Use a Custom Connector or wrap with mcp-remote instead |
| Vague docstrings on tools | Model can't reliably decide when/how to call the tool — write clear, example-driven docstrings |
| Blocking I/O inside async tool functions | Use async def consistently and await all I/O calls |
| Committing .env/API keys to git | Add .env to .gitignore; use secret managers in production |
| Assuming SSE is still the recommended remote transport | It's deprecated since 2025-03-26 — use Streamable HTTP for new servers |
| Not pinning dependency versions | Use uv.lock and uv sync --frozen in CI |
| Silent tool failures (returning nothing on error) | Return isError: true with a clear message so the model/host can react appropriately |

---

## 16. ❔ FREQUENTLY ASKED QUESTIONS

Q: Do I need to know JSON-RPC in depth to use FastMCP?
A: No — FastMCP abstracts the JSON-RPC wire format entirely. You write plain Python functions with decorators; FastMCP handles the protocol.

Q: Can one server support both stdio and HTTP?
A: Yes — most SDKs (including FastMCP) let you choose the transport at mcp.run(transport=...), so the same tool logic can run locally during dev (stdio) and remotely in production (Streamable HTTP).

Q: Will MCP replace REST APIs?
A: No. MCP is specifically a protocol for AI-tool access; REST/GraphQL APIs continue to serve human clients and other services. MCP servers typically wrap an existing API.

Q: How do I debug an MCP server without an LLM involved?
A: Use the official MCP Inspector (npx @modelcontextprotocol/inspector or uv run mcp dev server.py).

Q: What changed most in the 2026 MCP spec revision?
A: A move to a stateless protocol core, an Extensions framework (including MCP Apps for interactive UIs and a Tasks extension for long-running work), tighter OAuth/OIDC-style authorization, and a formal deprecation policy. Final release: July 28, 2026.

---

## 17. ✅ KEY TAKEAWAYS & REVISION NOTES

- MCP = open standard (JSON-RPC 2.0-based) connecting AI Hosts ↔ Servers via Clients.
- Solves the M×N integration problem → becomes M+N.
- Three primitives: Tools (actions), Resources (read-only data), Prompts (templates).
- Lifecycle: connect → initialize/handshake → capability negotiation → discovery → invocation → response/error handling → shutdown.
- Transports: stdio (local, current) and Streamable HTTP (remote, current); HTTP+SSE is deprecated (since 2025-03-26).
- FastMCP = the Pythonic, decorator-based way to build servers/clients fast; powers ~70% of MCP servers.
- uv = the modern, fast Python package/project manager; preferred for all MCP tooling.
- MCP Inspector = essential debugging tool, test servers without an LLM in the loop.
- Claude Desktop config lives at a fixed path per OS; edits require a full app restart.
- Remote servers need real auth (OAuth 2.1) — never rely on transport alone.
- The protocol is actively evolving: 2026 brings statelessness, Extensions, MCP Apps, and Tasks — always check the official spec for the latest before building production systems.

---

Compiled for personal revision — based on playlist study plus official MCP, FastMCP, and ecosystem documentation as of July 2026.
