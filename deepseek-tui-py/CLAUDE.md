# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DeepSeek TUI is a Python port of the Rust-based DeepSeek CLI, providing a terminal-native coding agent with a 1M-token context window. It supports interactive TUI, CLI execution, and HTTP API modes.

**Python requirement**: 3.12+

## Development Commands

```bash
# Install dependencies
pip install -r requirements.txt
pip install -e ".[tui,server,mcp]"  # with optional deps

# CLI mode (requires API key)
export DEEPSEEK_API_KEY=sk-...
python -m deepseek_tui.cli exec "Hello, what can you do?"

# TUI mode (requires textual)
python -m deepseek_tui.cli tui

# Configuration
python -m deepseek_tui.cli config list
python -m deepseek_tui.cli config set model deepseek-v4-flash

# Run tests
python -m pytest tests/ -v
python -m pytest tests/test_tools_and_runtime.py -v   # single test file

# Lint
python -m ruff check src/ tests/
```

## Architecture

### Data Flow

1. **Entry Point** (`cli.1.py`): Parses args, resolves config, dispatches to TUI or CLI mode
2. **Runtime** (`core/runtime.1.py`): ThreadManager, JobManager, tool orchestration
3. **LLM Client** (`llm/client.1.py`): DeepSeek API client with streaming and retry logic
4. **Tools** (`tools/`): Registered tools execute via ToolRegistry
5. **Cycle Manager** (`core/cycle.1.py`): Context window management with archiving

### Key Modules

| Module | Purpose |
|--------|---------|
| `config/` | Config loading, provider settings, secrets |
| `llm/` | API client, models, retry logic |
| `agent/` | Model registry and resolution |
| `protocol/` | Thread/message types |
| `state/` | SQLite persistence for sessions |
| `tools/` | Tool registry + implementations (file, shell, git, web, skills, LSP) |
| `execpolicy/` | Approval/sandbox policy engine |
| `mcp/` | MCP client integration |
| `core/` | Runtime orchestration, cycle management |
| `tui/` | Textual-based terminal UI |

### Configuration

- Config: `~/.deepseek/config.toml` or project `.deepseek/config.toml`
- MCP servers: `~/.deepseek/mcp.1.json`
- Skills: `~/.deepseek/skills/`
- Sessions: `~/.deepseek/sessions/`

### Adding a Tool

1. Create tool class in `tools/` (e.g., `tools/shell_1tools.py`)
2. Implement `ToolSpec` with name, description, input_schema
3. Register in `ToolRegistry` during app initialization