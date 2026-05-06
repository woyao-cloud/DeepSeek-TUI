"""CLI entry point — argparse-based command routing.

Port of `deepseek-tui-cli` (the `deepseek` dispatcher).
"""

from __future__ import annotations

import argparse
import asyncio
import io
import sys
from pathlib import Path
from typing import Optional

# Windows GBK workaround: force UTF-8 on stdout/stderr so emoji and
# non-ASCII characters from API responses don't crash on print().
if sys.stdout and hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr and hasattr(sys.stderr, 'buffer'):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from . import __version__
from .config import ConfigStore, ConfigToml, ProviderConfig, ProviderKind, CliRuntimeOverrides


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deepseek",
        description="DeepSeek TUI/CLI for DeepSeek models (Python port)",
        epilog="Not affiliated with DeepSeek Inc.",
    )
    parser.add_argument("--version", action="version", version=f"deepseek {__version__}")
    parser.add_argument("--config", type=Path, help="Path to config file")
    parser.add_argument("--profile", help="Config profile name")
    parser.add_argument("--provider", help="Provider (deepseek, openai, nvidia-nim, etc.)")
    parser.add_argument("--model", help="Model identifier")
    parser.add_argument("--api-key", help="API key")
    parser.add_argument("--base-url", help="API base URL")
    parser.add_argument("--output-mode", help="Output mode (json, text)")
    parser.add_argument("--log-level", help="Log level (debug, info, warn, error)")
    parser.add_argument("--telemetry", action="store_true", help="Enable telemetry")
    parser.add_argument("--approval-policy", help="Approval policy")
    parser.add_argument("--sandbox-mode", help="Sandbox mode")
    parser.add_argument("-p", "--prompt", help="One-shot prompt (non-interactive)")

    sub = parser.add_subparsers(dest="command", help="Subcommands")

    # Serve
    serve = sub.add_parser("serve", help="Run a local server")
    serve.add_argument("--http", action="store_true", help="Start HTTP server")
    serve.add_argument("--host", default="127.0.0.1", help="Bind host")
    serve.add_argument("--port", type=int, default=7878, help="Bind port")

    # Config
    cfg = sub.add_parser("config", help="Read/write/list config values")
    cfg.add_argument("action", choices=["get", "set", "unset", "list", "path"], help="Action")
    cfg.add_argument("key", nargs="?", help="Config key")
    cfg.add_argument("value", nargs="?", help="Config value")

    # Model
    mdl = sub.add_parser("model", help="Resolve or list models")
    mdl.add_argument("action", choices=["list", "resolve"], help="Action")
    mdl.add_argument("--provider", help="Provider filter")
    mdl.add_argument("model_id", nargs="?", help="Model ID to resolve")

    # Doctor
    sub.add_parser("doctor", help="Run system diagnostics")

    # Login
    login = sub.add_parser("login", help="Save an API key")
    login.add_argument("--api-key", help="API key to store")
    login.add_argument("--provider", default="deepseek", help="Provider name")

    # Logout
    sub.add_parser("logout", help="Remove saved API key")

    # Exec
    exec_cmd = sub.add_parser("exec", help="Run a non-interactive prompt")
    exec_cmd.add_argument("prompt", nargs="?", help="Prompt text")
    exec_cmd.add_argument("--model", help="Override model")
    exec_cmd.add_argument("--auto", action="store_true", help="Enable agentic mode")

    # Sessions
    sub.add_parser("sessions", help="List saved sessions")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Load config
    store = ConfigStore.load(args.config)
    cli_overrides = CliRuntimeOverrides(
        provider=ProviderKind.parse(args.provider) if args.provider else None,
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        output_mode=args.output_mode,
        log_level=args.log_level,
        telemetry=args.telemetry,
        approval_policy=args.approval_policy,
        sandbox_mode=args.sandbox_mode,
    )
    resolved = store.config.resolve_runtime_options(cli_overrides)

    if args.command is None or args.command == "exec":
        prompt = args.prompt or getattr(args, "prompt", None)
        if prompt:
            asyncio.run(run_one_shot(resolved, prompt, args.auto if hasattr(args, "auto") else False))
            return
        if args.command is None:
            parser.print_help()
            return

    match args.command:
        case "serve":
            print("Server mode not yet implemented (use --http)")
        case "config":
            handle_config(store, args)
        case "model":
            handle_model(args)
        case "doctor":
            print(f"DeepSeek TUI v{__version__}")
            print(f"Config: {store.path}")
            print(f"Provider: {resolved.provider.value}")
            print(f"Model: {resolved.model}")
            print(f"Base URL: {resolved.base_url}")
            print(f"API Key: {'set' if resolved.api_key else 'not set'}")
        case "login":
            handle_login(store, args)
        case "logout":
            handle_logout(store)
        case "sessions":
            from .state import StateStore
            state = StateStore()
            threads = state.list_threads()
            if not threads:
                print("No sessions found.")
                return
            for t in threads:
                name = t.name or "(unnamed)"
                print(f"{t.id[:12]}... | {name} | {t.model_provider} | {t.cwd}")
        case _:
            parser.print_help()


def handle_config(store: ConfigStore, args: argparse.Namespace) -> None:
    match args.action:
        case "get":
            if args.key:
                val = store.config.get_value(args.key)
                if val is not None:
                    print(val)
                else:
                    print(f"key not found: {args.key}", file=sys.stderr)
                    sys.exit(1)
        case "set":
            if args.key and args.value:
                store.config.set_value(args.key, args.value)
                store.save()
                print(f"set {args.key}")
        case "unset":
            if args.key:
                store.config.set_value(args.key, "")
                store.save()
                print(f"unset {args.key}")
        case "list":
            for k, v in store.config.list_values().items():
                print(f"{k} = {v}")
        case "path":
            print(store.path)


def handle_model(args: argparse.Namespace) -> None:
    from .agent import ModelRegistry
    registry = ModelRegistry()
    provider_hint = ProviderKind.parse(args.provider) if args.provider else None

    match args.action:
        case "list":
            for model in registry.list():
                if provider_hint and model.provider != provider_hint:
                    continue
                print(f"{model.id} ({model.provider.value})")
        case "resolve":
            resolved = registry.resolve(args.model_id, provider_hint)
            print(f"requested: {resolved.requested or ''}")
            print(f"resolved: {resolved.resolved.id}")
            print(f"provider: {resolved.resolved.provider.value}")
            print(f"used_fallback: {resolved.used_fallback}")


def handle_login(store: ConfigStore, args: argparse.Namespace) -> None:
    provider = ProviderKind.parse(args.provider) or ProviderKind.DEEPSEEK
    api_key = args.api_key
    if not api_key:
        api_key = input(f"Enter API key for {provider.value}: ").strip()
    if not api_key:
        print("API key required", file=sys.stderr)
        sys.exit(1)
    store.config.provider = provider
    if provider not in store.config.providers:
        store.config.providers[provider] = ProviderConfig()
    store.config.providers[provider].api_key = api_key
    if provider == ProviderKind.DEEPSEEK:
        store.config.api_key = api_key
    store.save()
    print(f"logged in using API key mode ({provider.value})")


def handle_logout(store: ConfigStore) -> None:
    store.config.api_key = None
    store.config.providers.clear()
    store.save()
    print("logged out")


async def run_one_shot(resolved, prompt: str, auto: bool = False) -> None:
    from .llm import DeepSeekClient, MessageRequest, Message, LlmError

    if not resolved.api_key:
        print("No API key configured. Use `deepseek login` or set DEEPSEEK_API_KEY.", file=sys.stderr)
        sys.exit(1)

    # Quick connectivity check via a non-streaming health request
    print(f"Connecting to {resolved.base_url} ...", file=sys.stderr, end=" ", flush=True)

    client = DeepSeekClient(
        api_key=resolved.api_key,
        base_url=resolved.base_url,
        model=resolved.model,
        timeout=30.0,
    )

    # First test: non-streaming call to verify API connectivity
    test_req = MessageRequest(
        model=resolved.model,
        messages=[Message(role="user", content=[{"type": "text", "text": prompt}])],
        max_tokens=64,
        stream=False,
    )
    try:
        resp = await asyncio.wait_for(client.create_message(test_req), timeout=30.0)
        # If non-streaming works, show full response
        for block in resp.content:
            text = getattr(block, "text", None) or (block.get("text") if isinstance(block, dict) else None)
            btype = getattr(block, "type", None) or (block.get("type") if isinstance(block, dict) else None)
            if btype == "text" and text:
                print(text)
        print(file=sys.stderr)
        return
    except asyncio.TimeoutError:
        print(file=sys.stderr)
        print("Timeout: API server did not respond within 15s.", file=sys.stderr)
        print(f"Check: 1) Network connectivity  2) API key validity  3) Base URL: {resolved.base_url}", file=sys.stderr)
        sys.exit(1)
    except LlmError as e:
        print(file=sys.stderr)
        print(f"API Error: {e}", file=sys.stderr)
        if "authentication" in str(e).lower() or "401" in str(e):
            print("Hint: Run `deepseek login --api-key sk-...` to set a valid key.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(file=sys.stderr)
        print(f"Connection failed: {e}", file=sys.stderr)
        print(f"Check base URL or network: {resolved.base_url}", file=sys.stderr)
        sys.exit(1)
    finally:
        await client.close()


if __name__ == "__main__":
    main()
