"""HTTP/SSE API server + JSON-RPC over stdio.

Port of `deepseek-app-server` crate.
"""

from .http_api import create_app
from .stdio_rpc import StdioRpcServer

__all__ = ["create_app", "StdioRpcServer"]
