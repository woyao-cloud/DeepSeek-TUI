"""LLM client — HTTP client, retry logic, streaming, and request/response types.

Port of `crates/tui/src/llm_client/` and `crates/tui/src/models.rs`.
"""

from .models import (
    MessageRequest,
    MessageResponse,
    Message,
    ContentBlock,
    ContentBlockData,
    SystemPrompt,
    Tool,
    Usage,
    StreamEvent,
    MessageDelta,
    Delta,
)
from .client import LlmClient, DeepSeekClient
from .retry import RetryConfig, RetryError, with_retry, LlmError

__all__ = [
    "MessageRequest",
    "MessageResponse",
    "Message",
    "ContentBlock",
    "ContentBlockData",
    "SystemPrompt",
    "Tool",
    "Usage",
    "StreamEvent",
    "MessageDelta",
    "Delta",
    "LlmClient",
    "DeepSeekClient",
    "RetryConfig",
    "RetryError",
    "with_retry",
    "LlmError",
]
