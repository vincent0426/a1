"""
Context (history) management for a1 agents.

Provides:
- Context: A list of messages with utilities
- no_history: Factory for throwaway contexts
- Compaction strategies for managing context size
"""

from typing import Dict, List, Optional
from .models import Message


class Context:
    """
    A context is a list of chat messages with convenience methods.
    
    Contexts track the conversation history for an agent execution,
    including user messages, assistant responses, and tool calls.
    """
    
    def __init__(self, messages: Optional[List[Message]] = None):
        self.messages: List[Message] = messages or []
    
    def append(self, message: Message):
        """Add a message to the context."""
        self.messages.append(message)
    
    def extend(self, messages: List[Message]):
        """Add multiple messages to the context."""
        self.messages.extend(messages)
    
    def user(self, content: str):
        """Add a user message."""
        self.append(Message(role="user", content=content))
    
    def assistant(self, content: str, tool_calls: Optional[List[Dict]] = None):
        """Add an assistant message."""
        self.append(Message(role="assistant", content=content, tool_calls=tool_calls))
    
    def system(self, content: str):
        """Add a system message."""
        self.append(Message(role="system", content=content))
    
    def tool(self, content: str, name: str, tool_call_id: str):
        """Add a tool message."""
        self.append(Message(role="tool", content=content, name=name, tool_call_id=tool_call_id))
    
    def clear(self):
        """Clear all messages."""
        self.messages.clear()
    
    def __len__(self) -> int:
        return len(self.messages)
    
    def __iter__(self):
        return iter(self.messages)
    
    def __getitem__(self, key):
        return self.messages[key]
    
    def to_dict_list(self) -> List[Dict]:
        """Convert to list of dicts for API calls."""
        return [msg.model_dump(exclude_none=True) for msg in self.messages]


def no_history() -> Context:
    """
    Create a throwaway context that won't be persisted.
    
    Useful for one-off LLM calls that shouldn't affect the main
    conversation history.
    """
    return Context()


class Compact:
    """
    Base class for context compaction strategies.
    
    Compaction reduces context size while preserving important information,
    preventing token limit issues and reducing costs.
    """
    
    def compact(self, contexts: Dict[str, Context]) -> Dict[str, Context]:
        """
        Compact one or more contexts.
        
        Args:
            contexts: Dictionary mapping context names to Context objects
        
        Returns:
            Compacted contexts (may modify in-place or return new dict)
        """
        raise NotImplementedError


class BaseCompact(Compact):
    """
    Base compaction strategy that does nothing.
    
    Keeps all messages without any reduction. Useful for:
    - Short conversations
    - Debugging
    - When context window is large enough
    """
    
    def compact(self, contexts: Dict[str, Context]) -> Dict[str, Context]:
        """No-op compaction."""
        return contexts


class SlidingWindowCompact(Compact):
    """
    Keep only the last N messages in each context.
    
    Args:
        window_size: Maximum number of messages to keep
        keep_system: Whether to always keep system messages
    """
    
    def __init__(self, window_size: int = 10, keep_system: bool = True):
        self.window_size = window_size
        self.keep_system = keep_system
    
    def compact(self, contexts: Dict[str, Context]) -> Dict[str, Context]:
        """Keep last N messages per context."""
        for name, context in contexts.items():
            if len(context) <= self.window_size:
                continue
            
            if self.keep_system:
                # Separate system messages from others
                system_msgs = [msg for msg in context if msg.role == "system"]
                other_msgs = [msg for msg in context if msg.role != "system"]
                
                # Keep system messages + last N-len(system) other messages
                keep_count = max(0, self.window_size - len(system_msgs))
                context.messages = system_msgs + other_msgs[-keep_count:]
            else:
                # Just keep last N messages
                context.messages = context.messages[-self.window_size:]
        
        return contexts


class SummarizationCompact(Compact):
    """
    Summarize old messages using an LLM when context gets too long.
    
    Args:
        max_messages: Trigger compaction when context exceeds this
        keep_recent: Number of recent messages to keep unsummarized
        model: Model to use for summarization
    """
    
    def __init__(self, max_messages: int = 20, keep_recent: int = 5, model: str = "gpt-4o-mini"):
        self.max_messages = max_messages
        self.keep_recent = keep_recent
        self.model = model
    
    def compact(self, contexts: Dict[str, Context]) -> Dict[str, Context]:
        """Summarize old messages when context is too long."""
        # TODO: Implement LLM-based summarization
        # For now, fall back to sliding window
        fallback = SlidingWindowCompact(window_size=self.max_messages)
        return fallback.compact(contexts)


__all__ = [
    "Context",
    "no_history",
    "Compact",
    "BaseCompact",
]
