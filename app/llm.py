"""LLM provider factory (OpenAI, Ollama, or HuggingFace)."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableSerializable

from app.config import settings

logger = logging.getLogger(__name__)


class _JsonExtractorChain(RunnableSerializable):
    """Runnable that invokes an LLM and parses the response as a Pydantic model.

    Used for HuggingFace models that don't support ``response_format``
    or structured outputs.  The schema is injected into a system message
    and the raw text response is parsed with ``PydanticOutputParser``.
    """

    llm: Any
    parser: Any

    class Config:
        arbitrary_types_allowed = True

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        from langchain_core.messages import HumanMessage, SystemMessage

        schema_instructions = self.parser.get_format_instructions()

        # Build messages — input can be a dict, string, or list of messages
        if isinstance(input, str):
            messages = [
                SystemMessage(content=f"You must respond with valid JSON only. {schema_instructions}"),
                HumanMessage(content=input),
            ]
        elif isinstance(input, dict):
            # Comes from a prompt chain — convert to string
            text = "\n".join(f"{k}: {v}" for k, v in input.items())
            messages = [
                SystemMessage(content=f"You must respond with valid JSON only. {schema_instructions}"),
                HumanMessage(content=text),
            ]
        elif isinstance(input, list):
            # Already a list of messages — inject schema into first system msg
            messages = list(input)
            injected = False
            for i, msg in enumerate(messages):
                if isinstance(msg, SystemMessage):
                    messages[i] = SystemMessage(
                        content=f"{msg.content}\n\nIMPORTANT: {schema_instructions}"
                    )
                    injected = True
                    break
            if not injected:
                messages.insert(0, SystemMessage(
                    content=f"You must respond with valid JSON only. {schema_instructions}"
                ))
        else:
            messages = [
                SystemMessage(content=f"You must respond with valid JSON only. {schema_instructions}"),
                HumanMessage(content=str(input)),
            ]

        response = self.llm.invoke(messages, config=config, **kwargs)
        text = response.content if hasattr(response, "content") else str(response)

        # Try to extract JSON from the response
        try:
            return self.parser.parse(text)
        except Exception:
            # Try to find JSON in markdown code blocks
            json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
            if json_match:
                try:
                    return self.parser.parse(json_match.group(1).strip())
                except Exception:
                    pass
            # Try to find raw JSON object
            json_match = re.search(r"\{[\s\S]*\}", text)
            if json_match:
                try:
                    return self.parser.parse(json_match.group(0))
                except Exception:
                    pass
            logger.error("Failed to parse structured output from HF response: %s", text[:500])
            raise


class _HuggingFaceChatWrapper(RunnableSerializable):
    """Runnable wrapper around ChatOpenAI for HuggingFace Inference API.

    HuggingFace's OpenAI-compatible endpoint does not support structured
    outputs or json_mode.  This wrapper intercepts ``with_structured_output``
    and returns a prompt-based JSON extraction chain instead.
    All other calls (invoke, |, etc.) are delegated to the underlying ChatOpenAI.
    """

    inner_llm: Any = None

    class Config:
        arbitrary_types_allowed = True

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        return self.inner_llm.invoke(input, config=config, **kwargs)

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        parser = PydanticOutputParser(pydantic_object=schema)
        return _JsonExtractorChain(llm=self.inner_llm, parser=parser)

    def __getattr__(self, name: str) -> Any:
        if name == "inner_llm":
            raise AttributeError(name)
        return getattr(self.inner_llm, name)


def get_chat_llm(
    *,
    model: str | None = None,
    temperature: float = 0,
    max_tokens: int | None = None,
) -> Any:
    """Return a chat model based on ``settings.llm_provider``."""
    provider = settings.llm_provider.strip().lower()
    selected_model = model or settings.active_llm_model

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        kwargs: dict[str, Any] = {
            "model": selected_model,
            "base_url": settings.ollama_base_url,
            "temperature": temperature,
        }
        if settings.ollama_num_ctx > 0:
            kwargs["num_ctx"] = settings.ollama_num_ctx
        if max_tokens is not None:
            kwargs["num_predict"] = max_tokens
        return ChatOllama(**kwargs)

    if provider == "huggingface":
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=selected_model,
            api_key=settings.huggingface_api_key,
            base_url="https://router.huggingface.co/v1",
            temperature=max(temperature, 0.01),
            max_tokens=max_tokens or 1024,
        )
        return _HuggingFaceChatWrapper(inner_llm=llm)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        kwargs = {
            "model": selected_model,
            "temperature": temperature,
        }
        if settings.openai_api_key:
            kwargs["api_key"] = settings.openai_api_key
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        return ChatOpenAI(**kwargs)

    raise ValueError(f"Unsupported LLM_PROVIDER '{settings.llm_provider}' (use 'openai', 'ollama', or 'huggingface').")
