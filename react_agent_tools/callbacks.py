from typing import Any, Optional
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


class AgentCallBackHandler(BaseCallbackHandler):
    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:  # type: ignore[override]
        print("LLM started with prompts:")
        for prompt in prompts:
            print(prompt)
        print("\n")

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:  # type:ignore[override]
        print("LLM finished with response:")
        for generation in response.generations:
            for gen in generation:
                print(gen.text)
        print("\n")
