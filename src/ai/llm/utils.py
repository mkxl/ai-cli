import dataclasses
from collections.abc import AsyncIterator, Iterable
from enum import StrEnum
from pathlib import Path
from typing import IO, Optional, Protocol

from mkutils import Enum, Utils


class LlmProvider(StrEnum):
    OPEN_AI = "open_ai"


@Utils.keyed_by(attr="model")
@dataclasses.dataclass(frozen=True, kw_only=True)
class LlmInfo:
    provider: LlmProvider
    model: str


class LlmType(Enum):
    OPEN_AI_GPT_4_1 = LlmInfo(provider=LlmProvider.OPEN_AI, model="gpt-4.1")
    OPEN_AI_GPT_5 = LlmInfo(provider=LlmProvider.OPEN_AI, model="gpt-5")
    OPEN_AI_GPT_5_MINI = LlmInfo(provider=LlmProvider.OPEN_AI, model="gpt-5-mini")
    OPEN_AI_O3 = LlmInfo(provider=LlmProvider.OPEN_AI, model="o3")
    OPEN_AI_O4_MINI = LlmInfo(provider=LlmProvider.OPEN_AI, model="o4-mini")


@dataclasses.dataclass(frozen=True, kw_only=True)
class Response:
    context_id: str
    text_aiter: AsyncIterator[str]

    def header(self) -> str:
        return f"context: {self.context_id!r}"

    async def write(self, *, stream: IO) -> None:
        stream.write(self.header())
        stream.write("\n\n")

        async for text in self.text_aiter:
            stream.write(text)

        stream.write("\n")


class Llm(Protocol):
    def respond(
        self, *, context: Optional[str], instructions: Optional[str], input_paths: Iterable[Path], query: str
    ) -> Response:
        raise NotImplementedError
