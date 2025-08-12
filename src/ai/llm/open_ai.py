import dataclasses
from collections.abc import AsyncIterator, Iterable
from pathlib import Path
from typing import ClassVar, Optional, Self

from mkutils import Logger, Utils
from openai import AsyncOpenAI
from openai.types.responses import ResponseCreatedEvent, ResponseInputParam, ResponseTextDeltaEvent
from openai.types.responses.response_input_text_param import ResponseInputTextParam

from ai.llm.utils import Llm, Response

logger: Logger = Logger.new(__name__)


@dataclasses.dataclass(frozen=True, kw_only=True)
class OpenAi(Llm):
    STREAM: ClassVar[bool] = True

    client: AsyncOpenAI
    model: str

    @classmethod
    def new(cls, *, api_key: str, model: str) -> Self:
        client = AsyncOpenAI(api_key=api_key)
        open_ai = cls(client=client, model=model)

        return open_ai

    # @staticmethod
    # def _response_input_file_param(*, filepath: Path) -> ResponseInputFileParam:
    #     filename = str(filepath)
    #     mime_type = Utils.mime_type(filepath=filepath)
    #     file_data = f"data:{mime_type};base64,{filepath.expanduser().read_text()}"
    #     response_input_file_param = {"type": "input_file", "file_data": file_data, "filename": filename}

    #     return response_input_file_param

    @staticmethod
    def _response_input_text_param(*, text: str) -> ResponseInputTextParam:
        return {"type": "input_text", "text": text}

    @staticmethod
    def _text_from_file(*, filepath: Path) -> str:
        return f"### {filepath}\n===\n{filepath.read_text()}\n"

    @classmethod
    def _iter_input_path_response_input_text_params(cls, *, input_paths: Iterable[Path]) -> ResponseInputTextParam:
        for input_path in input_paths:
            for filepath in Utils.iter_filepaths(input_path.expanduser()):
                try:
                    text = cls._text_from_file(filepath=filepath)
                except UnicodeDecodeError:
                    continue
                else:
                    yield cls._response_input_text_param(text=text)

    # pylint: disable=unused-argument
    def _response_input_param(
        self, *, context: Optional[str], input_paths: Iterable[Path], query: str
    ) -> ResponseInputParam:
        file_content_iter = self._iter_input_path_response_input_text_params(input_paths=input_paths)
        query_content = self._response_input_text_param(text=query)
        content = [*file_content_iter, query_content]
        response_input_param = [
            {
                "content": content,
                "role": "user",
                "type": "message",
            }
        ]

        return response_input_param

    async def _aiter_text(
        self, *, context: Optional[str], instructions: Optional[str], input_paths: Iterable[Path], query: str
    ) -> AsyncIterator[str]:
        response_input_param = self._response_input_param(context=context, input_paths=input_paths, query=query)
        chunk_aiter = await self.client.responses.create(
            input=response_input_param,
            instructions=instructions,
            model=self.model,
            stream=self.STREAM,
        )

        async for chunk in chunk_aiter:
            match chunk:
                case ResponseCreatedEvent(id=response_id):
                    logger.info(response_id=response_id)
                case ResponseTextDeltaEvent(delta=text):
                    yield text
                case ignored_event:
                    logger.debug(ignored_event=ignored_event)

    def respond(
        self, *, context: Optional[str], instructions: Optional[str], input_paths: Iterable[Path], query: str
    ) -> Response:
        text_aiter = self._aiter_text(context=context, instructions=instructions, input_paths=input_paths, query=query)
        response = Response(context_id="", text_aiter=text_aiter)

        return response
