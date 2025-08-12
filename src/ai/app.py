import contextlib
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Annotated, ClassVar, Optional, Self

from mkutils import Cli, Logger, Utils
from pydantic import BaseModel
from typer import Argument, Option

from ai.llm.open_ai import OpenAi
from ai.llm.utils import Llm, LlmProvider, LlmType

logger: Logger = Logger.new(__name__)


class Secret(BaseModel):
    open_ai_api_key: str

    @classmethod
    def from_filepath(cls, secret_filepath: Path) -> Self:
        return cls.model_validate_json(secret_filepath.read_text())


class App:
    DEFAULT_INSTRUCTIONS: ClassVar[str] = (
        "Answer user queries concisely but correctly; at the end of each response include in parentheses a better"
        " written version of the prompt to help the user get better at prompting."
    )
    DEFAULT_LLM_TYPE: ClassVar[LlmType] = LlmType.OPEN_AI_GPT_4_1
    DEFAULT_LOG_FILEPATH: ClassVar[Path] = Path("/dev/null")
    DEFAULT_SECRET_FILEPATH: ClassVar[Path] = Path(".env")
    LOG_FILEPATH_MODE: ClassVar[str] = "wt"

    @classmethod
    def cli(cls) -> None:
        Cli.new().add_command(fn=cls._run).run()

    @staticmethod
    @contextlib.contextmanager
    def _llm(*, secret: Secret, llm_type: LlmType) -> Iterator[Llm]:
        match llm_type.value.provider:
            case LlmProvider.OPEN_AI:
                yield OpenAi.new(api_key=secret.open_ai_api_key, model=llm_type.value.model)
            case unknown_llm_provider:
                raise Utils.value_error(unknown_llm_provider=unknown_llm_provider)

    @staticmethod
    def _query(*, query_list: Iterable[str]) -> str:
        return " ".join(query_list)

    # pylint: disable=too-many-arguments
    @classmethod
    async def _run(
        cls,
        *,
        secret_filepath: Annotated[Path, Option("--secret")] = DEFAULT_SECRET_FILEPATH,
        context: Annotated[Optional[str], Option()] = None,
        log_filepath: Annotated[Path, Option("--log")] = DEFAULT_LOG_FILEPATH,
        llm_type: Annotated[LlmType, Option("--model")] = DEFAULT_LLM_TYPE,
        instructions: Annotated[str, Option("--instructions")] = DEFAULT_INSTRUCTIONS,
        input_paths: Annotated[Optional[list[Path]], Option("--input")] = None,
        query_list: Annotated[list[str], Argument()],
    ) -> None:
        with log_filepath.open(cls.LOG_FILEPATH_MODE, encoding=Utils.ENCODING) as log_file:
            Logger.init(primary_file=log_file)

            secret = Secret.from_filepath(secret_filepath)
            query = cls._query(query_list=query_list)

            if input_paths is None:
                input_paths = []

            with cls._llm(secret=secret, llm_type=llm_type) as llm:
                await llm.respond(
                    context=context, instructions=instructions, input_paths=input_paths, query=query
                ).write(stream=sys.stdout)
