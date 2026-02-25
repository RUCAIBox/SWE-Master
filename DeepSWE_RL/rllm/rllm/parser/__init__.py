from rllm.parser.tool_parser.qwen_tool_parser import QwenToolParser
from rllm.parser.tool_parser.r1_tool_parser import R1ToolParser
from rllm.parser.tool_parser.tool_parser_base import ToolParser

PARSER_REGISTRY = {
    "r1": R1ToolParser,
    "qwen": QwenToolParser,
}


def get_tool_parser(parser_name: str) -> type[ToolParser]:
    assert parser_name in PARSER_REGISTRY, f"Tool parser {parser_name} not found in {PARSER_REGISTRY}"
    return PARSER_REGISTRY[parser_name]


__all__ = [
    "R1ToolParser",
    "QwenToolParser",
    "ToolParser",
    "get_tool_parser",
    "PARSER_REGISTRY",
]
