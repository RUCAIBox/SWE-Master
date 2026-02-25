##############################################################################
# tool definitions
##############################################################################

# Import allowed commands from the editor module
from .str_replace_editor import ALLOWED_STR_REPLACE_EDITOR_COMMANDS

# V1 File Editor
_FILE_EDITOR_DESCRIPTION = """Custom editing tool for viewing, creating and editing files
* State is persistent across command calls and discussions with the user
* If `path` is a file, `view` displays the result of applying `cat -n`. If `path` is a directory, `view` lists non-hidden files and directories up to 2 levels deep
* The `create` command cannot be used if the specified `path` already exists as a file
* If a `command` generates a long output, it will be truncated and marked with `<response clipped>`
* The `undo_edit` command will revert the last edit made to the file at `path`

Notes for using the `str_replace` command:
* The `old_str` parameter should match EXACTLY one or more consecutive lines from the original file. Be mindful of whitespaces!
* If the `old_str` parameter is not unique in the file, the replacement will not be performed. Make sure to include enough context in `old_str` to make it unique
* The `new_str` parameter should contain the edited lines that should replace the `old_str`
"""

file_editor = {
    "type": "function",
    "function": {
        "name": "file_editor",
        "description": _FILE_EDITOR_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "description": "The command to run. Allowed options are: `view`, `create`, `str_replace`, `insert`, `undo_edit`.",
                    "enum": ["view", "create", "str_replace", "insert", "undo_edit"],
                    "type": "string",
                },
                "path": {
                    "description": "Absolute path to file or directory, e.g. `/testbed/file.py` or `/testbed`.",
                    "type": "string",
                },
                "file_text": {
                    "description": "Required for the `create` command, contains the content of the file to be created.",
                    "type": "string",
                },
                "old_str": {
                    "description": "Required for the `str_replace` command, specifies the string in `path` to replace.",
                    "type": "string",
                },
                "new_str": {
                    "description": "Optional for the `str_replace` command to specify the replacement string. Required for the `insert` command to specify the string to insert.",
                    "type": "string",
                },
                "insert_line": {
                    "description": "Required for the `insert` command. The `new_str` will be inserted AFTER the line specified.",
                    "type": "integer",
                },
                "view_range": {
                    "description": "Optional for the `view` command when `path` points to a file. Specifies the line range to view. E.g., [11, 12] shows lines 11 and 12. Indexing starts at 1. Use [start_line, -1] to show all lines from `start_line` to the end.",
                    "type": "array",
                    "items": {"type": "integer"},
                },
                "concise": {
                    "description": "Optional for the `view` command. If `True`, displays a concise skeletal view of the file. Very useful for localization tasks. Highly recommended for large files.",
                    "type": "boolean",
                },
            },
            "required": ["command", "path"],
        },
    },
}


_STR_REPLACE_EDITOR_DESCRIPTION = """Custom editing tool for viewing, creating and editing files
* State is persistent across command calls and discussions with the user
* If `path` is a file, `view` displays the result of applying `cat -n`. If `path` is a directory, `view` lists non-hidden files and directories up to 2 levels deep
* The `create` command cannot be used if the specified `path` already exists as a file
* If a `command` generates a long output, it will be truncated and marked with `<response clipped>`

Notes for using the `str_replace` command:
* The `old_str` parameter should match EXACTLY one or more consecutive lines from the original file. Be mindful of whitespaces!
* If the `old_str` parameter is not unique in the file, the replacement will not be performed. Make sure to include enough context in `old_str` to make it unique
* The `new_str` parameter should contain the edited lines that should replace the `old_str`
"""

str_replace_editor_tool = {
    "type": "function",
    "function": {
        "name": "str_replace_editor",
        "description": _STR_REPLACE_EDITOR_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "description": f"The command to run. Allowed options are: {', '.join(f'`{cmd}`' for cmd in ALLOWED_STR_REPLACE_EDITOR_COMMANDS)}.",
                    "enum": ALLOWED_STR_REPLACE_EDITOR_COMMANDS,
                    "type": "string",
                },
                "path": {
                    "description": "Absolute path to file or directory, e.g. `/testbed/file.py` or `/testbed`.",
                    "type": "string",
                },
                "file_text": {
                    "description": "Required for the `create` command, contains the content of the file to be created.",
                    "type": "string",
                },
                "old_str": {
                    "description": "Required for the `str_replace` command, specifies the string in `path` to replace.",
                    "type": "string",
                },
                "new_str": {
                    "description": "Optional for the `str_replace` command to specify the replacement string. Required for the `insert` command to specify the string to insert.",
                    "type": "string",
                },
                "insert_line": {
                    "description": "Required for the `insert` command. The `new_str` will be inserted AFTER the line specified.",
                    "type": "integer",
                },
                "view_range": {
                    "description": "Optional for the `view` command when `path` points to a file. Specifies the line range to view. E.g., [11, 12] shows lines 11 and 12. Indexing starts at 1. Use [start_line, -1] to show all lines from `start_line` to the end.",
                    "type": "array",
                    "items": {"type": "integer"},
                },
            },
            "required": ["command", "path"],
        },
    },
}


_R2EGYM_BASH_EXECUTE_DESCRIPTION = """
Description: Execute a bash command in the terminal.

Parameters:
  (1) command (string, required): The bash command to execute. For example: `python my_script.py`
"""

r2egym_bash_execute_tool = {
    "type": "function",
    "function": {
        "name": "execute_bash",
        "description": _R2EGYM_BASH_EXECUTE_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {
                "cmd": {
                    "type": "string",
                    "description": "The command (and optional arguments) to execute. For example: 'python my_script.py'",
                }
            },
            "required": ["cmd"],
        },
    },
}



_BASH_DESCRIPTION = """
Description: Execute a bash command in the terminal.

Parameters:
  (1) command (string, optional): The bash command to execute. For example: `python my_script.py`. If not provided, will show help.
"""

execute_bash_tool = {
    "type": "function",
    "function": {
        "name": "execute_bash",
        "description": _BASH_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The command (and optional arguments) to execute. For example: 'python my_script.py'",
                }
            },
            "required": [],
        },
    },
}


_SEARCH_DESCRIPTION = """
Description: Search for a term in either a directory or a single file.

Behavior:
* If `--path` points to a directory (default is `.`), we recursively search all non-hidden files and directories.
* If `--path` points to a file, we run `grep -n` on that file to find line numbers containing the search term.
* If more than 100 files match (directory search scenario), the tool will stop listing and inform you to narrow your search.
* If no files are found that match your search term, the tool will inform you of that as well.

**Parameters:**
  1. **search_term** (`string`, required): The term to search for in files.
  2. **path** (`string`, optional): The file or directory in which to search. If not provided, defaults to the current directory (i.e., `.`).
"""

search_tool = {
    "type": "function",
    "function": {
        "name": "search",
        "description": _SEARCH_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {
                "search_term": {
                    "description": "The term to search for in files.",
                    "type": "string",
                },
                "path": {
                    "description": "The file or directory to search in. Defaults to `.` if not specified.",
                    "type": "string",
                },
            },
            "required": ["search_term"],
        },
    },
}

# V1 Finish
_FINISH_DESCRIPTION = """
"A simple finish tool with a 'submit' command.\n\n"
"Notes about the `submit` command:\n"
"* When invoked with `--result`, the provided string is used for submitting required task results.\n"
"* If no `--result` is provided, it defaults to an empty string.\n\n"
"**Parameters:**\n"
"  1. **command** (`string`, required): The command to run. Currently allowed option is: `submit`.\n"
"     - Allowed value: [`submit`]\n"
"  2. **result** (`string`, optional): The result text to submit. Defaults to an empty string.\n"
"""
finish_tool = {
    "type": "function",
    "function": {
        "name": "finish",
        "description": _FINISH_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "description": "The command to run. Currently only `submit` is supported.",
                    "type": "string",
                    "enum": ["submit"],
                },
                "result": {
                    "description": "Optional. The result text to submit. Defaults to an empty string if not provided.",
                    "type": "string",
                },
            },
            "required": ["command"],
        },
        # "cache_control": {"type": "ephemeral"},
    },
}

# V2 Submit
_SUBMIT_DESCRIPTION = """
A simple submit tool to finish tasks.

This tool signals completion of a task or submission of results.
No parameters required - simply call to indicate task completion.
"""

submit_tool = {
    "type": "function",
    "function": {
        "name": "submit",
        "description": _SUBMIT_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}



ALLOWED_LSP_COMMANDS = [
    "get_definition",
    "get_type_definition",
    "get_implementation",
    "get_hover",
    "get_references",
    "get_document_highlights",
    "get_document_symbols",
    "get_workspace_symbols",
    "get_call_hierarchy",
]

_LSP_TOOL_DESCRIPTION="""
Description: A powerful **code intelligence tool** that provides deep, semantic understanding of the code. It provides faster and more accurate code navigation and search, as well as comprehensive code analysis capabilities.
This tool connects to a **persistent Language Server Protocol (LSP)** client (based on **Pyright**) to analyze Python code intelligently – it understands code structure, types, imports, and symbol relationships across multiple files.

Use this tool when you need to understand *how* code works, not just *what* it says. 
It goes far beyond simple text search (`grep`) by analyzing the *abstract syntax tree* and symbol table.

Capabilities:
* `get_definition`: "Go to Definition." Find where a variable, function, or class is defined AND returns the FULL source code of the target function/class.
* `get_type_definition`: "Go to Type Definition." Finds the definition of a symbol's *type* AND returns the FULL source code. (e.g., on 'my_obj' in 'my_obj = MyClass()', this jumps to 'class MyClass:')
* `get_hover`: "Hover." Get the symbol's docstring, inferred type, and function signature at the cursor.
* `get_call_hierarchy`: "Call Hierarchy." Returns a complete report of Incoming Callers (all functions that *call* this function) and Outgoing Callees (all functions that *this* function calls) calls.
* `get_references`: "Find All References." Find all usages of a symbol across the *entire project*.
* `get_document_highlights`: "Highlight Usages." Highlights all usages of the symbol at the cursor *within the current file*.
* `get_document_symbols`: "Document Outline." List all symbols (classes, functions, variables) in the *current file* as a hierarchical tree.
* `get_workspace_symbols`: "Workspace Search." Search for symbols across the *entire project* by a query string. Note that the query string shoule be a symbol. (e.g., use "MyClass" instead of "class MyClass", use "my_function" instead of "def my_function")

CRITICAL: This tool is AUTOMATICALLY SYNCHRONIZED.
The system **automatically** syncs your file (e.g., when you use `str_replace_editor`) with this **Pyright-based LSP client**.
This means you can immediately use analysis commands (like `get_definition` or `get_references`) on any file right *after* you have viewed or edited it.

Notes:
This tool is for **ANALYSIS ONLY**. To make changes, you MUST use the `str_replace_editor` tool.
All positional commands (`get_definition`, `get_hover`, etc.) require `file_path`, `line` (1-indexed), and `symbol` (the name of a Python code entity).

Parameters:
  (1) command (string, required): The LSP command to run.
Allowed values: [`get_definition`, `get_type_definition`, `get_references`, `get_hover`, `get_document_highlights`, `get_document_symbols`, `get_workspace_symbols`, `prepare_call_hierarchy`, `get_incoming_calls`, `get_outgoing_calls`]
  (2) --file_path (string, optional): Absolute path to the file for the command.
  (3) --symbol (string, optional): The name of a Python code entity for positional commands. Includes identifiers for Classes, Functions, Methods, Variables, Fields, and Modules. (e.g., 'MyClass', 'process_data', 'user_id').
  (4) --line (integer, optional): Line number (1-indexed) for positional commands. Used with symbol to locate the specific symbol.
  (5) --query (string, optional): Required for `get_workspace_symbols`, the symbol to search for.
        Note that the query parameter shoule be a symbol. (e.g., use "MyClass" instead of "class MyClass", use "my_function" instead of "def my_function")
"""

lsp_tool = {
  "type": "function",
  "function": {
    "name": "lsp_tool",
    "description": _LSP_TOOL_DESCRIPTION,
    "parameters": {
      "type": "object",
      "properties": {
        "command": {
          "type": "string",
          "description": f"The LSP analysis or workspace command to run. Allowed options are: {', '.join(f'`{cmd}`' for cmd in ALLOWED_LSP_COMMANDS)}.",
          "enum": ALLOWED_LSP_COMMANDS
        },
        "file_path": {
          "type": "string",
          "description": "Absolute path to the file. Required by all *analysis* commands (like `get_definition`) except `get_workspace_symbols`."
        },
        "symbol":{
            "type":"string",
            "description": "The name of a Python code entity for positional commands. Includes identifiers for Classes, Functions, Methods, Variables, Fields, and Modules. Used with line to locate the specific symbol."
        },
        "line": {
          "type": "integer",
          "description": "Line number (1-indexed). Used with symbol to locate the specific symbol. Required for *positional* commands (e.g., `get_definition`, `get_hover`).",
          "minimum": 0
        },
        "query": {
          "type": "string",
          "description": "The symbol to search for. Required *only* for the `get_workspace_symbols` command."
        },
      },
      "required": [
        "command"
      ],
    }
  }
}
