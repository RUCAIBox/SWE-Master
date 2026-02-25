#!/root/.venv/bin/python
# -*- coding: utf-8 -*-
"""
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
  (2) --file_path (string, optional): Absolute path to the file for the command. Except for `get_workspace_symbols`
  (3) --symbol (string, optional): The name of a Python code entity for positional commands. Includes identifiers for Classes, Functions, Methods, Variables, Fields, and Modules. (e.g., 'MyClass', 'process_data', 'user_id').
  (4) --line (integer, optional): Line number (1-indexed) for positional commands. Used with symbol to locate the specific symbol.
  (5) --query (string, optional): Required for `get_workspace_symbols`, the symbol to search for.
        Note that the query parameter shoule be a symbol. (e.g., use "MyClass" instead of "class MyClass", use "my_function" instead of "def my_function")
"""

import json
import os
import asyncio
import sys
import io
import argparse
import struct
import traceback
from typing import Optional, Any, List, Dict, Union
from pathlib import Path
from urllib.parse import urlparse, unquote
import orjson

# --- Configuration ---
HOST = '127.0.0.1'
PORT_FILE_ENV_VAR = os.environ.get("LSP_PORT_FILE", "/var/tmp/lsp_port_session_abc.pid")

# --- Python 3.5/3.6 Compatibility Functions ---
def run_asyncio(coro):
    """Compatible with asyncio.run() (Python 3.7+)"""
    if sys.version_info >= (3, 7):
        return asyncio.run(coro)
    else:
        # Python 3.5/3.6: Use existing event loop or create a new one
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        try:
            return loop.run_until_complete(coro)
        finally:
            # For client tools, we can simply close the loop 
            # because each call is independent
            try:
                all_tasks_func = getattr(asyncio, "all_tasks", getattr(asyncio.Task, "all_tasks", None))
                if all_tasks_func:
                    pending = all_tasks_func(loop)
                    for task in pending:
                        task.cancel()
                    if pending:
                        loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception:
                pass


# --- Basic Helper Functions ---
def uri_to_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme in ('', 'file'):
        p = unquote(parsed.path)
        if os.name == 'nt' and p.startswith('/') and len(p) > 2 and p[2] == ':':
            p = p[1:]
        return Path(p)
    raise ValueError("Unsupported URI scheme: " + parsed.scheme)

def safe_print(x, file=None):
    if file is None: file = sys.stderr
    try:
        print(x, file=file)
    except UnicodeEncodeError:
        print(x.encode("utf-8", errors="replace").decode("utf-8", errors="replace"), file=file)

def _uri_to_path_str(uri: str) -> str:
    """(New) Convert file URI to a relative path string within the project"""
    try:
        path = uri_to_path(uri)
        # try:
        #     # Attempt to make it relative to project root
        #     relative_path = path.relative_to(Path(DEFAULT_PROJECT_ROOT))
        #     return str(relative_path)    
        # except ValueError:
        #     # If not under project root, return only the filename
        #     return path.name
        return str(path)
    except Exception:
        return uri

def _format_range(range_obj: dict, use_end: bool = False) -> str:
    """(New) Convert LSP range object to 'line:char' string (1-indexed)"""
    if not range_obj:
        return "Unknown location"
    start_pos = range_obj.get('start', {})
    end_pos = range_obj.get('end', {})
    
    s_line = start_pos.get('line', 0) + 1  # Convert to 1-indexed
    s_char = start_pos.get('character', 0) + 1 # Convert to 1-indexed
    e_line = end_pos.get('line', 0) + 1  # Convert to 1-indexed
    e_char = end_pos.get('character', 0) + 1 # Convert to 1-indexed

    # return f"from line {s_line} char {s_char} to line {e_line} char {e_char}"
    return f"(line {s_line} to {e_line})"

def _simple_format_range(range_obj: dict, use_end: bool = False) -> str:
    """(New) Convert LSP range object to 'line:char' string (1-indexed)"""
    if not range_obj:
        return "Unknown location"
    start_pos = range_obj.get('start', {})
    end_pos = range_obj.get('end', {})
    
    s_line = start_pos.get('line', 0) + 1  # Convert to 1-indexed
    s_char = start_pos.get('character', 0) + 1 # Convert to 1-indexed
    e_line = end_pos.get('line', 0) + 1  # Convert to 1-indexed
    e_char = end_pos.get('character', 0) + 1 # Convert to 1-indexed

    return f"(at line {s_line})"

def _format_location(location: dict) -> str:
    """(New) Format a single Location object"""
    if not location:
        return "Unknown location"

    # debug_raw = repr(location)
    # if len(debug_raw) > 1000:
    #     debug_raw = debug_raw[:1000] + "... (truncated)"
    # safe_print(f"debug raw response: {debug_raw}")

    uri = location.get('uri', 'unknown_file')
    range_obj = location.get('range', {})
    path_str = _uri_to_path_str(uri)
    pos_str = _format_range(range_obj)
    return f"{path_str}  {pos_str}"

def _format_location_link(location_link: dict) -> str:
    """(New) Format LocationLink (used for get_definition, etc.)"""
    if not location_link:
        return "Unknown location"
    uri = location_link.get('targetUri', 'unknown_file')
    # Use 'targetSelectionRange' because it points more accurately to the symbol name
    range_obj = location_link.get('targetSelectionRange', {})
    path_str = _uri_to_path_str(uri)
    pos_str = _format_range(range_obj)
    return f"{path_str} (at {pos_str})"

# --- Result Wrapper ---
class LSPResult:
    def __init__(self, result: Optional[dict] = None, error: Optional[str] = None):
        self.result = result
        self.error = error
        self.status_code = "success" if error is None else "error"

    def to_text(self) -> str:
        """
        (Modified) Convert results to LLM-friendly plain text format, 
        removing JSON wrapping and keeping only summary and necessary context data.
        """
        # 1. Handle error cases
        if self.error:
            return f"!! LSP Command Failed !!\nError: {self.error}"
        
        if not self.result:
            return "LSP Command Succeeded (No output content)."

        output_parts = []
        
        # 2. Extract Summary - this is the main content
        summary = self.result.get("summary", "")
        if summary:
            output_parts.append(summary)
        
        # # 3. Handle special data (Context Data)
        # # Some commands (like prepare_call_hierarchy) return 'item_data' as required parameters for subsequent steps.
        # # We keep it as a single-line JSON to facilitate model copying/referencing.
        # # Data already shown in summary like 'symbols' or 'locations' are ignored to reduce noise.
        # if "item_data" in self.result:
        #     try:
        #         # Use orjson to serialize into a compact JSON string
        #         data_str = orjson.dumps(self.result["item_data"]).decode('utf-8')
        #         output_parts.append(f"\n[DATA] item_data: {data_str}")
        #     except Exception:
        #         output_parts.append(f"\n[DATA] item_data: {self.result['item_data']}")

        # 4. Fallback: If neither summary nor special data exists, print raw results
        if not output_parts:
            try:
                fallback_json = json.dumps(self.result, indent=2, default=str)
                output_parts.append(fallback_json)
            except Exception:
                output_parts.append(str(self.result))

        return "[status_code]: \n {} \n".format(self.status_code) +  "[Result]: \n" + "\n".join(output_parts)

    def __str__(self):
        return self.to_text()

    def to_dict(self) -> dict:
        if self.error:
            return {"status_code": "error", "error": self.error}
        return {"status_code": "success", "result": self.result}

    def to_json_str(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    # def __str__(self):
    #     return self.to_json_str()
    
# --- Parser (Enhanced Version) ---
class LSPResponseParser:
    @staticmethod
    def parse(command: str, data: Optional[dict]) -> dict:
        if not data: return {"summary":"No data returned."}
        
        # Special rendering for enhanced features
        if "source_code" in data:
            return LSPResponseParser.render_enhanced_definition(data)
        if "incoming_calls" in data and "outgoing_calls" in data:
            return LSPResponseParser.render_full_call_hierarchy(data)

        # Basic functionality rendering
        parsers = {
            "get_references": LSPResponseParser.parse_references,
            "get_hover": LSPResponseParser.parse_hover,
            "get_document_symbols": LSPResponseParser.parse_document_symbols,
            "get_workspace_symbols": LSPResponseParser.parse_workspace_symbols,
            "get_document_highlights": LSPResponseParser.parse_document_highlights,
        }
        parser = parsers.get(command, LSPResponseParser.parse_default)
        return parser(data)

    @staticmethod
    def parse_error(error_data: dict) -> str:
        """Parse LSP error response"""
        code = error_data.get('code')
        message = error_data.get('message', 'Unknown LSP error')
        return f"LSP Server Error (Code {code}): {message}"

    @staticmethod
    def parse_empty_result(command: str) -> str:
        """Return readable string for 'null' or '[]' results"""
        return f"No results found for '{command}'."

    @staticmethod
    def parse_default(data) -> dict:
        """Default parser for unhandled (but successful) responses"""
        summary = f"Received an unparsed successful response for a command."
        # Return raw data because we don't know how to parse it
        return {"summary": summary, "raw_data": data}


    @staticmethod
    def render_enhanced_definition(data: dict) -> dict:
        summary = data.get("summary", "Definition found.")
        locs = data.get("locations", [])
        code_block = data.get("source_code", "")
        
        out = f"{summary}\n"
        if code_block:
            out += "\n--- SOURCE CODE START ---\n"
            out += code_block
            out += "\n--- SOURCE CODE END ---\n"
        else:
            out += "\n(Note: Could not automatically extract source code range via Document Symbols.)"
        return {"summary": out.strip()}

    @staticmethod
    def render_full_call_hierarchy(data: dict) -> dict:
        target = data.get("target", {})
        incoming = data.get("incoming_calls", [])
        outgoing = data.get("outgoing_calls", [])
        
        out = f"Call Hierarchy Analysis for: {target.get('name', 'Unknown')}\n"
        out += f"Location: {target.get('detail', '')} {target.get('uri', '')}\n\n"
        
        out += "=== Incoming Calls (Who calls this?) ===\n"
        if not incoming:
            out += "  None found.\n"
        else:
            for item in incoming:
                caller = item.get('from', {})
                out += f"- Caller: {caller.get('name')} ({LSPResponseParser._format_kind(caller.get('kind'))})\n"
                out += f"  Location: {uri_to_path(caller.get('uri', ''))} line {caller.get('range', {}).get('start', {}).get('line', 0)+1}\n"

        out += "\n=== Outgoing Calls (Who does this call?) ===\n"
        if not outgoing:
            out += "  None found.\n"
        else:
            for item in outgoing:
                callee = item.get('to', {})
                out += f"- Callee: {callee.get('name')} ({LSPResponseParser._format_kind(callee.get('kind'))})\n"
                out += f"  Location: {uri_to_path(callee.get('uri', ''))} line {callee.get('range', {}).get('start', {}).get('line', 0)+1}\n"
        
        return {"summary": out.strip()}

    @staticmethod
    def _format_kind(kind):
        # Simple mapping; complete mapping is quite long
        m = {
            1: "File", 2: "Module", 3: "Namespace", 4: "Package",
            5: "Class", 6: "Method", 7: "Property", 8: "Field",
            9: "Constructor", 10: "Enum", 11: "Interface", 12: "Function",
            13: "Variable", 14: "Constant", 15: "String", 16: "Number",
            17: "Boolean", 18: "Array", 19: "Object", 20: "Key",
            21: "Null", 22: "EnumMember", 23: "Struct", 24: "Event",
            25: "Operator", 26: "TypeParameter"
        }       
        return m.get(kind, f"Kind({kind})")

    @staticmethod
    def parse_hover(result: dict) -> dict:
        """Parse 'textDocument/hover'"""
        contents = result.get('contents', {})
        # 'contents' can be string, MarkedString, or MarkupContent
        if isinstance(contents, str):
            value = contents
        elif isinstance(contents, dict):
            value = contents.get('value', 'No hover information available.')
        else:
            value = 'No hover information available.'
        
        # Clean up pyright's markdown
        value = value.strip().strip('```python').strip('```').strip()
        summary = f"Hover Info:\n{value}"
        return {"summary": summary}

    @staticmethod
    def parse_definition_or_links(result: Any) -> dict:
        """Parse 'textDocument/definition' along with 'declaration', 'typeDefinition', 'implementation'"""
        if not result:
            return {"summary": "No definition found."}
        
        if not isinstance(result, list):
            result = [result]  # Standardize into list

        locations = []
        for item in result:
            if 'targetUri' in item:  # This is a LocationLink
                locations.append(_format_location_link(item))
            elif 'uri' in item:  # This is a Location
                locations.append(_format_location(item))

        if not locations:
            return {"summary": "No definition found."}

        summary = "Found definition at:\n"
        summary += "\n".join(f"- {loc}" for loc in locations)
        return {"summary": summary, "locations": locations}
        
    @staticmethod
    def parse_references(result: list) -> dict:
        """Parse 'textDocument/references'"""
        if not result:
            return {"summary": "No references found."}
        
        # 1. Group by file URI
        files_map = {}
        for loc in result:
            uri = loc.get('uri', '')
            if not uri: continue
            
            if uri not in files_map:
                files_map[uri] = []
            files_map[uri].append(loc)
            
        count = len(result)
        summary = f"Found {count} reference(s) across {len(files_map)} file(s):\n"
        
        # 2. Sort by filename to ensure stable output
        sorted_uris = sorted(files_map.keys())
        
        formatted_locations = []
        
        for uri in sorted_uris:
            # Get a relatively short file path as title
            path_str = _uri_to_path_str(uri)
            summary += f"\n{path_str}:\n"
            
            # 3. Get all references for this file and sort by (line, character)
            file_refs = files_map[uri]
            file_refs.sort(key=lambda x: (
                x.get('range', {}).get('start', {}).get('line', 0),
                x.get('range', {}).get('start', {}).get('character', 0)
            ))
            
            for ref in file_refs:
                # Reuse _format_range to generate position description
                range_obj = ref.get('range', {})
                pos_str = _simple_format_range(range_obj)
                
                # Indented output, omitting duplicate filenames
                summary += f"  - {pos_str}\n"
                
                # Keep original full description for programmatic use
                formatted_locations.append(f"{path_str} {pos_str}")
            
        return {"summary": summary.strip(), "locations": formatted_locations}
    
    @staticmethod
    def _build_summary_from_tree(nodes: list, indent: str = "") -> str:
        """(New) Helper function: Build string summary from parsed structured symbol tree"""
        result_parts = []
        # Use stack to simulate recursion: stores (node object, current indentation)
        # reversed so stack pop order matches original list order
        stack = [(node, indent) for node in reversed(nodes)]
        
        # Used to detect cyclic references, stores memory addresses of processed nodes
        visited_ids = set()

        while stack:
            current_node, current_indent = stack.pop()
            
            # 1. Cyclic reference protection: if node already processed, skip it
            node_id = id(current_node)
            if node_id in visited_ids:
                continue
            visited_ids.add(node_id)

            # 2. Build current node string
            if current_node.get('kind') == 'Variable' or  current_node.get('kind') == 'Constant':
                # result_parts.append(f"{current_indent}- {current_node.get('name')} ({current_node.get('kind')}) \n")
                pass
            else:
                location_str = _format_range(current_node.get('range'))
                result_parts.append(f"{current_indent}- {current_node.get('name')} ({current_node.get('kind')}) at {location_str} \n")

            # 3. Push child nodes onto stack
            # Note: children need to be pushed in reverse order so they are popped in correct sequence
            children = current_node.get('children')
            if children:
                next_indent = current_indent + "  "
                for child in reversed(children):
                    stack.append((child, next_indent))

        return "".join(result_parts)

    @staticmethod
    def _parse_hierarchical_symbols(symbols: list) -> list:
        """(New) Helper function: Recursively parse LSP 2.0+ (nested 'children') symbols"""
        parsed_list = []
        for symbol in symbols:
            node = {
                'name': symbol.get('name', 'unknown'),
                'kind': LSPResponseParser._format_kind(symbol.get('kind', 0)),
                'range': symbol.get('location', {}).get("range", {}),
                'children': []
            }
            if symbol.get('children'):
                node['children'] = LSPResponseParser._parse_hierarchical_symbols(symbol['children'])
            parsed_list.append(node)
        return parsed_list

    @staticmethod
    def _parse_flat_symbols(symbols: list) -> list:
        """(New) Helper function: Recursively parse LSP 1.0 (flat 'containerName') symbols"""
        top_level_nodes = []
        nested_map = {}
        all_parsed_nodes = {} # Mapping: original symbol -> parsed node

        # 1. Create parsed versions of all nodes and establish a mapping
        for symbol in symbols:
            node = {
                'name': symbol.get('name', 'unknown'),
                'kind': LSPResponseParser._format_kind(symbol.get('kind', 0)),
                'range': symbol.get('location', {}).get('range', {}),
                'children': []
            }
            # Use id(symbol) as a unique key
            all_parsed_nodes[id(symbol)] = node

        # 2. Establish parent-child relationships
        for symbol in symbols:
            parsed_node = all_parsed_nodes[id(symbol)]
            container = symbol.get('containerName')

            if container:
                # Add this node to its containerName list
                nested_map.setdefault(container, []).append(parsed_node)
            else:
                # This is a top-level node
                top_level_nodes.append(parsed_node)

        # 3. Attach child node lists to correct parent nodes
        # (Key step: Connect lists in 'nested_map' to parent nodes in 'all_parsed_nodes')
        for node in all_parsed_nodes.values():
            if node['name'] in nested_map:
                node['children'] = nested_map[node['name']]
        
        return top_level_nodes

    @staticmethod
    def parse_document_symbols(result: list) -> dict:
        """(Modified) Parse 'textDocument/documentSymbol'"""
        if not result:
            return {"summary": "No symbols found in this document."}
        
        structured_symbols = []
        summary = ""

        # Check if it's nested structure (LSP 2.0+) or flat structure (LSP 1.0)
        # Assumption: if the first element has 'children', the entire list follows this style
        if result and 'children' in result[0]:
            # safe_print("in _parse_hierarchical_symbols")
            summary = "Document Structure (Hierarchical):\n"
            structured_symbols = LSPResponseParser._parse_hierarchical_symbols(result)
        else:
            # safe_print("in _parse_flat_symbols")
            summary = "Document Symbols:\n"
            structured_symbols = LSPResponseParser._parse_flat_symbols(result)
        
        # Build summary from the structured tree
        summary += LSPResponseParser._build_summary_from_tree(structured_symbols)
        
        # (Fixed)
        # Return LLM-friendly 'summary' string and 'symbols' structured list
        return {"summary": summary.strip(), "symbols": structured_symbols}


    @staticmethod
    def parse_signature_help(result: dict) -> dict:
        """Parse 'textDocument/signatureHelp'"""
        signatures = result.get('signatures', [])
        if not signatures:
            return {"summary": "No signature help available."}
        
        active_sig_index = result.get('activeSignature', 0)
        active_param_index = result.get('activeParameter', 0)
        
        active_sig = signatures[active_sig_index]
        label = active_sig.get('label', 'No signature label')
        
        summary = f"Signature:\n{label}\n"
        
        params = active_sig.get('parameters', [])
        if params and len(params) > active_param_index:
            active_param = params[active_param_index]
            param_label = active_param.get('label', 'unknown parameter')
            summary += f"Active Parameter: {param_label}"
        
        return {"summary": summary.strip(), "signature_info": result}


    @staticmethod
    def parse_workspace_symbols(result: list, limited_lens: int = 100) -> dict:
        """Parse 'workspace/symbol'"""
        if not result:
            return {"summary": "No workspace symbols found for the query."}
        
        # 1. Group by file URI
        files_map = {}
        for sym in result:
            uri = sym.get('location', {}).get('uri', '')
            if not uri: continue
            
            if uri not in files_map:
                files_map[uri] = []
            files_map[uri].append(sym)
            
        count = len(result)
        summary = f"Found {count} workspace symbol(s) across {len(files_map)} file(s):\n"
        
        # 2. Sort by filename to ensure stable output
        sorted_uris = sorted(files_map.keys())
        
        symbols_list = [] # Keep structured data just in case, though summary is primarily used here
        
        for uri in sorted_uris[:limited_lens]:
            # Get a relatively short file path
            path_str = _uri_to_path_str(uri)
            summary += f"\n {path_str} :\n"
            
            # 3. Get all symbols for this file and sort by line number
            file_symbols = files_map[uri]
            file_symbols.sort(key=lambda x: x.get('location', {}).get('range', {}).get('start', {}).get('line', 0))
            
            for sym in file_symbols:
                name = sym.get('name', 'unknown')
                kind = LSPResponseParser._format_kind(sym.get('kind', 0))
                
                # Get line number (LSP is 0-indexed, converted to 1-indexed for display)
                start_line = sym.get('location', {}).get('range', {}).get('start', {}).get('line', 0) + 1
                
                # Format: "124:   name (Kind)"
                # <4 means left-aligned with width 4, keeping alignment
                entry_str = f"{start_line:<4}: {name} ({kind})"
                summary += f"{entry_str}\n"
                symbols_list.append(entry_str)

        TRUNCATED_MESSAGE = (
            "<response clipped><NOTE>To save on context only part of searched symbols has been "
            "shown to you. You can use grep -n with specific syntax structures within symbols" 
            "(such as `def function`, of `class my_class`) to locate specific symbol.</NOTE>"
        )
        if count > limited_lens:
            summary += TRUNCATED_MESSAGE
        return {"summary": summary.strip(), "symbols": symbols_list}


    @staticmethod
    def parse_document_highlights(result: list) -> dict:
        """Parse 'textDocument/documentHighlight'"""
        if not result:
            return {"summary": "No highlights found."}
            
        count = len(result)
        summary = f"Found {count} related highlight(s) in this document:\n"
   
        locations = []
        for item in result:
            range = item.get('range', {})
            kind = LSPResponseParser._format_kind(item.get('kind', 0))
            pos_str = _simple_format_range(range)

            # pos_str = _format_range(range)
            summary += f"- ({kind}) at {pos_str}\n"
            locations.append(pos_str)
            
        return {"summary": summary.strip(), "highlight_locations": locations}


# --- Communication Client ---
class LSPToolClient:
    """Handles low-level socket communication with the Daemon, supporting single requests"""
    def __init__(self):
        self.port = self._get_daemon_port()

    def _get_daemon_port(self) -> int:
        """Read daemon port number from the port file"""
        port_file = PORT_FILE_ENV_VAR
        if not port_file:
            raise ConnectionError(f"PORT_FILE_ENV_VAR {PORT_FILE_ENV_VAR} is not set.")
            
        try:
            with open(port_file, 'r') as f:
                port = int(f.read().strip())
                if port <= 0 or port > 65535:
                    raise ValueError("Invalid port number.")
                return port
        except FileNotFoundError:
            raise ConnectionError(f"LSP port file {port_file} not found. Is the daemon running?")
        except Exception as e:
            raise ConnectionError(f"Error reading LSP port file {port_file}: {e}")
        
    async def send_request(self, request_data: dict) -> LSPResult:
        reader, writer = None, None
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(HOST, self.port), timeout=5
            )
            
            # Encode
            json_data = json.dumps(request_data).encode('utf-8')
            cmd = request_data['command']
            writer.write(struct.pack('!I', len(json_data)))
            writer.write(json_data)
            await writer.drain()
            safe_print(f"Tool: Sent command '{cmd}' to daemon.")

            # Decode
            safe_print("Tool: Waiting for response from daemon...")
            len_data = await reader.readexactly(4)
            msg_len = struct.unpack('!I', len_data)[0]

            response_json = await reader.readexactly(msg_len)
            # safe_print(response_json.decode('utf-8'), file=sys.stdout)

            response_dict = json.loads(response_json.decode('utf-8'))

            if response_dict.get("status_code") == "error":
                return LSPResult(error=response_dict.get("error"))
            return LSPResult(result=response_dict.get("result"))


        except Exception as e:
            return LSPResult(error=str(e))

        finally:
            if writer:
                writer.close()
                if sys.version_info >= (3, 7):
                    try: await writer.wait_closed()
                    except: pass


# --- Core Business Logic (Enhanced) ---
class EnhancedLSPTool:
    def __init__(self):
        self.client = LSPToolClient()
        self.parser = LSPResponseParser()
    async def run_command(self, args) -> LSPResult:
        cmd = args.command
        # 1. Enhanced: Definition / Type / Implementation (with source code)
        if cmd in ["get_definition", "get_type_definition", "get_implementation"]:
            if args.line:
                args.line -= 1
                args.character = self._get_character(args.file_path, args.line, args.symbol)
                del args.symbol
            else:
                return LSPResult(error=f"'file_path', 'line' (1-indexed), and symbol are required for {cmd}")

            return await self.handle_definition_with_code(cmd, args)
        
        # 2. Aggregated: Call Hierarchy
        if cmd == "get_call_hierarchy":
            if args.line:
                args.line -= 1
                args.character = self._get_character(args.file_path, args.line, args.symbol)
                del args.symbol
            else:
                return LSPResult(error=f"'file_path', 'line' (1-indexed), and symbol are required for {cmd}")

            return await self.handle_full_call_hierarchy(args)
        
        # 3. Passthrough: Other standard commands
        if cmd in ["get_hover", "get_references", "get_document_highlights"]:
            if args.line:
                args.line -= 1
                args.character = self._get_character(args.file_path, args.line, args.symbol)
                del args.symbol
            else:
                return LSPResult(error=f"'file_path', 'line' (1-indexed), and symbol are required for {cmd}")
        if cmd in ["get_workspace_symbols"]:
            if args.file_path:
                del args.file_path
                safe_print("file_path is NOT allowed for `get_workspace_symbols` method, it searches the whole project as default.", file=sys.stdout)
        lsp_response = await self.client.send_request(vars(args))
        parsed_result_obj = self.parser.parse(command=cmd,data=lsp_response.result)
        result = LSPResult(result=parsed_result_obj)
        return result

    async def handle_definition_with_code(self, cmd, args) -> LSPResult:
        """Perform Go to Definition and use get_document_symbols to read full code block"""
        # Step 1: Get location
        resp = await self.client.send_request(vars(args))
        if resp.error: 
            safe_print(f"Tool: command '{cmd}' return error.")
            return resp
        
        result_data = resp.result
        if not result_data:
            safe_print(f"Tool: command '{cmd}' no result data")
            return resp # No definition found, return directly
        
        raw_locations = result_data if isinstance(result_data, list) else [result_data]

        first_loc = raw_locations[0]
        target_uri = first_loc.get("uri") or first_loc.get("targetUri")
        target_range = first_loc.get("range") or first_loc.get("targetSelectionRange")
        
        if not target_uri: 
            safe_print(f"Tool: command '{cmd}' cannot locate uri:{target_uri}")
            return resp # Unable to locate file

        # Step 2: Read file structure (Symbols)
        file_path = str(uri_to_path(target_uri))
        
        # file_path = args.file_path
        sym_req = {
            "command": "get_document_symbols", 
            "file_path": file_path
        }
        sym_resp = await self.client.send_request(sym_req)
        
        code_content = ""
        if not sym_resp.error:
            # Step 3: Find smallest node in Symbols containing target_range
            symbols = sym_resp.result # list[]
            # Note: daemon's get_document_symbols returns a list directly, not filtered via Parser into plain text
            
            target_line = target_range.get('start', {}).get('line', 0)
            # target_line = args.line
            
            # Recursively find symbol matching line number
            found_symbol = self._find_symbol_at_line(symbols, target_line)
            
            if found_symbol:
                # Step 4: Read file content
                try:
                    full_range = found_symbol.get("location", {}).get("range") or found_symbol.get("range")
                    if full_range:
                        start_line = full_range['start']['line']
                        end_line = full_range['end']['line']
                        with open(file_path, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                            # Simple boundary checks
                            start_line = max(0, start_line)
                            end_line = min(len(lines), end_line + 1) 

                            # (New) Add line numbers, similar to cat -n
                            numbered_lines = []
                            for idx, line in enumerate(lines[start_line:end_line]):
                                current_line_num = start_line + 1 + idx
                                numbered_lines.append(f"{current_line_num:>6}\t{line}")
                            code_content = "".join(numbered_lines)
                except Exception as e:
                    code_content = f"Error reading file: {e}"
        
        # Construct enhanced return result
        # We keep the original summary generation logic but inject source_code
        enhanced_result = {
            "summary": self.parser.parse_definition_or_links(raw_locations).get("summary"),
            "locations": raw_locations,
            "source_code": code_content
        }

        parsed_result_obj = self.parser.parse(command=cmd,data=enhanced_result)

        return LSPResult(result=parsed_result_obj)

    def _get_character(self, file_path:str = "", line: int = 0, symbol: str = ""):
        """
            Get the character offset (0-indexed) of the first occurrence of a symbol in a given file and line.
            Args:
                file_path: Absolute path to the file.
                line: Line number (0-indexed).
                symbol: The string symbol to search for.

            Returns:
                The starting character offset (int) in the line, or None if error/not found.
        """
        # --- 1. Parameter validation and preparation ---
        if not file_path or not symbol:
            safe_print("File path or search symbol cannot be empty.")
            return None
        if line < 0:
            safe_print(f"Line number must be non-negative, got: {line}")
            return None
        # --- 2. File access and I/O error handling ---
        try:
            if not os.path.exists(file_path):
                safe_print(f"File not found at path: {file_path}")
                return None
            
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
        except IOError as e:
            safe_print(f"Error reading file {file_path}: {e}")
            return None
        except Exception as e:
            # Catch other possible I/O errors, such as permissions
            safe_print(f"An unexpected error occurred during file reading: {e}")
            return None
            
        # --- 3. Line index boundary checks ---
        if line >= len(lines):
            # If the requested line number exceeds the total number of lines in the file
            safe_print(
                f"Requested line index ({line}) is out of bounds (Total lines: {len(lines)})."
            )
            return None
            
        # --- 4. Core logic and result processing ---
        
        # Remove trailing newline for precise matching
        code_content = lines[line].rstrip('\r\n')
        
        # Use find method to locate the first occurrence of the symbol
        start = code_content.find(symbol)
        if start == -1:
            # If symbol is not found, find() returns -1
            safe_print(f"Symbol '{symbol}' not found on line {line} in {file_path}.")
            return None 
        return start

    def _find_symbol_at_line(self, symbols: Any, line: int) -> Optional[dict]:
        """Find the deepest symbol in the tree containing that line"""
        best_match = None
        
        # LSP symbols can be flat (with containerName) or nested (with children)
        # Perform a simple general traversal here
        stack = list(symbols)
        while stack:
            sym = stack.pop()
            # Check range
            r = sym.get("location", {}).get("range") or sym.get("range")
            if not r: continue
            
            s_line = r['start']['line']
            e_line = r['end']['line']
            
            if s_line == line:
                # This is a candidate; check if there are deeper child nodes
                best_match = sym
                # Push children to stack to continue searching (if nested structure)
                if "children" in sym and sym["children"]:
                    stack.extend(sym["children"])
        
        return best_match


    async def handle_full_call_hierarchy(self, args) -> LSPResult:
        """Merge Prepare -> Incoming -> Outgoing"""
        # Step 1: Prepare
        prep_args = {
            "command": "prepare_call_hierarchy",
            "file_path": args.file_path, "line": args.line, "character": args.character
        }
        prep_resp = await self.client.send_request(prep_args)
        if prep_resp.error or not prep_resp.result:
            return LSPResult(error=f"Could not prepare hierarchy: {prep_resp.error}")
        
        # Prepare usually returns a list; take the first item
        items = prep_resp.result
        if not items: return LSPResult(error="Symbol not valid for call hierarchy.")
        root_item = items[0]
        
        # Step 2 & 3: Parallelize getting Incoming and Outgoing
        # Does item need to be converted to json string before sending to daemon? No, tool client sends dict directly.
        # Check daemon code: get_incoming_calls requires 'item' parameter.
        
        task_in = self.client.send_request({"command": "get_incoming_calls", "item": root_item})
        task_out = self.client.send_request({"command": "get_outgoing_calls", "item": root_item})
        
        res_in, res_out = await asyncio.gather(task_in, task_out)
        
        full_result = {
            "target": root_item,
            "incoming_calls": res_in.result if not res_in.error else [],
            "outgoing_calls": res_out.result if not res_out.error else []
        }

        parsed_result_obj = self.parser.parse(command="get_call_hierarchy",data=full_result)

        return LSPResult(result=parsed_result_obj)



ALLOWED_LSP_COMMANDS = [
    "open_document", "change_document", "did_save", "did_close",
    "did_create_files", "did_delete_files", "did_rename_files",
    "get_definition", "get_declaration", "get_type_definition",
    "get_implementation", "get_references", "get_hover",
    "get_signature_help", "get_document_highlights",
    "get_call_hierarchy",
    "get_document_symbols","get_workspace_symbols",
    "daemon_shutdown"
]

# --- Main ---
async def main_async():
    parser = argparse.ArgumentParser(
        description="LSP Tool Client",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "command",
        type=str,
        choices=ALLOWED_LSP_COMMANDS,
        help="The LSP command to run."
    )
    # Optional parameters
    parser.add_argument("--file_path", type=str, default=None, help="Absolute path to the file for the command.")
    parser.add_argument("--symbol", type=str, default=None, help="The name of a Python code entity to locate. Includes identifiers for Classes, Functions, Methods, Variables, Fields, and Modules (e.g., 'MyClass', 'process_data', 'MAX_RETRIES').")
    parser.add_argument("--line", type=int, default=None, help="Line number (1-indexed) for positional commands.")
    parser.add_argument("--character", type=int, default=None, help="Character offset (1-indexed) for positional commands.")
    parser.add_argument("--new_text", type=str, default=None, help="Required for 'change_document', provides the new file content.")
    parser.add_argument("--query", type=str, default=None, help="Required for 'get_workspace_symbols', the string to search for.")

    
    # --- File change notifications ---
    parser.add_argument(
        "--files", 
        type=str, 
        nargs='+',  
        default=None, 
        help="List of file paths for 'did_create_files' or 'did_delete_files'."
    )
    parser.add_argument(
        "--old_path", 
        type=str, 
        default=None, 
        help="The old file path for 'did_rename_files'."
    )
    parser.add_argument(
        "--new_path", 
        type=str, 
        default=None, 
        help="The new file path for 'did_rename_files'."
    )
    # ---
    
    args = parser.parse_args()
    
    tool = EnhancedLSPTool()
    
    try:
        # Run command
        lsp_result = await tool.run_command(args)
        response_str = lsp_result.to_text().encode('utf-8')
        # 4. Output response to stdout
        safe_print(response_str.decode('utf-8'), file=sys.stdout)

        # if lsp_result.error:
        #     print(json.dumps({"status": "error", "message": lsp_result.error}, indent=2))
        # else:
        #     # Parse and print results friendly to humans/LLMs
        #     parsed_text = LSPResponseParser.parse(args.command, lsp_result.result)
        #     print(parsed_text)
            
    except Exception as e:
        tb_str = traceback.format_exc()
        result = LSPResult(error=f"Fatal asyncio error: {e}\n{tb_str}")
        safe_print(result.to_json_str(), file=sys.stdout)
    except KeyboardInterrupt:
        safe_print("LSP tool interrupted by user.", file=sys.stderr)


def main():
    try:
        run_asyncio(main_async())
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()