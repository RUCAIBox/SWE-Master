#!/root/.venv/bin/python
# -*- coding: utf-8 -*-

"""
LSP Daemon for Python Code Intelligence
This script runs as a background daemon, managing a persistent Language Server (Pyright) 
and providing an asynchronous API via sockets for code analysis.
"""
import subprocess
import json
import os
import asyncio
import orjson
import traceback
import io
import sys
import struct
import signal
# --- Configuration ---
from pathlib import Path
from urllib.parse import urlparse, unquote
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
import atexit
from idna import encode

# --- Configuration ---
PORT_FILE_ENV_VAR = "/var/tmp/lsp_port_session_abc.pid"
HOST = '127.0.0.1'
PORT = 0
LSP_HISTORY_FILE = "/tmp/lsp_history.json"
DEFAULT_PROJECT_ROOT = "/testbed"
DEFAULT_LSP_COMMAND = os.environ.get("LSP_COMMAND", "pyright-langserver --stdio")

# Global variables for signal handling
shutdown_event = None
main_loop = None

# --- Python 3.5/3.6 Compatibility Functions ---
def run_asyncio(coro):
    """Compatible with asyncio.run() (Python 3.7+)"""
    if sys.version_info >= (3, 7):
        return asyncio.run(coro)
    else:
        # Python 3.5/3.6: Manually manage event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            try:
                # Cancel all outstanding tasks
                pending = asyncio.Task.all_tasks(loop)
                for task in pending:
                    task.cancel()
                # Run event loop until all tasks are cancelled
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception as e:
                safe_print(f"Error during cleanup: {e}")
            finally:
                loop.close()

def create_task_compat(coro, loop=None):
    """Compatible with asyncio.create_task() (Python 3.7+)"""
    if loop is None:
        loop = asyncio.get_event_loop()
    
    if sys.version_info >= (3, 7):
        return asyncio.create_task(coro)
    else:
        return loop.create_task(coro)

# --- UTF-8 & Logging ---
if hasattr(sys.stdout, 'buffer'):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", write_through=True)
else:
    sys.stderr.write("sys.stderr does not have a 'buffer' attribute.\n")

def safe_print(x, file=sys.stderr):
    """All logging goes to stderr to keep IPC clean"""
    try:
        print(x, file=file, flush=True)
    except UnicodeEncodeError:
        print(x.encode("utf-8", errors="replace").decode("utf-8", errors="replace"), file=file, flush=True)
    except Exception:
            try:
                # If it fails, attempt to escape and print
                # Using repr() safely prints any binary data or invisible characters
                print(repr(x), file=file, flush=True)
            except Exception:
                # If even repr fails (rare), do nothing to ensure daemon stability
                pass

# --- Daemonization ---
def daemonize():
    """
    Daemonizes the current process using the double fork technique.
    This ensures the process is completely detached from the controlling terminal 
    and won't receive signals when the parent process exits.
    """
    # First fork
    try:
        pid = os.fork()
        if pid > 0:
            # Parent process exits
            sys.exit(0)
    except OSError as e:
        safe_print(f"Daemon: First fork failed: {e}")
        sys.exit(1)
    
    # Detach from parent environment
    os.chdir('/')
    os.setsid()  # Create new session, become session leader
    os.umask(0)
    
    # Second fork
    try:
        pid = os.fork()
        if pid > 0:
            # Second parent process exits
            sys.exit(0)
    except OSError as e:
        safe_print(f"Daemon: Second fork failed: {e}")
        sys.exit(1)
    
    # Current process is now a true daemon
    # Redirect standard file descriptors
    # Flush all buffers first
    sys.stdout.flush()
    sys.stderr.flush()
    
    try:
        # dev_null_r = os.open('/dev/null', os.O_RDONLY)    
        # os.dup2(dev_null_r, 0)
        # os.close(dev_null_r)
        dev_null = open('/dev/null', 'r')
        os.dup2(dev_null.fileno(), sys.stdin.fileno())
    except Exception as e:
        safe_print(f"Daemon: Warning - failed to redirect stdin: {e}")

    safe_print("Daemon: Process successfully daemonized (all fds cleaned)")

    
# --- Helper Functions & Result Class ---
def uri_to_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme in ('', 'file'):
        p = unquote(parsed.path)
        if os.name == 'nt' and p.startswith('/') and len(p) > 2 and p[2] == ':':
            p = p[1:]
        return Path(p)
    raise ValueError("Unsupported URI scheme: " + parsed.scheme)

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
        
        # 3. Handle special data (Context Data)
        # Some commands (like prepare_call_hierarchy) return 'item_data' as required parameters for subsequent steps.
        # We keep it as a single-line JSON to facilitate model copying/referencing.
        # Data already shown in summary like 'symbols' or 'locations' are ignored to reduce noise.
        if "item_data" in self.result:
            try:
                # Use orjson to serialize into a compact JSON string
                data_str = orjson.dumps(self.result["item_data"]).decode('utf-8')
                output_parts.append(f"\n[DATA] item_data: {data_str}")
            except Exception:
                output_parts.append(f"\n[DATA] item_data: {self.result['item_data']}")

        # 4. Fallback: If neither summary nor special data exists, print raw results
        if not output_parts:
            try:
                fallback_json = json.dumps(self.result, indent=2, default=str)
                output_parts.append(fallback_json)
            except Exception:
                output_parts.append(str(self.result))

        return "[status_code]: \n {} \n".format(self.status_code) +  "Execution output of [lsp_tool]: \n" + "\n".join(output_parts)

    def __str__(self):
        return self.to_text()

    def to_dict(self) -> dict:
        if self.error:
            return {"status_code": "error", "error": self.error}
        return {"status_code": "success", "result": self.result}

    def to_json_str(self) -> str:
        # (Modified) Use orjson for faster serialization and Path object handling
        try:
            return orjson.dumps(self.to_dict()).decode('utf-8')
        except (TypeError, orjson.JSONEncodeError):
            # Fallback for complex types orjson might not handle (though it's rare)
            return json.dumps(self.to_dict(), indent=2, default=str)

    def __json_str__(self):
        return self.to_json_str()



class LSPClient:
    """An ASYNCHRONOUS LSP client to interact with a language server."""

    def __init__(self, lsp_command: List[str], project_root: str):
        self.project_root = project_root
        self.project_root_uri = Path(project_root).as_uri()
        self.lsp_command = lsp_command
        safe_print(f"Daemon: Initializing LSP server for project root: {self.project_root_uri}")

        self._doc_versions = {}
        self._doc_texts = {}
        self._doc_locks = {}
        self._debounce_timers = {}
        self._history = self.load_history()

        self._lsp_process: Optional[asyncio.subprocess.Process] = None
        self._message_id = 1
        self._pending: Dict[int, asyncio.Future] = {}
        self._diagnostics: Dict[str, list] = {}
        self._reader_task: Optional[asyncio.Task] = None
        self._writer_task: Optional[asyncio.Task] = None
        self._stderr_task: Optional[asyncio.Task] = None
        self._write_queue: asyncio.Queue = asyncio.Queue()
        self._loop = asyncio.get_event_loop()

    async def start(self):
        """Start LSP server and listener tasks"""
        safe_print(f"Daemon: Starting LSP subprocess: {' '.join(self.lsp_command)}")
        self._lsp_process = await asyncio.create_subprocess_exec(
            *self.lsp_command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.project_root,
            start_new_session=True, 
        )
        self._reader_task = create_task_compat(self._read_loop(), self._loop)
        self._writer_task = create_task_compat(self._writer_loop(), self._loop)
        self._stderr_task = create_task_compat(self._read_stderr(), self._loop)

        await self.initialize()
        return self

    async def close(self):
        """Close LSP server and tasks"""
        await self.shutdown_and_exit()

    async def _writer_loop(self):
        try:
            while True:
                message = await self._write_queue.get()
                if message is None:
                    break
                self._lsp_process.stdin.write(message)
                await self._lsp_process.stdin.drain()
        except Exception as e:
            safe_print(f"Daemon: LSPClient writer_loop exception: {e}")
            self._fail_pending(e)

    async def _read_loop(self):
        reader = self._lsp_process.stdout
        try:
            while True:
                headers = await reader.readuntil(b"\r\n\r\n")
                if not headers:
                    safe_print("Daemon: LSP stdout closed (EOF)")
                    break
                
                header_str = headers.decode('utf-8')
                content_length = 0
                for line in header_str.split("\r\n"):
                    if line.lower().startswith("content-length:"):
                        content_length = int(line.split(":")[1].strip())
                        break

                if content_length > 0:
                    body = await reader.readexactly(content_length)
                    if not body:
                        safe_print("Daemon: No body received from LSP (EOF).")
                        break

                    try:
                        response = orjson.loads(body.decode('utf-8'))
                    except Exception as e:
                        safe_print(f"Daemon: JSON decode error: {e}, body={body[:200]!r}")
                        continue
                    
                    if 'id' in response:
                        fut = self._pending.pop(response['id'], None)
                        if fut and not fut.done():
                            fut.set_result(response)
                    else:
                        if response.get("method") == "textDocument/publishDiagnostics":
                            params = response.get("params", {})
                            uri = params.get("uri")
                            if uri:
                                self._diagnostics.setdefault(uri, []).append(response)
                        else:
                            safe_print(f"Daemon: Received server notification: {json.dumps(response, indent=2)}")

        except asyncio.IncompleteReadError:
            safe_print(f"Daemon: [LSP] stdout stream closed.")
        except Exception as e:
            safe_print(f"Daemon: LSPClient read_loop exception: {e}")
            self._fail_pending(e)

    async def _read_stderr(self):
        """Continuously read LSP server stderr, improved error handling"""
        
        reader = self._lsp_process.stderr
        buffer_size = 4096
        try:
            while True:
                # Use small chunks to avoid blocking
                if reader.at_eof():
                    break
                
                # Simple blocking read; wait if no data instead of timeout error
                chunk = await reader.read(buffer_size)
                if not chunk:
                    break
                
                try:
                    # Attempt decoding, ignore errors if non-utf-8 characters are present
                    text = chunk.decode('utf-8', errors='replace')
                    if text.strip():
                         safe_print(f"Daemon: [LSP-STDERR] {text.strip()}")
                except Exception:
                    # Final defense to prevent binary data from breaking logs
                    safe_print(f"Daemon: [LSP-STDERR RAW] {repr(chunk)}")                
             
        except asyncio.CancelledError:
            safe_print("Daemon: [LSP-STDERR] Task cancelled.")
            raise
        except Exception as e:
            safe_print(f"Daemon: Error reading LSP stderr: {e}")
    
    def _fail_pending(self, exc: Exception):
        safe_print(f"Daemon: LSPClient failing all pending requests due to: {exc}")
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(exc)
        self._pending.clear()

    def _make_lsp_message(self, payload: dict) -> bytes:
        # (Modified) Use orjson for faster serialization
        body_bytes = orjson.dumps(payload)
        content_length = len(body_bytes)
        header = f"Content-Length: {content_length}\r\n\r\n"
        return header.encode('utf-8') + body_bytes

    async def send_request(self, method: str, params: Any, timeout: int = 600) -> dict:
        message_id = self._message_id
        self._message_id += 1
        payload = {
            "jsonrpc": "2.0",
            "id": message_id,
            "method": method,
            "params": params
        }
        message = self._make_lsp_message(payload)

        fut = asyncio.Future()
        self._pending[message_id] = fut

        await self._write_queue.put(message)
        safe_print(f"Daemon: Sent request (ID: {message_id}): {method}")

        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            safe_print(f"Daemon: Timeout waiting for response (ID: {message_id})!")
            self._pending.pop(message_id, None)
            return {"jsonrpc": "2.0", "id": message_id, "error": {"code": -32000, "message": "Request timed out"}}
        except Exception as e:
            safe_print(f"Daemon: Error waiting for response (ID: {message_id}): {e}")
            self._pending.pop(message_id, None)
            return {"jsonrpc": "2.0", "id": message_id, "error": {"code": -32001, "message": f"Client error: {e}"}}

    async def send_notification(self, method, params):
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params
        }
        message = self._make_lsp_message(payload)
        await self._write_queue.put(message)
        safe_print(f"Daemon: Sent notification: {method}")

    async def _ensure_doc_lock(self, uri) -> asyncio.Lock:
        if uri not in self._doc_locks:
            self._doc_locks[uri] = asyncio.Lock()
        return self._doc_locks[uri]
    
    async def _get_next_version(self, uri):
        v = self._doc_versions.get(uri, 0) + 1
        self._doc_versions[uri] = v
        return v

    async def run(
        self,
        command: str,
        file_path: Optional[str] = None,
        line: Optional[int] = None,
        character: Optional[int] = None,
        new_text: Optional[str] = None,
        query: Optional[str] = None,
        include_declaration: bool = True,
        files: Optional[List[str]] = None,
        old_path: Optional[str] = None,
        new_path: Optional[str] = None,
        item: Optional[dict] = None, # (New) Add item for call hierarchy
        **kwargs
    ) -> LSPResult:
        """(Modified) Fully rewritten to integrate response parsing"""
        
        timestamp = datetime.now(timezone.utc).isoformat()
        request_args = {
            "command": command, "file_path": file_path, "line": line, "character": character,
            "new_text": "[...]" if new_text is not None else None, "query": query,
            "include_declaration": include_declaration,
            "files": files, "old_path": old_path, "new_path": new_path,
            "item": item
        }
        
        result: LSPResult = LSPResult() 
        response_data: Optional[dict] = None

        try:
            def check_positional_args():
                if file_path is None or line is None or character is None:
                    return LSPResult(error=f"'file_path', 'line' (1-indexed), and symbol are required for {command}")
                return None

            # 1. Handle Notifications
            if command == "open_document":
                if not file_path:
                    result = LSPResult(error="'file_path' is required for open_document")
                else:
                    await self.open_document(file_path)
                    safe_print("Daemon: 'open_document' received. Waiting 1s for analysis to complete...")
                    await asyncio.sleep(1)
                    safe_print("Daemon: Analysis wait complete.")
                    result = LSPResult(result={"summary": f"Document opened and analysis started for: {file_path}"})

            elif command == "change_document":
                if not file_path or new_text is None:
                    result = LSPResult(error="'file_path' and 'new_text' are required for change_document (full sync)")
                else:
                    await self.change_document(file_path=file_path, new_text=new_text)
                    safe_print("Daemon: 'change_document' received. Waiting 1s for re-analysis...")
                    await asyncio.sleep(1)
                    safe_print("Daemon: Analysis wait complete.")
                    result = LSPResult(result={"summary": f"Document changed and re-analysis started for: {file_path}"})
    
            elif command == "did_save":
                if not file_path:
                    result = LSPResult(error="'file_path' is required for did_save")
                else:
                    await self.did_save(file_path)
                    result = LSPResult(result={"summary": f"Notified server of save for: {file_path}"})

            elif command == "did_close":
                if not file_path:
                    result = LSPResult(error="'file_path' is required for did_close")
                else:
                    await self.did_close(file_path)
                    result = LSPResult(result={"summary": f"Document closed: {file_path}"})

            elif command == "did_create_files":
                if not files:
                    result = LSPResult(error="'files' (a list of paths) is required for did_create_files")
                else:
                    await self.did_create_files(files)
                    result = LSPResult(result={"summary": f"Notified server of file creation: {files}"})

            elif command == "did_delete_files":
                if not files:
                    result = LSPResult(error="'files' (a list of paths) is required for did_delete_files")
                else:
                    await self.did_delete_files(files)
                    result = LSPResult(result={"summary": f"Notified server of file deletion: {files}"})

            elif command == "did_rename_files":
                if not old_path or not new_path:
                    result = LSPResult(error="'old_path' and 'new_path' are required for did_rename_files")
                else:
                    await self.did_rename_files(old_path, new_path)
                    result = LSPResult(result={"summary": f"Notified server of file rename: {old_path} -> {new_path}"})

            # 2. Handle Requests
            elif command == "get_definition":
                err = check_positional_args();
                if err: result = err
                else: response_data = await self.get_definition(file_path, line, character)

            elif command == "get_declaration":
                err = check_positional_args();
                if err: result = err
                else: response_data = await self.get_declaration(file_path, line, character)

            elif command == "get_type_definition":
                err = check_positional_args();
                if err: result = err
                else: response_data = await self.get_type_definition(file_path, line, character)

            elif command == "get_implementation":
                err = check_positional_args();
                if err: result = err
                else: response_data = await self.get_implementation(file_path, line, character)

            elif command == "get_references":
                err = check_positional_args();
                if err: result = err
                else: response_data = await self.get_references(file_path, line, character, include_declaration)

            elif command == "get_hover":
                err = check_positional_args();
                if err: result = err
                else: response_data = await self.get_hover(file_path, line, character)

            elif command == "get_signature_help":
                err = check_positional_args();
                if err: result = err
                else: response_data = await self.get_signature_help(file_path, line, character)

            elif command == "get_document_highlights":
                err = check_positional_args();
                if err: result = err
                else: response_data = await self.get_document_highlights(file_path, line, character)

            elif command == "prepare_call_hierarchy":
                err = check_positional_args();
                if err: result = err
                else: response_data = await self.prepare_call_hierarchy(file_path, line, character)

            elif command == "get_incoming_calls":
                if not item:
                    result = LSPResult(error="'item' (from 'prepare_call_hierarchy') is required for get_incoming_calls")
                else:
                    response_data = await self.get_incoming_calls(item)

            elif command == "get_outgoing_calls":
                if not item:
                    result = LSPResult(error="'item' (from 'prepare_call_hierarchy') is required for get_outgoing_calls")
                else:
                    response_data = await self.get_outgoing_calls(item)

            elif command == "get_document_symbols":
                if not file_path:
                    result = LSPResult(error="'file_path' is required for get_document_symbols")
                else:
                    response_data = await self.get_document_symbols(file_path)

            elif command == "get_workspace_symbols":
                response_data = await self.get_workspace_symbols(query or "")

            elif command == "daemon_shutdown":
                await self.shutdown_and_exit()
                result = LSPResult(error="LSP daemon has been shutdown") # This message might not be sent
            
            else:
                result = LSPResult(error=f"Unrecognized or unsupported command: {command}")

            # 3. (New) Unified response parsing
            if response_data is not None:
                if 'error' in response_data:
                    # LSP server returned an error
                    result = LSPResult(error=response_data['error'])
                elif not response_data.get('result'):
                    # LSP server returned 'null' or '[]'
                    result = LSPResult(error=f"Command error: '{command}' was recognized but produced no response.")
                else:
                    # Valid result obtained, process it
                    try:
                        result = LSPResult(result=response_data.get('result'))
                    except Exception as e:
                        tb_str = traceback.format_exc()
                        safe_print(f"Daemon: Error parsing LSP response: {e}\n{tb_str}")
                        result = LSPResult(error=f"Daemon error: Failed to parse LSP response: {e}")
            
            elif result.status_code == 'success' and result.result is None:
                # Capture commands recognized but produced no response (logical error)
                result = LSPResult(error=f"Command error: '{command}' was recognized but produced no response.")

        except Exception as e:
            tb_str = traceback.format_exc()
            safe_print(f"Daemon: Unhandled exception in LSPClient.run: {e}\n{tb_str}")
            result = LSPResult(error=f"Unhandled exception: {e}")
    
        self._log_history(timestamp, request_args, result)
        return result


    async def initialize(self):
        init_params = {
            "processId": os.getpid(),
            "rootUri": self.project_root_uri,
            "workspaceFolders": [
                {
                    "uri": self.project_root_uri,
                    "name": Path(self.project_root_uri).name
                }
            ],
            "capabilities": {
                "textDocument": {
                    "synchronization": {"didSave": True, "change": 2},
                    "hover": {"dynamicRegistration": False},
                    "definition": {"dynamicRegistration": False},
                    "declaration": {"dynamicRegistration": False},
                    "typeDefinition": {"dynamicRegistration": False},
                    "implementation": {"dynamicRegistration": False},
                    "references": {"dynamicRegistration": False},
                    "documentHighlight": {"dynamicRegistration": False},
                    "signatureHelp": {"dynamicRegistration": False},
                    "callHierarchy": {"dynamicRegistration": False},
                    "documentSymbol": {"dynamicRegistration": False},
                },
                "workspace": {
                    "workspaceFolders": True,
                    "symbol": {"dynamicRegistration": False}
                }
            },
            "trace": "off"
        }
        # Check if the process is alive before sending the initialize request
        if self._lsp_process.returncode is not None:
            stderr_output = ""
            try:
                stderr_output = await asyncio.wait_for(
                    self._lsp_process.stderr.read(), 
                    timeout=10
                )
                stderr_output = stderr_output.decode('utf-8', errors='replace')
            except:
                pass
            
            raise RuntimeError(
                f"LSP server process died before initialization. "
                f"Exit code: {self._lsp_process.returncode}. "
                f"Stderr: {stderr_output or '(empty)'}"
            )
        init_response = await self.send_request("initialize", init_params, timeout=600)
        if not init_response or 'error' in init_response:
            raise RuntimeError(f"LSP server initialization failed: {init_response}")

        safe_print("Daemon: LSP server initialized successfully.")
        await self.send_notification("initialized", {})
        await self.send_notification(
            "workspace/didChangeConfiguration",
            {"settings": {"python": {"analysis": {"autoSearchPaths": True, "useLibraryCodeForTypes": True}}}}
        )

    async def open_document(self, file_path, language_id="python"):
        uri = Path(file_path).as_uri()
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            safe_print(f"Daemon: Error reading file {file_path}: {e}")
            return

        lock = await self._ensure_doc_lock(uri)
        async with lock:
            self._doc_versions[uri] = 1
            self._doc_texts[uri] = text

        await self.send_notification(
            "textDocument/didOpen",
            {
                "textDocument": {
                    "uri": uri, "languageId": language_id,
                    "version": self._doc_versions[uri], "text": text
                }
            }
        )
        safe_print(f"Daemon: didOpen -> {uri} (version={self._doc_versions[uri]})")

    async def change_document(self, file_path=None, uri=None, new_text=None, incremental_changes=None):
        if file_path and not uri:
            uri = Path(file_path).as_uri()
        if not uri:
            raise ValueError("Either file_path or uri must be provided")
        
        lock = await self._ensure_doc_lock(uri)
        async with lock:
            if incremental_changes is not None:
                version = await self._get_next_version(uri)
                params = {
                    "textDocument": {"uri": uri, "version": version},
                    "contentChanges": incremental_changes
                }
                if new_text is not None:
                    self._doc_texts[uri] = new_text
            else:
                if new_text is None:
                    try:
                        new_text_local = uri_to_path(uri).read_text(encoding='utf-8')
                    except Exception:
                        new_text_local = self._doc_texts.get(uri, "")
                else:
                    new_text_local = new_text
                version = await self._get_next_version(uri)
                self._doc_texts[uri] = new_text_local
                params = {
                    "textDocument": {"uri": uri, "version": version},
                    "contentChanges": [{"text": new_text_local}]
                }
        await self.send_notification("textDocument/didChange", params)
        safe_print(f"Daemon: didChange -> {uri} (version={version})")

    async def did_save(self, file_path=None, uri=None, include_text=False):
        if file_path and not uri:
            uri = Path(file_path).as_uri()
        if not uri:
            raise ValueError("Either file_path or uri must be provided")

        params = {"textDocument": {"uri": uri}}
        if include_text:
            lock = await self._ensure_doc_lock(uri)
            async with lock:
                text = self._doc_texts.get(uri, "")
            params["textDocument"]["text"] = text

        await self.send_notification("textDocument/didSave", params)
        safe_print(f"Daemon: didSave -> {uri}")

    async def did_close(self, file_path=None, uri=None):
        if file_path and not uri:
            uri = Path(file_path).as_uri()
        if not uri:
            raise ValueError("Either file_path or uri must be provided")

        params = {"textDocument": {"uri": uri}}
        await self.send_notification("textDocument/didClose", params)
        
        lock = await self._ensure_doc_lock(uri)
        async with lock:
            self._doc_versions.pop(uri, None)
            self._doc_texts.pop(uri, None)
        
        timer_task = self._debounce_timers.pop(uri, None)
        if timer_task:
            timer_task.cancel()
        safe_print(f"Daemon: didClose -> {uri}")

    async def did_create_files(self, files: List[str]):
        creates = [{"uri": Path(f).as_uri()} for f in files]
        safe_print(f"Daemon: Notifying server of {len(creates)} created files.")
        return await self.send_notification(
            "workspace/didCreateFiles",
            {"files": creates}
        )

    async def did_delete_files(self, files: List[str]):
        deletes = [{"uri": Path(f).as_uri()} for f in files]
        safe_print(f"Daemon: Notifying server of {len(deletes)} deleted files.")
        return await self.send_notification(
            "workspace/didDeleteFiles",
            {"files": deletes}
        )

    async def did_rename_files(self, old_path: str, new_path: str):
        renames = [{
            "oldUri": Path(old_path).as_uri(),
            "newUri": Path(new_path).as_uri()
        }]
        safe_print(f"Daemon: Notifying server of rename: {old_path} -> {new_path}")
        return await self.send_notification(
            "workspace/didRenameFiles",
            {"files": renames}
        )

    async def get_declaration(self, file_path, line, character):
        return await self.send_request(
            "textDocument/declaration",
            {"textDocument": {"uri": Path(file_path).as_uri()}, "position": {"line": line, "character": character}}
        )

    async def get_definition(self, file_path, line, character):
        return await self.send_request(
            "textDocument/definition",
            {"textDocument": {"uri": Path(file_path).as_uri()}, "position": {"line": line, "character": character}}
        )

    async def get_type_definition(self, file_path, line, character):
        return await self.send_request(
            "textDocument/typeDefinition",
            {"textDocument": {"uri": Path(file_path).as_uri()}, "position": {"line": line, "character": character}}
        )

    async def get_implementation(self, file_path, line, character):
        return await self.send_request(
            "textDocument/implementation",
            {"textDocument": {"uri": Path(file_path).as_uri()}, "position": {"line": line, "character": character}}
        )

    async def get_references(self, file_path, line, character, include_declaration=True):
        return await self.send_request(
            "textDocument/references",
            {
                "textDocument": {"uri": Path(file_path).as_uri()},
                "position": {"line": line, "character": character},
                "context": {"includeDeclaration": include_declaration}
            }
        )

    async def get_hover(self, file_path, line, character):
        return await self.send_request(
            "textDocument/hover",
            {"textDocument": {"uri": Path(file_path).as_uri()}, "position": {"line": line, "character": character}}
        )

    async def get_signature_help(self, file_path, line, character):
        return await self.send_request(
            "textDocument/signatureHelp",
            {"textDocument": {"uri": Path(file_path).as_uri()}, "position": {"line": line, "character": character}}
        )

    async def get_document_symbols(self, file_path):
        return await self.send_request(
            "textDocument/documentSymbol",
            {"textDocument": {"uri": Path(file_path).as_uri()}}
        )
    
    async def get_workspace_symbols(self, query=""):
        return await self.send_request("workspace/symbol", {"query": query})

    async def get_document_highlights(self, file_path, line, character):
        return await self.send_request(
            "textDocument/documentHighlight",
            {"textDocument": {"uri": Path(file_path).as_uri()}, "position": {"line": line, "character": character}}
        )

    async def prepare_call_hierarchy(self, file_path, line, character):
        return await self.send_request(
            "textDocument/prepareCallHierarchy",
            {"textDocument": {"uri": Path(file_path).as_uri()}, "position": {"line": line, "character": character}}
        )

    async def get_incoming_calls(self, item):
        return await self.send_request("callHierarchy/incomingCalls", {"item": item})

    async def get_outgoing_calls(self, item):
        return await self.send_request("callHierarchy/outgoingCalls", {"item": item})

    async def shutdown_and_exit(self):
        safe_print("Daemon: Shutting down LSP server...")
        self.save_history()
        
        # Cancel all background tasks first to prevent further operations
        for task in [self._reader_task, self._writer_task, self._stderr_task]:
            if task and not task.done():
                task.cancel()
        
        try:
            if self._lsp_process and self._lsp_process.returncode is None:
                try:
                    await self.send_request("shutdown", None, timeout=30)
                except Exception as e:
                    safe_print(f"Daemon: Error during shutdown request: {e}")
                
                try:
                    await self.send_notification("exit", None)
                except Exception as e:
                    safe_print(f"Daemon: Error during exit notification: {e}")
        except Exception as e:
            safe_print(f"Daemon: Error during graceful shutdown: {e}")
        
        # Terminate LSP process
        if self._lsp_process and self._lsp_process.returncode is None:
            try:
                self._lsp_process.terminate()
                await asyncio.wait_for(self._lsp_process.wait(), timeout=30)
                safe_print("Daemon: LSP process terminated.")
            except asyncio.TimeoutError:
                safe_print("Daemon: LSP process termination timed out, killing...")
                if self._lsp_process.returncode is None:
                    self._lsp_process.kill()
                    try:
                        await asyncio.wait_for(self._lsp_process.wait(), timeout=5)
                    except:
                        pass
                    safe_print("Daemon: LSP process killed.")
            except Exception as e:
                safe_print(f"Daemon: Error terminating LSP process: {e}")
                if self._lsp_process.returncode is None:
                    self._lsp_process.kill()
                    safe_print("Daemon: LSP process killed.")
        
        # Wait for tasks to complete (they should be cancelled already)
        for task in [self._reader_task, self._writer_task, self._stderr_task]:
            if task and not task.done():
                try:
                    await asyncio.wait_for(task, timeout=30)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
                except Exception as e:
                    safe_print(f"Daemon: Error waiting for task: {e}")
        
        # Stop write queue
        try:
            if self._write_queue:
                await self._write_queue.put(None)
        except Exception:
            pass
            
        safe_print("Daemon: LSP client closed.")

    def _log_history(self, timestamp: str, request_args: dict, result: LSPResult):
        history_entry = {
            "timestamp": timestamp,
            "request": request_args,
            "response": result.to_dict()
        }
        self._history.append(history_entry)

    def load_history(self) -> List[dict]:
        try:
            with open(LSP_HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return []
        except Exception as e:
            safe_print(f"Daemon: Warning: Could not load LSP history from {LSP_HISTORY_FILE}: {e}")
            return []

    def save_history(self):
        try:
            with open(LSP_HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(self._history, f, indent=2, default=str) # (Modified) Add default=str
            safe_print(f"Daemon: LSP history saved to {LSP_HISTORY_FILE}.")
        except Exception as e:
            safe_print(f"Daemon: Warning: Could not write LSP history to {LSP_HISTORY_FILE}: {e}")


async def handle_connection(
    client: LSPClient, 
    reader: asyncio.StreamReader, 
    writer: asyncio.StreamWriter
):
    """Handles a single client connection"""
    request_data = None
    response_result = None
    addr = writer.get_extra_info('peername')
    safe_print(f"Daemon: Connection from {addr}")

    try:
        len_data = await reader.readexactly(4)
        msg_len = struct.unpack('!I', len_data)[0]
        
        json_data = await reader.readexactly(msg_len)
        request_data = orjson.loads(json_data.decode('utf-8')) # (Modified) Use orjson
        safe_print(f"Daemon: Received command: {request_data.get('command')}")

        if request_data.get('command') == 'daemon_shutdown':
            safe_print("Daemon: Received shutdown command.")
            response_result = LSPResult(result={"summary": "Shutdown initiated"})
            # Trigger global shutdown event
            if shutdown_event:
                shutdown_event.set()
        else:
            response_result = await client.run(**request_data)

    except asyncio.IncompleteReadError:
        safe_print(f"Daemon: Client {addr} disconnected before sending full message.")
        return
    except Exception as e:
        safe_print(f"Daemon: Error handling connection: {e}\n{traceback.format_exc()}")
        response_result = LSPResult(error=f"Daemon error: {e}")
    
    try:
        if response_result is None:
            response_result = LSPResult(error="Daemon: No response was generated.")
        
        # response_json = response_result.to_text().encode('utf-8')
        response_json = response_result.to_json_str().encode('utf-8')
        writer.write(struct.pack('!I', len(response_json)))
        writer.write(response_json)
        await writer.drain()

    except Exception as e:
        safe_print(f"Daemon: Error sending response: {e}")
    finally:
        safe_print(f"Daemon: Closing connection to {addr}")
        writer.close()
        if sys.version_info >= (3, 7):
            try:
                await writer.wait_closed()
            except Exception:
                pass


def signal_handler(signum, frame):
    """Signal handler - Synchronous version"""
    safe_print(f"Daemon: Received signal {signum}")
    if shutdown_event and main_loop:
        # Set event in the event loop
        try:
            main_loop.call_soon_threadsafe(shutdown_event.set)
        except RuntimeError:
            pass


async def main_daemon(port_file_path: str):
    """Main daemon function"""
    global shutdown_event, main_loop
    shutdown_event = asyncio.Event()
    main_loop = asyncio.get_event_loop()
    
    # Set up signal handling
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    
    lsp_command = DEFAULT_LSP_COMMAND.split()
    project_root = DEFAULT_PROJECT_ROOT
    
    client = None
    server = None
    try:
        client = LSPClient(lsp_command, project_root)
        await client.start()
        
        async def handler_with_client(reader, writer):
            await handle_connection(client, reader, writer)
        
        server = await asyncio.start_server(handler_with_client, HOST, PORT)

        try:
            addr = server.sockets[0].getsockname()
            actual_port = addr[1]
            safe_print(f'Daemon: Serving on {addr[0]}:{actual_port} for {project_root}')
            
            # Write port to file
            with open(port_file_path, 'w') as f:
                f.write(str(actual_port))
                f.flush()
                os.fsync(f.fileno())
            
            safe_print(f'Daemon: Port {actual_port} written to {port_file_path}')
            safe_print("Daemon: Ready to accept connections")

        except Exception as e:
            safe_print(f"Daemon: Fatal error - cannot get or write port file: {e}")
            if server:
                server.close()
                await server.wait_closed()
            return

        # Start server and wait for shutdown signal
        try:
            # Python 3.6 compatibility: use server.wait_closed() and shutdown_event
            # Do not use serve_forever(); let the server run in background
            
            # Wait for shutdown event
            await shutdown_event.wait()
            
            safe_print("Daemon: Shutting down server...")
            
        except asyncio.CancelledError:
            safe_print("Daemon: Server task cancelled.")
        except Exception as e:
            safe_print(f"Daemon: Error in server loop: {e}")
        finally:
            safe_print("Daemon: Server loop exited.")
            
    except Exception as e:
        safe_print(f"Daemon: Fatal error in main_daemon: {e}\n{traceback.format_exc()}")
    finally:
        # Close server
        if server:
            try:
                server.close()
                await server.wait_closed()
                safe_print("Daemon: Server closed.")
            except Exception as e:
                safe_print(f"Daemon: Error closing server: {e}")
        
        # Close client
        if client:
            try:
                await client.close()
            except Exception as e:
                safe_print(f"Daemon: Error closing client: {e}")
        
        # Clean up port file
        try:
            if os.path.exists(port_file_path):
                os.remove(port_file_path)
                safe_print(f"Daemon: Cleaned up port file {port_file_path}")
        except OSError as e:
            safe_print(f"Daemon: Cannot clean up port file {port_file_path}: {e}")


if __name__ == "__main__":
    safe_print("Starting LSP Daemon...")

    port_file = PORT_FILE_ENV_VAR
    if not port_file:
        safe_print(f"Fatal error: PORT_FILE_ENV_VAR environment variable not set.")
        safe_print("Daemon cannot know where to write its port number.")
        sys.exit(1)

    # Daemonize (double fork)
    daemonize()

    try:
        run_asyncio(main_daemon(port_file))
    except KeyboardInterrupt:
        safe_print("Daemon: Interrupted by user (Ctrl+C)")
    except Exception as e:
        safe_print(f"Daemon: Unexpected error: {e}\n{traceback.format_exc()}")
    finally:
        safe_print("Daemon: Main daemon process has exited.")