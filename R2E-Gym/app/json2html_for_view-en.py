import json
import html
import re
from typing import Any, Dict, List, Union
import argparse
import os

class JSONToHTMLConverter:
    def __init__(self):
        self.global_counter = 0  # Global counter to ensure unique IDs
        
    def escape_html(self, text: str) -> str:
        """Escape HTML special characters"""
        return html.escape(str(text))
    
    def format_diff_patch(self, patch_str: str) -> str:
        """Format diff patch content"""
        if not isinstance(patch_str, str):
            patch_str = str(patch_str)
        lines = patch_str.split('\n')
        formatted_lines = []
        for line in lines:
            if line.startswith('+++') or line.startswith('---'):
                formatted_lines.append(f'<span class="diff-header">{self.escape_html(line)}</span>')
            elif line.startswith('+'):
                formatted_lines.append(f'<span class="diff-added">{self.escape_html(line)}</span>')
            elif line.startswith('-'):
                formatted_lines.append(f'<span class="diff-removed">{self.escape_html(line)}</span>')
            elif line.startswith('@@'):
                formatted_lines.append(f'<span class="diff-info">{self.escape_html(line)}</span>')
            else:
                formatted_lines.append(f'<span class="diff-context">{self.escape_html(line)}</span>')
        return '<div class="diff-content">' + '<br>'.join(formatted_lines) + '</div>'
    
    def format_python_code(self, code_str: str) -> str:
        """Format Python code (simplified version for display)"""
        if not isinstance(code_str, str):
            code_str = str(code_str)
        lines = code_str.split('\n')
        formatted_lines = []
        
        for i, line in enumerate(lines, 1):
            # Only escape HTML, no syntax highlighting
            escaped_line = self.escape_html(line)
            # Layout line number and code content
            formatted_lines.append(f'<div class="code-line"><span class="line-number">{i:3d}</span><span class="code-content">{escaped_line}</span></div>')
        
        return '<div class="python-content">' + ''.join(formatted_lines) + '</div>'
    
    def format_list(self, lst: List[Any]) -> str:
        """Format list content"""
        items = []
        for i, item in enumerate(lst):
            if isinstance(item, dict):
                items.append(f'<div class="list-item"><span class="list-index">[{i}]</span> {self.render_json_value(item, f"item_{i}")}</div>')
            elif isinstance(item, list):
                items.append(f'<div class="list-item"><span class="list-index">[{i}]</span> {self.format_list(item)}</div>')
            else:
                items.append(f'<div class="list-item"><span class="list-index">[{i}]</span> <span class="value-text">{self.escape_html(item)}</span></div>')
        return f'<div class="list-content">{"".join(items)}</div>'
    
    def render_json_value(self, value: Any, key_name: str = "", level: int = 0) -> str:
        """Recursively render JSON values"""
        self.global_counter += 1
        element_id = f"elem_{self.global_counter}"  # Use global counter to ensure unique IDs
        
        if isinstance(value, dict):
            items = []
            for k, v in value.items():
                child_html = self.render_json_value(v, k, level + 1)
                items.append(f'<div class="json-item">{child_html}</div>')
            return f'''
                <div class="json-dict collapsible" data-level="{level}">
                    <div class="dict-header" onclick="toggleCollapse('{element_id}')">
                        <span class="toggle-icon">▼</span>
                        <span class="key-name">{self.escape_html(key_name) if key_name else "object"}</span>
                        <span class="type-info">dict ({len(value)})</span>
                    </div>
                    <div class="dict-content" id="{element_id}">
                        {"".join(items)}
                    </div>
                </div>
            '''
        elif isinstance(value, list):
            return f'''
                <div class="json-list collapsible" data-level="{level}">
                    <div class="list-header" onclick="toggleCollapse('{element_id}')">
                        <span class="toggle-icon">▼</span>
                        <span class="key-name">{self.escape_html(key_name) if key_name else "array"}</span>
                        <span class="type-info">list ({len(value)})</span>
                    </div>
                    <div class="list-content" id="{element_id}">
                        {self.format_list(value)}
                    </div>
                </div>
            '''
        else:
            # Handle special formats
            value_str = str(value)
            content = self.escape_html(value_str)
            css_class = "value-text"
            
            if 'patch' in key_name.lower():
                content = self.format_diff_patch(value_str)
                css_class = "diff-container"
            elif '.py' in key_name.lower():
                content = self.format_python_code(value_str)
                css_class = "python-container"
            
            # Fix backslash issues in f-strings
            preview_text = self.escape_html(value_str.replace('\n', '\\n'))[:50]
            if len(value_str) > 50:
                preview_text += '...'
            
            # Every key-value pair can be collapsed
            return f'''
                <div class="json-value collapsible" data-level="{level}">
                    <div class="value-header" onclick="toggleCollapse('{element_id}')">
                        <span class="toggle-icon">▼</span>
                        <span class="key-name">{self.escape_html(key_name)}:</span>
                        <span class="value-preview">{preview_text}</span>
                    </div>
                    <div class="value-content" id="{element_id}">
                        <div class="{css_class}">{content}</div>
                    </div>
                </div>
            '''
    
    def generate_html(self, json_data: List[Dict], output_file: str):
        """Generate a complete HTML file"""
        total_items = len(json_data)
        
        # Generate data item HTML
        items_html = []
        for i, item in enumerate(json_data):
            # No longer reset counter to maintain global uniqueness
            html_content = self.render_json_value(item, f"item_{i}")
            
            items_html.append(f'''
                <div class="data-item" data-id="{i}" style="display: none;">
                    <div class="item-header">
                        <span class="item-id">ID: {i}</span>
                        <button class="collapse-all-btn" onclick="collapseAll()">Collapse All</button>
                        <button class="expand-all-btn" onclick="expandAll()">Expand All</button>
                    </div>
                    <div class="item-content">{html_content}</div>
                </div>
            ''')
        
        # Generate complete HTML
        html_content = f'''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SWE Data Viewer</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #f5f7fa 0%, #e9ecef 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #6c757d 0%, #495057 100%);
            color: white;
            padding: 20px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 28px;
            margin-bottom: 15px;
        }}
        
        .controls {{
            display: flex;
            justify-content: center;
            gap: 15px;
            flex-wrap: wrap;
        }}
        
        .control-group {{
            display: flex;
            align-items: center;
            gap: 10px;
            background: rgba(255,255,255,0.2);
            padding: 8px 15px;
            border-radius: 25px;
        }}
        
        .control-group label {{
            font-weight: 500;
        }}
        
        .control-group input, .control-group button {{
            padding: 5px 10px;
            border: none;
            border-radius: 5px;
            font-size: 14px;
        }}
        
        .control-group button {{
            background: white;
            color: #6c757d;
            cursor: pointer;
            transition: all 0.3s;
            font-weight: 600;
        }}
        
        .control-group button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }}
        
        .control-group button:disabled {{
            opacity: 0.5;
            cursor: not-allowed;
        }}
        
        .content {{
            height: calc(100vh - 200px);
            min-height: 600px;
            padding: 20px;
            overflow-y: auto;
        }}
        
        .data-item {{
            background: #fafbfc;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            animation: fadeIn 0.5s;
        }}
        
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        .item-header {{
            background: linear-gradient(135deg, #6c757d 0%, #495057 100%);
            color: white;
            padding: 10px 15px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-radius: 10px 10px 0 0;
        }}
        
        .item-id {{
            font-weight: 600;
            font-size: 16px;
        }}
        
        .collapse-all-btn, .expand-all-btn {{
            background: rgba(255,255,255,0.2);
            color: white;
            border: 1px solid rgba(255,255,255,0.3);
            padding: 5px 10px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 12px;
            transition: all 0.3s;
        }}
        
        .collapse-all-btn:hover, .expand-all-btn:hover {{
            background: rgba(255,255,255,0.3);
        }}
        
        .item-content {{
            padding: 15px;
        }}
        
        .json-dict, .json-list, .json-value {{
            margin-bottom: 8px;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            overflow: hidden;
        }}
        
        .dict-header, .list-header, .value-header {{
            background: #f8f9fa;
            padding: 10px 15px;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 10px;
            transition: background 0.3s;
            user-select: none;
        }}
        
        .dict-header:hover, .list-header:hover, .value-header:hover {{
            background: #e9ecef;
        }}
        
        .toggle-icon {{
            transition: transform 0.3s;
            font-size: 12px;
            color: #6c757d;
        }}
        
        .collapsed .toggle-icon {{
            transform: rotate(-90deg);
        }}
        
        .key-name {{
            font-weight: 600;
            color: #495057;
        }}
        
        .value-preview {{
            color: #6c757d;
            font-size: 12px;
            margin-left: auto;
            font-style: italic;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            max-width: 200px;
        }}
        
        .type-info {{
            color: #6c757d;
            font-size: 12px;
            margin-left: auto;
        }}
        
        .dict-content, .list-content, .value-content {{
            padding: 10px 15px;
            background: white;
            border-top: 1px solid #e9ecef;
        }}
        
        .json-value {{
            margin-left: 20px;
        }}
        
        .value-text {{
            color: #495057;
            font-family: 'Courier New', monospace;
            background: #f8f9fa;
            padding: 10px 12px;
            border-radius: 5px;
            display: block;
            width: 100%;
            white-space: pre-wrap;
            word-wrap: break-word;
            line-height: 1.5;
            border: 1px solid #e9ecef;
        }}
        
        .list-item {{
            padding: 5px 0;
            margin-left: 20px;
            border-left: 2px solid #e9ecef;
            padding-left: 10px;
        }}
        
        .list-index {{
            color: #6c757d;
            font-weight: 600;
            margin-right: 5px;
        }}
        
        .diff-container {{
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 5px;
            font-family: 'Courier New', monospace;
            font-size: 13px;
            margin-top: 5px;
        }}
        
        .diff-content {{
            padding: 10px;
            overflow-x: auto;
        }}
        
        .diff-header {{
            color: #6c757d;
            font-weight: 600;
        }}
        
        .diff-added {{
            color: #28a745;
            background: #d4edda;
        }}
        
        .diff-removed {{
            color: #dc3545;
            background: #f8d7da;
        }}
        
        .diff-info {{
            color: #6c757d;
            background: #e9ecef;
        }}
        
        .diff-context {{
            color: #495057;
        }}
        
        .python-container {{
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 5px;
            font-family: 'Courier New', monospace;
            font-size: 13px;
            margin-top: 5px;
            max-width: 100%;
            overflow: hidden;
        }}
        
        .python-content {{
            padding: 10px;
            overflow-x: auto;
            max-width: 100%;
        }}
        
        .code-line {{
            display: flex;
            align-items: flex-start;
            margin-bottom: 2px;
            width: 100%;
        }}
        
        .line-number {{
            color: #adb5bd;
            margin-right: 10px;
            user-select: none;
            flex-shrink: 0;
            width: 3em;
            text-align: right;
            font-family: 'Courier New', monospace;
        }}
        
        .code-content {{
            flex: 1;
            white-space: pre-wrap;
            word-wrap: break-word;
            word-break: break-all;
            overflow-wrap: break-word;
            font-family: 'Courier New', monospace;
            line-height: 1.4;
        }}
        
        .status-bar {{
            background: #f8f9fa;
            padding: 10px 20px;
            text-align: center;
            color: #6c757d;
            border-top: 1px solid #e9ecef;
        }}
        
        .loading {{
            text-align: center;
            padding: 20px;
            color: #6c757d;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>SWE Data Viewer</h1>
            <div class="controls">
                <div class="control-group">
                    <button onclick="previousItem()" id="prevBtn">Previous</button>
                    <span id="currentInfo">0 / {total_items}</span>
                    <button onclick="nextItem()" id="nextBtn">Next</button>
                </div>
                <div class="control-group">
                    <label>Jump to ID:</label>
                    <input type="number" id="jumpInput" min="0" max="{total_items-1}" style="width: 80px;">
                    <button onclick="jumpToItem()">Jump</button>
                </div>
                <div class="control-group">
                    <label>Auto Play:</label>
                    <button onclick="toggleAutoPlay()" id="autoPlayBtn">Start</button>
                    <input type="number" id="intervalInput" value="2" min="0.5" max="10" step="0.5" style="width: 60px;">
                    <span>s</span>
                </div>
            </div>
        </div>
        
        <div class="content">
            <div id="content-area">
                {"".join(items_html)}
            </div>
        </div>
        
        <div class="status-bar">
            <span id="statusText">Ready</span>
        </div>
    </div>
    
    <script>
        let currentIndex = 0;
        const totalItems = {total_items};
        let autoPlayInterval = null;
        let isUpdating = false;
        
        // Use requestAnimationFrame for performance optimization
        function requestUpdateDisplay() {{
            if (isUpdating) return;
            isUpdating = true;
            
            requestAnimationFrame(() => {{
                try {{
                    updateDisplayInternal();
                }} catch (error) {{
                    console.error('Error updating display:', error);
                    document.getElementById('statusText').textContent = 'Error updating display: ' + error.message;
                }} finally {{
                    isUpdating = false;
                }}
            }});
        }}
        
        function updateDisplayInternal() {{
            // Hide all items
            const allItems = document.querySelectorAll('.data-item');
            allItems.forEach(item => {{
                item.style.display = 'none';
            }});
            
            // Validate index
            if (currentIndex < 0) currentIndex = 0;
            if (currentIndex >= totalItems) currentIndex = totalItems - 1;
            
            // Show current item
            const items = document.querySelectorAll('#content-area .data-item');
            
            if (currentIndex >= 0 && currentIndex < totalItems && items[currentIndex]) {{
                items[currentIndex].style.display = 'block';
                
                // Update information
                document.getElementById('currentInfo').textContent = `${{currentIndex + 1}} / ${{totalItems}}`;
                document.getElementById('statusText').textContent = `Showing ID: ${{currentIndex}}`;
                
                // Update button states
                document.getElementById('prevBtn').disabled = currentIndex === 0;
                document.getElementById('nextBtn').disabled = currentIndex === totalItems - 1;
            }} else {{
                document.getElementById('statusText').textContent = `Cannot display item, index out of range`;
            }}
        }}
        
        function updateDisplay() {{
            requestUpdateDisplay();
        }}
        
        function previousItem() {{
            if (currentIndex > 0 && !isUpdating) {{
                currentIndex--;
                updateDisplay();
            }}
        }}
        
        function nextItem() {{
            if (currentIndex < totalItems - 1 && !isUpdating) {{
                currentIndex++;
                updateDisplay();
            }}
        }}
        
        function jumpToItem() {{
            if (isUpdating) return;
            
            const input = document.getElementById('jumpInput');
            const targetId = parseInt(input.value);
            
            if (isNaN(targetId)) {{
                document.getElementById('statusText').textContent = 'Please enter a valid number';
                setTimeout(() => {{
                    updateDisplay();
                }}, 2000);
                return;
            }}
            
            if (targetId >= 0 && targetId < totalItems) {{
                currentIndex = targetId;
                updateDisplay();
            }} else {{
                document.getElementById('statusText').textContent = `Invalid ID: ${{targetId}} (Range: 0-${{totalItems-1}})`;
                setTimeout(() => {{
                    updateDisplay();
                }}, 2000);
            }}
        }}
        
        function toggleCollapse(elementId) {{
            const element = document.getElementById(elementId);
            if (!element) {{
                console.error('Element not found:', elementId);
                return;
            }}
            
            const parent = element.parentElement;
            if (!parent) return;
            
            if (parent.classList.contains('collapsed')) {{
                parent.classList.remove('collapsed');
                element.style.display = 'block';
            }} else {{
                parent.classList.add('collapsed');
                element.style.display = 'none';
            }}
        }}
        
        function collapseAll() {{
            document.querySelectorAll('.collapsible').forEach(item => {{
                if (!item.classList.contains('collapsed')) {{
                    const content = item.querySelector('.dict-content, .list-content, .value-content');
                    if (content) {{
                        item.classList.add('collapsed');
                        content.style.display = 'none';
                    }}
                }}
            }});
        }}
        
        function expandAll() {{
            document.querySelectorAll('.collapsible').forEach(item => {{
                if (item.classList.contains('collapsed')) {{
                    const content = item.querySelector('.dict-content, .list-content, .value-content');
                    if (content) {{
                        item.classList.remove('collapsed');
                        content.style.display = 'block';
                    }}
                }}
            }});
        }}
        
        function toggleAutoPlay() {{
            if (isUpdating) return;
            
            const btn = document.getElementById('autoPlayBtn');
            const intervalInput = document.getElementById('intervalInput');
            
            if (autoPlayInterval) {{
                clearInterval(autoPlayInterval);
                autoPlayInterval = null;
                btn.textContent = 'Start';
                document.getElementById('statusText').textContent = 'Auto play stopped';
            }} else {{
                const interval = parseFloat(intervalInput.value) * 1000;
                if (isNaN(interval) || interval < 500) {{
                    document.getElementById('statusText').textContent = 'Please enter a valid interval';
                    return;
                }}
                
                autoPlayInterval = setInterval(() => {{
                    if (currentIndex < totalItems - 1) {{
                        nextItem();
                    }} else {{
                        currentIndex = 0;
                        updateDisplay();
                    }}
                }}, interval);
                btn.textContent = 'Stop';
                document.getElementById('statusText').textContent = 'Auto playing...';
            }}
        }}
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {{
            if (isUpdating) return;
            
            switch(e.key) {{
                case 'ArrowLeft':
                    e.preventDefault();
                    previousItem();
                    break;
                case 'ArrowRight':
                    e.preventDefault();
                    nextItem();
                    break;
                case ' ':
                    e.preventDefault();
                    toggleAutoPlay();
                    break;
            }}
        }});
        
        // Jump on Enter key
        document.getElementById('jumpInput').addEventListener('keypress', (e) => {{
            if (e.key === 'Enter') {{
                jumpToItem();
            }}
        }});
        
        // Initialize display
        document.addEventListener('DOMContentLoaded', () => {{
            updateDisplay();
        }});
    </script>
</body>
</html>
        '''
        
        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"HTML file generated: {output_file}")
        print(f"Processed {len(json_data)} items total")


def read_jsonl(json_file):
    data = []
    with open(json_file, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            data.append(obj)
    print(f"Processed {len(data)} items from {json_file}")
    return data

            
def main():
    parser = argparse.ArgumentParser(description='Convert JSON files to HTML viewer')
    parser.add_argument('json_file', help='Path to the JSON file')
    parser.add_argument('-o', '--output', default='output.html', help='Path to the output HTML file')
    
    args = parser.parse_args()
    
    # Read JSON file
    if args.json_file.endswith('.jsonl'):
        json_data = read_jsonl(args.json_file)
    else:
        try:
            with open(args.json_file, 'r', encoding='utf-8') as f:
                json_data = json.load(f)[20:25]  # Taking items 20-25 as an example
        except Exception as e:
            print(f"Failed to read JSON file: {e}")
            return
    
    # Ensure data is in list format
    if not isinstance(json_data, list):
        json_data = [json_data]
    
    # Convert to HTML
    converter = JSONToHTMLConverter()
    converter.generate_html(json_data, args.output)
    print(f"HTML file generated: {args.output}")

if __name__ == '__main__':
    main()