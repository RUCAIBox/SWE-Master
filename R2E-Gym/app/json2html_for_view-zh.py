import json
import html
import re
from typing import Any, Dict, List, Union
import argparse
import os

class JSONToHTMLConverter:
    def __init__(self):
        self.global_counter = 0  # 全局计数器，确保ID唯一
        
    def escape_html(self, text: str) -> str:
        """转义HTML特殊字符"""
        return html.escape(str(text))
    
    def format_diff_patch(self, patch_str: str) -> str:
        """格式化diff patch内容"""
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
        """格式化Python代码（简化版，只做基本显示）"""
        if not isinstance(code_str, str):
            code_str = str(code_str)
        lines = code_str.split('\n')
        formatted_lines = []
        
        for i, line in enumerate(lines, 1):
            # 只转义HTML，不做任何高亮处理
            escaped_line = self.escape_html(line)
            # 将行号和代码内容放在同一行，但使用更好的布局
            formatted_lines.append(f'<div class="code-line"><span class="line-number">{i:3d}</span><span class="code-content">{escaped_line}</span></div>')
        
        return '<div class="python-content">' + ''.join(formatted_lines) + '</div>'
    
    def format_list(self, lst: List[Any]) -> str:
        """格式化列表内容"""
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
        """递归渲染JSON值"""
        self.global_counter += 1
        element_id = f"elem_{self.global_counter}"  # 使用全局计数器确保ID唯一
        
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
            # 处理特殊格式
            value_str = str(value)
            content = self.escape_html(value_str)
            css_class = "value-text"
            
            if 'patch' in key_name.lower():
                content = self.format_diff_patch(value_str)
                css_class = "diff-container"
            elif '.py' in key_name.lower():
                content = self.format_python_code(value_str)
                css_class = "python-container"
            
            # 修复f-string中的反斜杠问题
            preview_text = self.escape_html(value_str.replace('\n', '\\n'))[:50]
            if len(value_str) > 50:
                preview_text += '...'
            
            # 每个key-value对都可以折叠
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
        """生成完整的HTML文件"""
        total_items = len(json_data)
        
        # 生成数据项HTML
        items_html = []
        for i, item in enumerate(json_data):
            # 不再重置计数器，保持全局唯一性
            html_content = self.render_json_value(item, f"item_{i}")
            
            items_html.append(f'''
                <div class="data-item" data-id="{i}" style="display: none;">
                    <div class="item-header">
                        <span class="item-id">ID: {i}</span>
                        <button class="collapse-all-btn" onclick="collapseAll()">全部折叠</button>
                        <button class="expand-all-btn" onclick="expandAll()">全部展开</button>
                    </div>
                    <div class="item-content">{html_content}</div>
                </div>
            ''')
        
        # 生成完整HTML
        html_content = f'''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SWE轨迹查看器</title>
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
            <h1>SWE轨迹查看器</h1>
            <div class="controls">
                <div class="control-group">
                    <button onclick="previousItem()" id="prevBtn">上一条</button>
                    <span id="currentInfo">0 / {total_items}</span>
                    <button onclick="nextItem()" id="nextBtn">下一条</button>
                </div>
                <div class="control-group">
                    <label>跳转到ID:</label>
                    <input type="number" id="jumpInput" min="0" max="{total_items-1}" style="width: 80px;">
                    <button onclick="jumpToItem()">跳转</button>
                </div>
                <div class="control-group">
                    <label>自动播放:</label>
                    <button onclick="toggleAutoPlay()" id="autoPlayBtn">开始</button>
                    <input type="number" id="intervalInput" value="2" min="0.5" max="10" step="0.5" style="width: 60px;">
                    <span>秒</span>
                </div>
            </div>
        </div>
        
        <div class="content">
            <div id="content-area">
                {"".join(items_html)}
            </div>
        </div>
        
        <div class="status-bar">
            <span id="statusText">就绪</span>
        </div>
    </div>
    
    <script>
        let currentIndex = 0;
        const totalItems = {total_items};
        let autoPlayInterval = null;
        let isUpdating = false;
        
        // 使用requestAnimationFrame优化性能
        function requestUpdateDisplay() {{
            if (isUpdating) return;
            isUpdating = true;
            
            requestAnimationFrame(() => {{
                try {{
                    updateDisplayInternal();
                }} catch (error) {{
                    console.error('更新显示时出错:', error);
                    document.getElementById('statusText').textContent = '更新显示时出错: ' + error.message;
                }} finally {{
                    isUpdating = false;
                }}
            }});
        }}
        
        function updateDisplayInternal() {{
            // 隐藏所有项目
            const allItems = document.querySelectorAll('.data-item');
            allItems.forEach(item => {{
                item.style.display = 'none';
            }});
            
            // 验证索引
            if (currentIndex < 0) currentIndex = 0;
            if (currentIndex >= totalItems) currentIndex = totalItems - 1;
            
            // 显示当前项目
            const items = document.querySelectorAll('#content-area .data-item');
            
            if (currentIndex >= 0 && currentIndex < totalItems && items[currentIndex]) {{
                items[currentIndex].style.display = 'block';
                
                // 更新信息
                document.getElementById('currentInfo').textContent = `${{currentIndex + 1}} / ${{totalItems}}`;
                document.getElementById('statusText').textContent = `正在显示 ID: ${{currentIndex}}`;
                
                // 更新按钮状态
                document.getElementById('prevBtn').disabled = currentIndex === 0;
                document.getElementById('nextBtn').disabled = currentIndex === totalItems - 1;
            }} else {{
                document.getElementById('statusText').textContent = `无法显示项目，索引超出范围`;
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
                document.getElementById('statusText').textContent = '请输入有效的数字';
                setTimeout(() => {{
                    updateDisplay();
                }}, 2000);
                return;
            }}
            
            if (targetId >= 0 && targetId < totalItems) {{
                currentIndex = targetId;
                updateDisplay();
            }} else {{
                document.getElementById('statusText').textContent = `无效的ID: ${{targetId}} (范围: 0-${{totalItems-1}})`;
                setTimeout(() => {{
                    updateDisplay();
                }}, 2000);
            }}
        }}
        
        function toggleCollapse(elementId) {{
            const element = document.getElementById(elementId);
            if (!element) {{
                console.error('找不到元素:', elementId);
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
                btn.textContent = '开始';
                document.getElementById('statusText').textContent = '自动播放已停止';
            }} else {{
                const interval = parseFloat(intervalInput.value) * 1000;
                if (isNaN(interval) || interval < 500) {{
                    document.getElementById('statusText').textContent = '请输入有效的时间间隔';
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
                btn.textContent = '停止';
                document.getElementById('statusText').textContent = '自动播放中...';
            }}
        }}
        
        // 键盘快捷键
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
        
        // 输入框回车跳转
        document.getElementById('jumpInput').addEventListener('keypress', (e) => {{
            if (e.key === 'Enter') {{
                jumpToItem();
            }}
        }});
        
        // 初始化显示
        document.addEventListener('DOMContentLoaded', () => {{
            updateDisplay();
        }});
    </script>
</body>
</html>
        '''
        
        # 写入文件
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"HTML文件已生成: {output_file}")
        print(f"共处理 {len(json_data)} 条数据")


def read_jsonl(json_file):
    data = []
    with open(json_file, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            data.append(obj)
    print(f"共处理 {len(data)} 条数据, 从 {json_file} 读取")
    return data

            
def main():
    parser = argparse.ArgumentParser(description='将JSON文件转换为HTML查看器')
    parser.add_argument('json_file', help='JSON文件路径')
    parser.add_argument('-o', '--output', default='output.html', help='输出HTML文件路径')
    
    args = parser.parse_args()
    
    # 读取JSON文件
    if args.json_file.endswith('.jsonl'):
        json_data = read_jsonl(args.json_file)
    else:
        try:
            with open(args.json_file, 'r', encoding='utf-8') as f:
                json_data = json.load(f)[20:25]  # 仍然只取20-25条数据作为示例
        except Exception as e:
            print(f"读取JSON文件失败: {e}")
            return
    
    # 确保数据是列表格式
    if not isinstance(json_data, list):
        json_data = [json_data]
    json_data = json_data
    # 转换为HTML
    converter = JSONToHTMLConverter()
    converter.generate_html(json_data, args.output)
    print(f"已生成HTML文件: {args.output}")
    
    # 自动打开浏览器
    # import webbrowser
    # import time
    # abs_path = os.path.abspath(args.output)
    # webbrowser.open(f'file://{abs_path}')

if __name__ == '__main__':
    main()