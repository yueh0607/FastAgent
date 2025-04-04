from src.llm_proxy.tool import BaseTool
import re
from typing import Dict, Type, List, Generator, Any
import json

class FunctionCall:
    """函数调用管理器，用于处理工具调用和流式响应"""
    
    tools: Dict[str, BaseTool]

    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
        self.func_regex = re.compile(r"<function_call>(.*?)</function_call>")
        self.buffer = ""
        self.executed_calls = set()  # 记录已执行的函数调用

    def add_tool(self, tool: BaseTool) -> None:
        """添加工具到管理器"""
        self.tools[tool.name] = tool

    def get_system_prompt(self) -> str:
        """获取包含所有工具信息的系统提示"""
        if not self.tools:
            return ""
        
        prompts = ["You can use the following tools to help user:"]
        for tool in self.tools.values():
            prompts.append(f"{tool}")
        prompts.append(
            "You can use the following format to call tools:\n"
            "<function_call>tool_name(parameter_JSON)</function_call>\n"
            "The parameter_JSON must match the input pattern of the tool.\n"
            "You can insert these function calls in the middle of your response, you will get the result in the next response."
        )

        return "\n".join(prompts)
    
    def handle_stream(self, stream: Generator[str, None, None]) -> Generator[str, None, None]:
        """处理流式响应，自动执行并替换检测到的函数调用
        
        Args:
            stream: 原始响应流
            
        Yields:
            处理后的响应文本，包含函数调用结果和最终汇总
        """
        self.buffer = ""
        self.executed_calls = set()
        self.executed_tools = []  # 存储工具调用详情
        
        start_tag = '<function_call>'
        end_tag = '</function_call>'
        
        for chunk in stream:
            self.buffer += chunk
            
            while True:
                # 查找完整标签开始位置
                start_idx = self.buffer.find(start_tag)
                if start_idx != -1:
                    # 查找对应结束标签
                    end_idx = self.buffer.find(end_tag, start_idx + len(start_tag))
                    if end_idx != -1:
                        # 提取并执行函数调用
                        content = self.buffer[start_idx+len(start_tag):end_idx]
                        result_gen = self._execute_function_call(content)
                        
                        # 拼接并返回处理结果
                        if start_idx > 0:
                            yield self.buffer[:start_idx]  # 标签前内容
                        
                        # 处理函数调用结果
                        if isinstance(result_gen, Generator):
                            result_chunks = []
                            for chunk in result_gen:
                                result_chunks.append(chunk)
                                yield chunk
                            result = ''.join(result_chunks)
                        else:
                            result = result_gen
                            yield result
                        
                        # 重置buffer为剩余内容
                        self.buffer = self.buffer[end_idx+len(end_tag):]
                        continue  # 继续处理剩余内容
                    else:
                        # 只有开始标签，保留标签开始后的内容
                        if start_idx > 0:
                            yield self.buffer[:start_idx]
                            self.buffer = self.buffer[start_idx:]
                        break
                else:
                    # 检查是否有可能形成开始标签的部分匹配
                    max_prefix = 0
                    max_check = min(len(self.buffer), len(start_tag))
                    for k in range(1, max_check+1):
                        if self.buffer[-k:] == start_tag[:k]:
                            max_prefix = k
                    
                    if max_prefix > 0:
                        # 保留可能形成标签头的内容
                        output = self.buffer[:-max_prefix]
                        if output:
                            yield output
                        self.buffer = self.buffer[-max_prefix:]
                    else:
                        # 直接返回所有内容
                        if self.buffer:
                            yield self.buffer
                            self.buffer = ""
                    break

    def _parse_function_call(self, call_str: str) -> tuple[str, str]:
        """解析函数调用字符串"""
        paren_idx = call_str.find("(")
        if paren_idx == -1 or not call_str.endswith(")"):
            raise ValueError("无效的函数调用格式")
        
        tool_name = call_str[:paren_idx].strip()
        args_json = call_str[paren_idx+1:-1].strip()
        return tool_name, args_json
    
    def _execute_function_call(self, function_str: str) -> Generator[str, None, None] | str:
        """执行函数调用并返回结果
        
        Args:
            function_str: 函数调用字符串，格式为"tool_name(parameter_JSON)"
            
        Returns:
            如果是流式输出，返回生成器；否则返回字符串结果
        """
        try:
            # 解析工具名称和参数
            tool_name, params_str = self._parse_function_call(function_str)
            
            if tool_name not in self.tools:
                return f"[Function Call Error: Tool '{tool_name}' not found]"
            
            try:
                params = json.loads(params_str)
            except Exception as e:
                return f"[Function Call Error: Parameter parsing failed: {str(e)}]"
            
            # 执行工具调用
            tool = self.tools[tool_name]
            result = tool._run(**params)
            
            # 记录工具调用
            self.executed_tools.append({
                'tool': tool_name,
                'params': params_str,
                'result': str(result) if not isinstance(result, Generator) else "streaming..."
            })
            
            # 如果是生成器，直接返回
            if isinstance(result, Generator):
                return result
            
            # 如果是字符串，包装成结果格式
            return f"[Function Call: {function_str}, Result: {result}]"
        
        except Exception as e:
            return f"[Function Call Error: {str(e)}]"