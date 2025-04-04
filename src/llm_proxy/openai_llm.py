from .llm_base import LLMBase
import requests
import json
from typing import Generator, Dict, Any,List

class OpenAILLM(LLMBase):
    def __init__(self,base_url:str,api_key:str):
        super().__init__(base_url,api_key)
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def _chat_raw(
        self,
        messages: List[Dict[str,str]],
        model: str,
        temperature: float = 0.7,
        stream: bool = False,
        **kwargs
    ) -> Generator[str, None, None] | str:
        url = f"{self.base_url}/chat/completions"
        data = {
            "messages": messages,
            "model": model,
            "temperature": temperature,
            "stream": stream,
            **kwargs
        }

        response = requests.post(
            url,
            headers=self.headers,
            json=data,
            stream=stream
        )

        if response.status_code != 200:
            raise Exception(f"API请求失败，状态码：{response.status_code}，错误：{response.text}")

        if stream:
            return self._handle_stream_response(response)
        else:
            return response.json()['choices'][0]['message']['content']
    

    def _handle_stream_response(self, response: requests.Response) -> Generator[str, None, None]:
        """
        处理流式响应，生成连续的数据块
        """
        buffer = ""
        for chunk in response.iter_lines():
            if chunk:
                # 处理数据分片
                chunk_str = chunk.decode('utf-8')
                if chunk_str.startswith("data: "):
                    data = chunk_str[6:]  # 去掉"data: "前缀
                    
                    # 处理结束标志
                    if data.strip() == "[DONE]":
                        break
                    
                    try:
                        # 解析JSON数据
                        parsed = json.loads(data)
                        if "choices" in parsed:
                            delta = parsed["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            yield content
                    except json.JSONDecodeError:
                        print(f"JSON解析错误: {data}")
                        continue

class DeepSeekLLM(OpenAILLM):
    def __init__(self,api_key:str):
        super().__init__(base_url="https://api.deepseek.com/v1",api_key=api_key)
