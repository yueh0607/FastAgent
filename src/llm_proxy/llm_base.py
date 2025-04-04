from typing import Generator, Dict, Any,List,Union
import json
from abc import ABC, abstractmethod

class LLMMessage:
    role:str
    content:str

    def __init__(self,role:str,content:str):
        self.role = role
        self.content = content

    def to_dict(self):
        return {'role': self.role, 'content':self.content}

    def from_json(json_str:str):
        return LLMMessage(**json.loads(json_str))
    
    #自动转str
    def __str__(self):
        return str(self.to_dict())

class LLMBase(ABC):

    base_url:str
    api_key:str

    session:List[LLMMessage]

    def __init__(self,base_url:str,api_key:str):
        self.base_url = base_url
        self.api_key = api_key
        self.session = []

    def dump_session(self):
        return [msg.to_dict() for msg in self.session]

    def __process_messages(self,msgs:Union[str,Dict[str,str],List[Dict[str,str]],LLMMessage,List[LLMMessage]]):
        # 统一转换为 List[LLMMessage] 格式
        if isinstance(msgs, str):
            msgs = [LLMMessage(role='user', content=msgs)]
        elif isinstance(msgs, LLMMessage):
            msgs = [msgs]
        elif isinstance(msgs, dict):
            # 处理单个字典
            if 'role' not in msgs or 'content' not in msgs:
                raise ValueError("Dict must contain 'role' and 'content' fields")
            msgs = [LLMMessage(role=msgs['role'], content=msgs['content'])]
        elif isinstance(msgs, list):
            # 处理列表
            if all(isinstance(msg, LLMMessage) for msg in msgs):
                # 已经是 LLMMessage 列表，保持不变
                pass
            elif all(isinstance(msg, dict) for msg in msgs):
                # 转换字典列表为 LLMMessage 列表
                msgs = [LLMMessage(role=msg['role'], content=msg['content']) for msg in msgs]
            else:
                raise ValueError("List items must be either all dicts or all LLMMessage objects")
        else:
            raise ValueError(f"Invalid message type: {type(msgs)}")
        return msgs

    def chat(self, msgs: Union[str, Dict[str,str], List[Dict[str,str]], LLMMessage, List[LLMMessage]], 
             model: str, temperature: float, **kwargs) -> Generator[str, None, None]:
        messages = self.__process_messages(msgs)

        # 验证消息格式
        for msg in messages:
            if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                raise ValueError("Messages must contain 'role' and 'content' fields")
            if msg["role"] not in ["system", "user", "assistant"]:
                raise ValueError(f"Invalid role: {msg['role']}")

        # 调用底层的 chat_raw 方法
        response = self._chat_raw(messages, model, temperature, **kwargs)
        return response

    @abstractmethod
    def _chat_raw(self, messages: List[Dict[str, str]], model: str, temperature: float, **kwargs) -> Generator[str, None, None]:
        """底层的 chat 实现，子类必须实现此方法"""
        pass

    def chat_with_context(self, msgs: Union[str, LLMMessage, List[LLMMessage], Dict[str,str], List[Dict[str,str]]], 
                         model: str, temperature: float, **kwargs) -> Generator[str, None, None]:
        
        msgs :List[LLMMessage] = self.__process_messages(msgs)
        
        # 先添加消息到会话历史
        self.session.extend(msgs)

        # 如果消息不包含 user 消息，直接返回 None
        if not any(msg.role == 'user' for msg in msgs):
            return None

        messages = [msg.to_dict() for msg in self.session]
        response = self._chat_raw(messages, model, temperature, **kwargs)
        
        # 流式模式：返回生成器
        def stream_generator():
            stream_str_list = []
            for chunk in response:
                stream_str_list.append(chunk)
                yield chunk
            # 在生成器结束时添加消息到会话
            n_msg = LLMMessage(role='assistant', content=''.join(stream_str_list))
            self.session.append(n_msg)
        
        return stream_generator()

    def clear_session(self):
        self.session = []

