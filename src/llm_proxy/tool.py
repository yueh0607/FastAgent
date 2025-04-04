from typing import Dict, Any, Generator, Union
from abc import ABC, abstractmethod
import json
from pydantic import BaseModel

class BaseTool:
    name: str
    description: str
    argSchema: type[BaseModel]

    def _parse_args(self, args: str):
        return self.argSchema.model_validate_json(args)

    @abstractmethod
    def _run(self, **kwargs) -> Union[Any, Generator[str, None, None]]:
        """运行工具
        
        Returns:
            Union[Any, Generator[str, None, None]]: 可以返回单个结果或生成器
        """
        pass
    
    def __str__(self):
        return json.dumps({
            "name": self.name,
            "description": self.description,
            "argSchema": self.argSchema.model_json_schema()
        })

