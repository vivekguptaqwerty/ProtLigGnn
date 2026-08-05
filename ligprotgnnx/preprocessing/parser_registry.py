import threading
from abc import ABC, abstractmethod
from typing import Dict, Type
from ligprotgnnx.preprocessing.protein_structure import ProteinStructure
from ligprotgnnx.preprocessing.exceptions import ParserRegistrationError

class BaseProteinParser(ABC):
    @abstractmethod
    def parse(self, path: str) -> ProteinStructure:
        pass

class ParserRegistry:
    _lock = threading.Lock()
    _registry: Dict[str, Type[BaseProteinParser]] = {}

    @classmethod
    def register_parser(cls, format_name: str, parser_class: Type[BaseProteinParser]) -> None:
        with cls._lock:
            name = format_name.lower().strip()
            if not name:
                raise ParserRegistrationError(name, "Format name cannot be empty")
            if name in cls._registry:
                raise ParserRegistrationError(name, f"Parser for format '{name}' is already registered")
            cls._registry[name] = parser_class

    @classmethod
    def unregister_parser(cls, format_name: str) -> None:
        with cls._lock:
            name = format_name.lower().strip()
            if name in cls._registry:
                del cls._registry[name]

    @classmethod
    def get_parser(cls, format_name: str) -> Type[BaseProteinParser]:
        with cls._lock:
            name = format_name.lower().strip()
            if name not in cls._registry:
                raise ParserRegistrationError(name, f"No parser registered for format '{name}'")
            return cls._registry[name]
