import pytest
import threading
from ligprotgnnx.preprocessing.parser_registry import ParserRegistry, BaseProteinParser
from ligprotgnnx.preprocessing.protein_structure import ProteinStructure
from ligprotgnnx.preprocessing.exceptions import ParserRegistrationError

class DummyPDBParser(BaseProteinParser):
    def parse(self, path: str) -> ProteinStructure:
        raise NotImplementedError()

def test_parser_registry_registration():
    # Setup - clear any previous registration
    ParserRegistry.unregister_parser("dummy_pdb")
    
    # 1. Register successfully
    ParserRegistry.register_parser("dummy_pdb", DummyPDBParser)
    assert ParserRegistry.get_parser("dummy_pdb") == DummyPDBParser
    
    # 2. Register duplicate should raise error
    with pytest.raises(ParserRegistrationError):
        ParserRegistry.register_parser("dummy_pdb", DummyPDBParser)
        
    # 3. Unregister successfully
    ParserRegistry.unregister_parser("dummy_pdb")
    with pytest.raises(ParserRegistrationError):
        ParserRegistry.get_parser("dummy_pdb")

def test_parser_registry_thread_safety():
    # Registering unique formats concurrently should not corrupt registry state
    def worker(i: int):
        fmt = f"format_{i}"
        ParserRegistry.register_parser(fmt, DummyPDBParser)
        
    threads = []
    for idx in range(10):
        t = threading.Thread(target=worker, args=(idx,))
        threads.append(t)
        t.start()
        
    for t in threads:
        t.join()
        
    for idx in range(10):
        fmt = f"format_{idx}"
        assert ParserRegistry.get_parser(fmt) == DummyPDBParser
        ParserRegistry.unregister_parser(fmt)
