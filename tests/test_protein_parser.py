import pytest
import tempfile
import os
from ligprotgnnx.preprocessing.protein_parser import ProteinParser
from ligprotgnnx.preprocessing.parser_registry import ParserRegistry, BaseProteinParser
from ligprotgnnx.preprocessing.protein_structure import ProteinStructure, ProteinResidue, ProteinAtom

class MockPDBParser(BaseProteinParser):
    def parse(self, path: str) -> ProteinStructure:
        atom = ProteinAtom(coordinates=[0.0, 0.0, 0.0], atom_name="CA", element="C", atom_serial=1)
        residue = ProteinResidue(residue_name="ALA", residue_id="1", chain_id="A", atoms=[atom])
        return ProteinStructure(
            structure_id="mock_pdb",
            experimental_method="X-RAY",
            resolution=1.8,
            residues=[residue],
            source_file=path,
            parser_version="1.0"
        )

class MockValidator:
    def __init__(self):
        self.validated = False

    def validate(self, structure: ProteinStructure):
        self.validated = True

def test_protein_parser_dispatcher():
    # Register mock parser
    ParserRegistry.unregister_parser("pdb")
    ParserRegistry.register_parser("pdb", MockPDBParser)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a valid pdb-like file
        pdb_path = os.path.join(tmpdir, "test.pdb")
        with open(pdb_path, "w") as f:
            f.write("HEADER    TEST COMPLEX                      18-JUL-26\nATOM      1  CA  ALA A   1       0.000   0.000   0.000\n")
            
        validator = MockValidator()
        struct = ProteinParser.load(pdb_path, validator=validator)
        
        assert struct.structure_id == "mock_pdb"
        assert struct.num_atoms == 1
        assert validator.validated is True
        
    ParserRegistry.unregister_parser("pdb")
