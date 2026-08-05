import pytest
from pydantic import ValidationError
from ligprotgnnx.preprocessing.protein_structure import ProteinStructure, ProteinResidue, ProteinAtom

def test_protein_structure_properties():
    atom = ProteinAtom(
        coordinates=[0.0, 1.0, 2.0],
        atom_name="CA",
        element="C",
        atom_serial=1,
        occupancy=1.0,
        b_factor=10.0,
        altloc=" ",
        formal_charge=0.0
    )
    
    residue = ProteinResidue(
        residue_name="ALA",
        residue_id="1",
        insertion_code=" ",
        chain_id="A",
        is_hetero=False,
        is_water=False,
        atoms=[atom]
    )
    
    struct = ProteinStructure(
        structure_id="1abc",
        experimental_method="X-RAY",
        resolution=2.0,
        organism="Homo sapiens",
        residues=[residue],
        source_file="1abc.pdb",
        parser_version="1.0"
    )
    
    assert struct.num_atoms == 1
    assert struct.num_residues == 1
    assert struct.num_chains == 1
    assert struct.bounding_box == [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]]
    assert struct.center_of_mass == [0.0, 1.0, 2.0]

def test_protein_structure_immutability():
    atom = ProteinAtom(
        coordinates=[0.0, 0.0, 0.0],
        atom_name="CA",
        element="C",
        atom_serial=1
    )
    # Check that setting an attribute raises ValidationError / TypeError
    with pytest.raises(ValidationError):
        # Coordinates must be a list
        ProteinAtom(coordinates="invalid", atom_name="CA", element="C", atom_serial=1)
        
    with pytest.raises(Exception):
        # Verify object is frozen
        atom.atom_name = "CB"
