import numpy as np
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict

class ProteinAtom(BaseModel):
    model_config = ConfigDict(frozen=True)
    
    coordinates: List[float] = Field(..., description="3D coordinates [x, y, z] in Angstroms")
    atom_name: str = Field(..., description="Atom name (e.g., CA)")
    element: str = Field(..., description="Chemical element (e.g., C)")
    atom_serial: int = Field(..., description="Serial number of the atom")
    occupancy: float = Field(1.0, description="Occupancy factor")
    b_factor: float = Field(0.0, description="Temperature B-factor")
    altloc: str = Field(" ", description="Alternate location indicator")
    formal_charge: float = Field(0.0, description="Formal charge of the atom")

class ProteinResidue(BaseModel):
    model_config = ConfigDict(frozen=True)
    
    residue_name: str = Field(..., description="Residue name (e.g., ALA)")
    residue_id: str = Field(..., description="Residue seq number (e.g., 42)")
    insertion_code: str = Field(" ", description="Insertion code")
    chain_id: str = Field(..., description="Chain ID (e.g., A)")
    is_hetero: bool = Field(False, description="True if HETATM")
    is_water: bool = Field(False, description="True if water molecule")
    atoms: List[ProteinAtom] = Field(default_factory=list, description="Atoms composing this residue")

class ProteinStructure(BaseModel):
    model_config = ConfigDict(frozen=True)
    
    structure_id: str = Field(..., description="Structure identifier (e.g., 1abc)")
    experimental_method: str = Field("unknown", description="Experimental method (e.g., X-RAY DIFFRACTION)")
    resolution: Optional[float] = Field(None, description="Resolution in Angstroms")
    organism: Optional[str] = Field(None, description="Source organism")
    residues: List[ProteinResidue] = Field(default_factory=list, description="Residues composing the protein")
    source_file: str = Field("unknown", description="Source path")
    parser_version: str = Field("1.0", description="Parser version")

    @property
    def num_atoms(self) -> int:
        return sum(len(r.atoms) for r in self.residues)

    @property
    def num_residues(self) -> int:
        return len(self.residues)

    @property
    def num_chains(self) -> int:
        return len(set(r.chain_id for r in self.residues))

    @property
    def bounding_box(self) -> List[List[float]]:
        coords = np.array([atom.coordinates for r in self.residues for atom in r.atoms])
        if len(coords) == 0:
            return [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        min_coords = coords.min(axis=0).tolist()
        max_coords = coords.max(axis=0).tolist()
        return [min_coords, max_coords]

    @property
    def center_of_mass(self) -> List[float]:
        coords = np.array([atom.coordinates for r in self.residues for atom in r.atoms])
        if len(coords) == 0:
            return [0.0, 0.0, 0.0]
        return coords.mean(axis=0).tolist()
