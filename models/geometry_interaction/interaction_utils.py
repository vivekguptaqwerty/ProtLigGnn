import torch
from torch import Tensor
from typing import Dict, Any

# Centralized Chemistry Constants for Element Mapping
# Keys are atomic numbers. Values are dictionaries of properties.
ATOMIC_PROPERTIES_TABLE: Dict[int, Dict[str, float]] = {
    5: {  # B (Boron)
        "atomic_number": 5.0,
        "atomic_mass": 10.811,
        "electronegativity": 2.04,
        "atomic_radius": 0.90,
        "covalent_radius": 0.82,
        "vdw_radius": 1.92,
        "period": 2.0,
        "group": 13.0,
        "common_valence": 3.0
    },
    6: {  # C (Carbon)
        "atomic_number": 6.0,
        "atomic_mass": 12.011,
        "electronegativity": 2.55,
        "atomic_radius": 0.77,
        "covalent_radius": 0.77,
        "vdw_radius": 1.70,
        "period": 2.0,
        "group": 14.0,
        "common_valence": 4.0
    },
    7: {  # N (Nitrogen)
        "atomic_number": 7.0,
        "atomic_mass": 14.007,
        "electronegativity": 3.04,
        "atomic_radius": 0.75,
        "covalent_radius": 0.75,
        "vdw_radius": 1.55,
        "period": 2.0,
        "group": 15.0,
        "common_valence": 3.0
    },
    8: {  # O (Oxygen)
        "atomic_number": 8.0,
        "atomic_mass": 15.999,
        "electronegativity": 3.44,
        "atomic_radius": 0.73,
        "covalent_radius": 0.73,
        "vdw_radius": 1.52,
        "period": 2.0,
        "group": 16.0,
        "common_valence": 2.0
    },
    9: {  # F (Fluorine)
        "atomic_number": 9.0,
        "atomic_mass": 18.998,
        "electronegativity": 3.98,
        "atomic_radius": 0.71,
        "covalent_radius": 0.71,
        "vdw_radius": 1.47,
        "period": 2.0,
        "group": 17.0,
        "common_valence": 1.0
    },
    15: {  # P (Phosphorus)
        "atomic_number": 15.0,
        "atomic_mass": 30.974,
        "electronegativity": 2.19,
        "atomic_radius": 1.10,
        "covalent_radius": 1.06,
        "vdw_radius": 1.80,
        "period": 3.0,
        "group": 15.0,
        "common_valence": 5.0
    },
    16: {  # S (Sulfur)
        "atomic_number": 16.0,
        "atomic_mass": 32.06,
        "electronegativity": 2.58,
        "atomic_radius": 1.03,
        "covalent_radius": 1.02,
        "vdw_radius": 1.80,
        "period": 3.0,
        "group": 16.0,
        "common_valence": 2.0
    },
    17: {  # Cl (Chlorine)
        "atomic_number": 17.0,
        "atomic_mass": 35.45,
        "electronegativity": 3.16,
        "atomic_radius": 0.99,
        "covalent_radius": 0.99,
        "vdw_radius": 1.75,
        "period": 3.0,
        "group": 17.0,
        "common_valence": 1.0
    },
    35: {  # Br (Bromine)
        "atomic_number": 35.0,
        "atomic_mass": 79.904,
        "electronegativity": 2.96,
        "atomic_radius": 1.14,
        "covalent_radius": 1.14,
        "vdw_radius": 1.85,
        "period": 4.0,
        "group": 17.0,
        "common_valence": 1.0
    },
    53: {  # I (Iodine)
        "atomic_number": 53.0,
        "atomic_mass": 126.90,
        "electronegativity": 2.66,
        "atomic_radius": 1.33,
        "covalent_radius": 1.33,
        "vdw_radius": 1.98,
        "period": 5.0,
        "group": 17.0,
        "common_valence": 1.0
    }
}

DEFAULT_PROPERTIES = {
    "atomic_number": 6.0,
    "atomic_mass": 12.011,
    "electronegativity": 2.55,
    "atomic_radius": 0.77,
    "covalent_radius": 0.77,
    "vdw_radius": 1.70,
    "period": 2.0,
    "group": 14.0,
    "common_valence": 4.0
}


class AtomicPropertyEncoder:
    """Encoder for standardizing and scaling atomic properties dynamically."""
    @staticmethod
    def get_properties_vector(atomic_num: int) -> Tensor:
        """Returns a 9-dimensional normalized float tensor of atomic properties."""
        props = ATOMIC_PROPERTIES_TABLE.get(atomic_num, DEFAULT_PROPERTIES)
        return torch.tensor([
            props["atomic_number"] / 100.0,
            props["atomic_mass"] / 250.0,
            props["electronegativity"] / 4.0,
            props["atomic_radius"] / 3.0,
            props["covalent_radius"] / 3.0,
            props["vdw_radius"] / 3.0,
            props["common_valence"] / 7.0,
            props["period"] / 7.0,
            props["group"] / 18.0
        ], dtype=torch.float32)

    @staticmethod
    def get_electronegativity(atomic_num: int) -> float:
        """Helper to get Pauling electronegativity value."""
        return ATOMIC_PROPERTIES_TABLE.get(atomic_num, DEFAULT_PROPERTIES)["electronegativity"]


def compute_pairwise_distances(pos1: Tensor, pos2: Tensor) -> Tensor:
    """Computes pairwise Euclidean distances between two sets of positions.

    Args:
        pos1: Shape [N1, 3]
        pos2: Shape [N2, 3]
    Returns:
        Shape [N1, N2]
    """
    diff = pos1.unsqueeze(1) - pos2.unsqueeze(0)  # [N1, N2, 3]
    return torch.sqrt(torch.sum(diff ** 2, dim=-1) + 1e-12)
