import os
from pathlib import Path
from ligprotgnnx.preprocessing.exceptions import UnsupportedFormatError, CorruptedFileError

class FileTypeDetector:
    SUPPORTED_EXTENSIONS = {".pdb", ".ent", ".cif", ".mmcif"}

    @classmethod
    def detect_format(cls, path: str) -> str:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if p.stat().st_size == 0:
            raise CorruptedFileError(
                file_path=path,
                parser_name="FileTypeDetector",
                reason="File is empty",
                recovery_suggestion="Verify structure file generation or download source."
            )
            
        ext = p.suffix.lower()
        # Handle secondary extension mapping like .pdb.gz as compressed/unsupported
        if p.suffixes and len(p.suffixes) > 1 and p.suffixes[-1].lower() in {".gz", ".zip", ".bz2", ".tar"}:
            raise UnsupportedFormatError(
                file_path=path,
                parser_name="FileTypeDetector",
                reason=f"Compressed formats ({p.suffixes[-1]}) are unsupported",
                recovery_suggestion="Decompress the file before passing it to the parser."
            )

        if ext not in cls.SUPPORTED_EXTENSIONS:
            # Fallback to header inspection if extension is unknown or missing
            return cls._detect_from_content(path)

        if ext in {".pdb", ".ent"}:
            return "pdb"
        elif ext in {".cif", ".mmcif"}:
            return "mmcif"
        
        raise UnsupportedFormatError(
            file_path=path,
            parser_name="FileTypeDetector",
            reason=f"Extension '{ext}' is unsupported",
            recovery_suggestion="Supported extensions are: " + ", ".join(cls.SUPPORTED_EXTENSIONS)
        )

    @classmethod
    def _detect_from_content(cls, path: str) -> str:
        # Inspect first 1024 bytes of content
        try:
            with open(path, "r", errors="ignore") as f:
                header = f.read(1024)
        except Exception as e:
            raise CorruptedFileError(
                file_path=path,
                parser_name="FileTypeDetector",
                reason=f"Failed to read file header: {e}",
                recovery_suggestion="Ensure file is not corrupted and has read permissions."
            )

        if "data_" in header or "loop_" in header:
            return "mmcif"
        elif "ATOM  " in header or "HETATM" in header or "HEADER" in header:
            return "pdb"
            
        raise UnsupportedFormatError(
            file_path=path,
            parser_name="FileTypeDetector",
            reason="Could not determine file format from magic header or content heuristics",
            recovery_suggestion="Ensure file complies with standard PDB or mmCIF formats."
        )
