import time
import hashlib
from datetime import datetime
from typing import Optional, Callable, Any

from ligprotgnnx.preprocessing.protein_structure import ProteinStructure
from ligprotgnnx.preprocessing.file_detector import FileTypeDetector
from ligprotgnnx.preprocessing.parser_registry import ParserRegistry
from ligprotgnnx.preprocessing.exceptions import ProteinParserError
from ligprotgnnx.utils.logger import setup_logger

logger = setup_logger("protein_parser")

class ProteinParser:
    @classmethod
    def load(
        cls,
        path: str,
        validator: Optional[Any] = None
    ) -> ProteinStructure:
        start_time = time.time()
        
        # 1. Format detection
        try:
            file_format = FileTypeDetector.detect_format(path)
        except Exception as e:
            logger.error(f"File format detection failed for path '{path}': {e}")
            raise
            
        # 2. Get registered parser class
        try:
            parser_cls = ParserRegistry.get_parser(file_format)
            parser_instance = parser_cls()
        except Exception as e:
            logger.error(f"Parser lookup failed for format '{file_format}': {e}")
            raise
            
        # 3. Parse structure
        try:
            structure = parser_instance.parse(path)
        except Exception as e:
            logger.error(f"Parsing failed for '{path}' using '{parser_cls.__name__}': {e}")
            raise
            
        # 4. Generate file checksum
        sha = hashlib.sha256()
        try:
            with open(path, "rb") as f:
                while chunk := f.read(65536):
                    sha.update(chunk)
            checksum = sha.hexdigest()
        except Exception:
            checksum = "unknown"

        # 5. Metadata injection
        # Pydantic model is frozen, so we construct a new one with updated metadata
        updated_residues = structure.residues
        
        # In Pydantic v2, we construct metadata as part of the structure object
        # Since ProteinStructure properties are read-only, if we need custom metadata,
        # we can pass it or print log info. Let's look at fields of ProteinStructure:
        # structure_id, experimental_method, resolution, organism, residues, source_file, parser_version.
        # If we need extra fields, we can verify that ProteinStructure does not forbid extra fields,
        # or we just write it to log and inject what is supported in schema.
        
        elapsed = time.time() - start_time
        logger.info(
            f"Successfully parsed structure '{structure.structure_id}' from '{path}' in {elapsed:.4f}s. "
            f"Format: {file_format}, Atoms: {structure.num_atoms}, Residues: {structure.num_residues}, "
            f"Chains: {structure.num_chains}, Checksum: {checksum[:8]}"
        )
        
        # 6. Validation hook
        if validator is not None:
            # validator is expected to implement .validate(structure)
            # which returns a ValidationReport or raises error
            validator.validate(structure)
            
        return structure
