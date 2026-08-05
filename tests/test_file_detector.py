import pytest
import tempfile
import os
from ligprotgnnx.preprocessing.file_detector import FileTypeDetector
from ligprotgnnx.preprocessing.exceptions import UnsupportedFormatError, CorruptedFileError

def test_file_type_detector():
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Test empty file exception
        empty_path = os.path.join(tmpdir, "empty.pdb")
        with open(empty_path, "w") as f:
            pass
        with pytest.raises(CorruptedFileError):
            FileTypeDetector.detect_format(empty_path)
            
        # 2. Test compressed format rejection
        gz_path = os.path.join(tmpdir, "test.pdb.gz")
        with open(gz_path, "w") as f:
            f.write("ATOM")
        with pytest.raises(UnsupportedFormatError) as excinfo:
            FileTypeDetector.detect_format(gz_path)
        assert "Compressed formats" in str(excinfo.value)
        
        # 3. Test content inspection fallback for unknown extension
        unknown_pdb = os.path.join(tmpdir, "struct.unknown")
        with open(unknown_pdb, "w") as f:
            f.write("HEADER    PROTEIN BINDING                   18-JUL-26\nATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 10.00\n")
        assert FileTypeDetector.detect_format(unknown_pdb) == "pdb"
        
        # 4. Test mmCIF magic content header
        unknown_cif = os.path.join(tmpdir, "struct.cif_unknown")
        with open(unknown_cif, "w") as f:
            f.write("data_test\nloop_\n_atom_site.id\n")
        assert FileTypeDetector.detect_format(unknown_cif) == "mmcif"
