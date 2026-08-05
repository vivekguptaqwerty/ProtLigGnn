import os
import pytest
import tempfile
from ligprotgnnx.datasets.registry import DatasetRegistry, DatasetMetadata

def test_dataset_registry_persistence():
    with tempfile.TemporaryDirectory() as tmpdir:
        reg_file = os.path.join(tmpdir, "test_registry.json")
        registry = DatasetRegistry(registry_file=reg_file)
        
        meta = DatasetMetadata(
            dataset_name="PDBbind_refined_test",
            dataset_version="2020",
            download_date="2026-07-18",
            source="http://www.pdbbind.org.cn/",
            checksum="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            license="academic",
            num_complexes=100,
            num_failures=0,
            processing_version="1.0",
            cache_version="1.0",
            statistics={"mean_affinity": 6.5}
        )
        
        registry.register_dataset("test_set", meta)
        
        # Reload from disk
        new_registry = DatasetRegistry(registry_file=reg_file)
        loaded = new_registry.get_dataset("test_set")
        assert loaded is not None
        assert loaded.dataset_name == "PDBbind_refined_test"
        assert loaded.statistics["mean_affinity"] == 6.5
