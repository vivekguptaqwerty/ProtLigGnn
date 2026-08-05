import json
import os
import threading
from typing import Dict, Optional, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field

class DatasetMetadata(BaseModel):
    dataset_name: str = Field(..., description="Name of the dataset (e.g., PDBbind_refined)")
    dataset_version: str = Field(..., description="Version identifier of the raw dataset")
    source: str = Field(..., description="Source URL or repository of raw dataset")
    download_date: Union[datetime, str] = Field(..., description="ISO 8601 date or datetime of raw dataset download")
    checksum: str = Field(..., description="SHA256 checksum of raw dataset archive")
    processing_version: str = Field("1.0", description="Version of processing code used")
    cache_version: str = Field("1.0", description="Version of graph cache layout used")
    git_commit: str = Field("unknown", description="Git commit hash of the codebase used to process this dataset")
    config_hash: str = Field("unknown", description="SHA256 hash of configuration overrides used")
    num_complexes: int = Field(0, description="Total number of complexes loaded")
    num_failed: int = Field(0, description="Number of complexes failing parsing or preprocessing")
    license: Optional[str] = Field(None, description="License under which raw dataset is distributed")
    notes: Optional[str] = Field(None, description="Additional metadata notes")
    statistics: Dict[str, Any] = Field(default_factory=dict, description="Calculated statistics of the dataset")

class DatasetRegistry:
    def __init__(self, registry_file: str = "data/dataset_registry.json"):
        self.registry_file = registry_file
        self.lock = threading.Lock()
        self.metadata_store: Dict[str, DatasetMetadata] = {}
        self._load_registry()

    def _load_registry(self) -> None:
        with self.lock:
            if os.path.exists(self.registry_file):
                try:
                    with open(self.registry_file, "r") as f:
                        data = json.load(f)
                    for key, val in data.items():
                        # Migration logic for older schemas
                        if "git_commit" not in val:
                            val["git_commit"] = "unknown"
                        if "config_hash" not in val:
                            val["config_hash"] = "unknown"
                        if "num_failed" not in val and "num_failures" in val:
                            val["num_failed"] = val.pop("num_failures")
                        elif "num_failed" not in val:
                            val["num_failed"] = 0
                            
                        self.metadata_store[key] = DatasetMetadata(**val)
                except Exception as e:
                    # Registry is corrupted, initialize empty
                    self.metadata_store = {}

    def save_registry(self) -> None:
        with self.lock:
            os.makedirs(os.path.dirname(self.registry_file), exist_ok=True)
            serialized = {}
            for k, v in self.metadata_store.items():
                m_dump = v.model_dump()
                # Serialize datetime objects to ISO strings
                if isinstance(m_dump["download_date"], datetime):
                    m_dump["download_date"] = m_dump["download_date"].isoformat()
                serialized[k] = m_dump
            with open(self.registry_file, "w") as f:
                json.dump(serialized, f, indent=2)

    def register_dataset(self, name: str, metadata: DatasetMetadata) -> None:
        with self.lock:
            self.metadata_store[name] = metadata
        self.save_registry()

    def get_dataset(self, name: str) -> Optional[DatasetMetadata]:
        with self.lock:
            return self.metadata_store.get(name)
