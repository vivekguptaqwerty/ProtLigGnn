import os
from ligprotgnnx.utils.logger import setup_logger

def test_phase0_setup():
    logger = setup_logger("test")
    logger.info("Phase 0 setup verified successfully.")
    assert os.path.exists("d:/ProtLigGnn/config/config.yaml")
