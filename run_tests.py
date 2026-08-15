#!/usr/bin/env python3
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import pytest
sys.exit(pytest.main([str(ROOT / "tests"), "-q", "--tb=short"]))
