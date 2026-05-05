"""Script to run pytest with proper path setup."""
import sys
import os

# Add src to path
src_path = os.path.join(os.path.dirname(__file__), "src")
sys.path.insert(0, src_path)

import pytest
sys.exit(pytest.main(sys.argv[1:] if len(sys.argv) > 1 else []))
