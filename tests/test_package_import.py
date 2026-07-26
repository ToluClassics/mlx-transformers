import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path


class TestPackageImportContract(unittest.TestCase):
    def test_version_import_does_not_require_mlx_runtime(self):
        script = textwrap.dedent(
            """
            import sys

            class BlockMlxImports:
                def find_spec(self, fullname, path=None, target=None):
                    if fullname == "mlx" or fullname.startswith("mlx."):
                        raise ModuleNotFoundError("MLX intentionally unavailable")
                    return None

            sys.meta_path.insert(0, BlockMlxImports())

            import mlx_transformers

            assert mlx_transformers.__version__
            assert "mlx" not in sys.modules
            """
        )
        environment = os.environ.copy()
        source_root = str(Path(__file__).resolve().parents[1] / "src")
        existing_python_path = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            source_root
            if not existing_python_path
            else os.pathsep.join((source_root, existing_python_path))
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            env=environment,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
