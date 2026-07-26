from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mlx-transformers")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = ["__version__"]
