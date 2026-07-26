import os
import unittest


RUN_HUB_TESTS_ENV = "MLX_TRANSFORMERS_RUN_HUB_TESTS"
_TRUE_VALUES = {"1", "true", "yes"}


def hub_tests_enabled() -> bool:
    return os.environ.get(RUN_HUB_TESTS_ENV, "").lower() in _TRUE_VALUES


requires_hub = unittest.skipUnless(
    hub_tests_enabled(),
    f"set {RUN_HUB_TESTS_ENV}=1 to run tests that download Hub models",
)
