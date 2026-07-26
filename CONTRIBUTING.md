# Contributing to mlx-transformers

Contributions to mlx-transformers are welcome. Reliability fixes, tests,
documentation, and focused model improvements are especially useful.

## Pull Requests

1. Fork the repository and submit a focused pull request.
2. Add or update tests for behavior changed by the pull request.
3. Run the bounded offline suite and quality checks below.
4. Keep tests that download Hub models behind the explicit integration gate.
5. Obtain at least one review before merge.

## Local validation

The default suite is network-free. It discovers every test but skips Hub model
tests unless they are explicitly enabled:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m unittest discover -s tests -v
```

The expected baseline is 98 discovered tests: 76 pass and 22 Hub tests skip.
The exact count should change only when tests are intentionally added or
removed.

Install the same quality tools used by CI and run all checks:

```bash
python -m pip install "ruff==0.3.0" build twine pre-commit
ruff check .
ruff format --check .
python -m build
python -m twine check dist/*
pre-commit validate-config
```

To install the hooks locally:

```bash
pre-commit install
pre-commit run --all-files
```

The hooks run Ruff lint and formatting checks.

## Hub integration tests

Hub tests can download large or gated checkpoints. Run them only after
reviewing the model IDs and expected storage requirements:

```bash
MLX_TRANSFORMERS_RUN_HUB_TESTS=1 python -m unittest discover -s tests -v
```

Set `HF_TOKEN` only when an explicitly reviewed gated checkpoint requires it.
Never commit credentials or include them in logs.

## Issues

We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

## License

By contributing to mlx-transformers, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
