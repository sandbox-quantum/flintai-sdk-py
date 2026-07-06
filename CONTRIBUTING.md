# Contributing to FlintAI SDK (Python)

Thank you for your interest in contributing!

## Getting Started

1. Fork the repository and clone your fork.
2. Create a virtual environment and install dev dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   ```
3. Create a feature branch: `git checkout -b my-feature`

## Running Tests

```bash
pytest
```

## Code Style

This project uses [ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
ruff check .
ruff format .
```

Type checking with mypy:

```bash
mypy src/
```

## Submitting a Pull Request

1. Ensure all tests pass and there are no lint errors.
2. Write a clear PR description explaining what the change does and why.
3. Reference any relevant issues.

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.
