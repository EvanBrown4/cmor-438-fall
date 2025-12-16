# Tests

This directory contains all tests for the `rice_ml` package.

The goal of the test suite is to verify **correctness, stability, and reproducibility**
of all machine learning algorithms implemented from scratch in this library.

---

## Testing Philosophy

The tests in this project are designed to:

- Verify mathematical and logical correctness
- Enforce strict input validation and error handling
- Ensure deterministic behavior when randomness is involved
- Validate that algorithms compose correctly into pipelines

These tests **do not** benchmark against external libraries.

---

## Directory Structure

```
tests/
├── unit/
├── integration/
```

- **unit/** — isolated correctness and validation tests
- **integration/** — end-to-end pipeline and interaction tests

---

## What Is Tested

- Input shapes, types, and parameter constraints
- Output shapes and invariants
- Core algorithm behavior on small datasets
- Controlled randomness via `random_state`
- End-to-end model workflows

---

## What Is Not Tested

- Performance benchmarking
- Comparison to `scikit-learn`
- Probabilistic pass conditions

---

## Running the Tests

```bash
pytest
pytest tests/unit
pytest tests/integration
```

All tests are expected to be deterministic and reproducible.