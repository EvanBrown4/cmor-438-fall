# rice_ml

`rice_ml` is a from-scratch machine learning library designed for **learning and correctness**, rather than production performance.

## Design Principles

- No reliance on `scikit-learn`
- NumPy-first implementations
- Explicit input validation
- Deterministic behavior via `random_state`
- Clear separation of concerns

## Package Structure

- `supervised_learning/` — models trained on labeled data
- `unsupervised_learning/` — models for structure discovery
- `utilities/` — shared helpers, metrics, and preprocessing