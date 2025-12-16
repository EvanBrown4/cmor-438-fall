# Utilities Unit Tests

This directory contains **unit tests for utility functions** used throughout `rice_ml`.

Utilities are tested independently since many algorithms depend on them.

---

## Modules Covered

- Metrics (Euclidean distance, manhattance distance, R^2 score)
- Preprocessing utilities (Data normalizing, train/test splitting)
- Postprocessing utilities (Majority and average labels)
- Result information (Confusion matrix)

---

## What Is Tested

- Mathematical correctness of metrics
- Input validation and error handling
- Output consistency and invariants

---

## Design Principle

Utilities must be correct and stable, as errors here propagate to all models.