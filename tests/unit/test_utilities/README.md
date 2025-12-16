# Utilities Unit Tests

This directory contains **unit tests for utility functions** used throughout `rice_ml`.

Utilities are tested independently since many algorithms depend on them.

---

## Modules Covered

- Metrics (euclidean_dist, manhattan_dist, r2_score)
- Preprocessing utilities (normalize, train_test_split)
- Postprocessing utilities (majority_label, average_label)

---

## What Is Tested

- Mathematical correctness of metrics
- Input validation and error handling
- Output consistency and invariants

---

## Design Principle

Utilities must be correct and stable, as errors here propagate to all models.