# Integration Tests

This directory contains **integration tests** for the `rice_ml` package.

Integration tests verify that independently correct components interact correctly
when composed into full machine learning workflows.

---

## Purpose

Integration tests ensure that:

- Algorithms compose correctly into pipelines
- Randomness is propagated and controlled consistently
- End-to-end workflows behave as expected

These tests validate **system-level behavior**, not isolated logic.

---

## What Is Tested

- PCA → classifier pipelines
- Decision trees → ensemble methods
- End-to-end fit → predict flows
- Reproducibility across runs with fixed seeds
- Error propagation across components

---

## Error Path Integration

Integration tests explicitly verify that:

- Invalid pipelines fail early
- Errors are informative and consistent
- No silent failures occur

---

## Determinism and Reproducibility

When `random_state` is provided:

- Multiple runs produce identical results
- Randomness is controlled, not removed

---

## Design Principle

A failing integration test indicates a **systemic interaction issue**
between otherwise correct components.