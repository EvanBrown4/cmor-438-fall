# Unit Tests

This directory contains **unit tests** for individual components of the `rice_ml` package.

Unit tests verify correctness in isolation, without relying on interactions with other algorithms.

---

## Purpose

Unit tests ensure that:

- Each algorithm behaves correctly on its own
- Input validation is strict and explicit
- Errors are raised early and clearly
- Output shapes and types are consistent

A failing unit test should indicate a **localized logic error**.

---

## What Is Tested

- Input shape and type validation
- Parameter bounds and constraints
- Core mathematical operations
- Metric computations (e.g., MSE, R², accuracy)
- Deterministic behavior when applicable

---

## What Is Avoided

- Pipeline composition
- Large datasets
- Uncontrolled randomness
- Performance comparisons

---

## Design Principle

Unit tests should be fast, deterministic, and narrowly scoped.