# Preprocessing Utilities Unit Tests

This directory contains **unit tests for preprocessing utilities**.

Preprocessing functions are foundational and must behave consistently
across all algorithms.

---

## Functions Covered

- Data normalization
- Train-test splitting

---

## What Is Tested

- Input validation and edge cases
- Shape preservation
- Deterministic behavior with fixed seeds
- Correct statistical properties where applicable

---

## Design Principle

Preprocessing tests enforce **strong guarantees** with many tests, as these utilities are
used across many algorithms.