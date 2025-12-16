# Utilities

This module contains shared utility functions used across `rice_ml`.

Utilities are separated to ensure reuse and consistent behavior across models.

## Modules Included

- Validation helpers
- Metrics (Euclidean distance, manhattan distance, R^2 score)
- Preprocessing utilities (Data normalizing, train/test splitting)
- Postprocessing utilities (Majority and average labels)
- Results containers (Confusion matrix helpers)

## Design Notes

- Utilities are tested independently
- Strong input validation guarantees
- Errors here propagate widely, so correctness is critical
