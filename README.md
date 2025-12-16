# rice_ml - Machine Learning Framework
Author: Evan Brown

## Package Information
rice_ml is a lightweight, educational machine learning framework implemented from scratch in Python. It provides clear, modular implementations of common supervised and unsupervised algorithms.

Jupyter notebooks are provided in the `examples/` directory to walk through each algorithm with detailed explanations, visualizations, and evaluation.

---

## Package Structure

```
rice_ml/
├── supervised_learning/
├── unsupervised_learning/
├── utilities/
├── metrics/
└── examples/
```

- **supervised_learning/** – Classification and regression models  
- **unsupervised_learning/** – Clustering and dimensionality reduction  
- **utilities/** – Data validation, preprocessing, normalization, and helper functions  
- **metrics/** – Evaluation metrics  
- **examples/** – End-to-end notebooks with explanations and visualizations  

---

## Machine Learning Algorithms

### Supervised Algorithms
- Linear Regression  
- Logistic Regression  
- K-Nearest Neighbors (KNN)  
  - Classification  
  - Regression  
- Perceptron  
- Multilayer Perceptron (MLP)  
- Decision Trees  
  - Classifier  
  - Regressor  
- Regression Trees  
  - Independent implementation with equivalent functionality to decision tree regressors  
- Ensemble Methods (Random Forest)  
  - Classifier  
  - Regressor  

### Unsupervised Algorithms
- Principal Component Analysis (PCA)  
- K-Means Clustering  
- Density-Based Spatial Clustering of Applications with Noise (DBSCAN)  
- Community Detection  

---

## Installation Information

### Requirements
- **Python ≥ 3.9**
- **Conda** (optional, for environment management)

Core dependencies are managed via `pyproject.toml` and include:

- numpy  
- pandas  
- matplotlib  
- seaborn  
- requests  
- pytest  
- networkx (only required for the community detection example notebook)

---

### 1. Activate the Environment

Recommended:
```bash
conda create -n myenv python=3.9
conda activate myenv
```

On Windows:
```bash
myenv\Scripts\activate
```

Use this environment as the kernel for notebooks in `examples/`.

---

### 2. Install the Package

```bash
pip install .
```

This installs the `rice_ml` package.

---

### Optional: Developer Installation

If you plan to modify the source code, run tests, or develop new models, install the package in editable (development) mode.

```bash
pip install -e .
pip install -e .[dev]
```

#### Running Tests

Tests are written using `pytest`.

Run all tests:
```bash
pytest
```

Run a specific subset:
```bash
pytest tests/
pytest tests/unit/
pytest tests/integration/
```

---

## Usage Example

```python
from rice_ml.neighbors import KNNClassifier

model = KNNClassifier(k=5)
model.fit(X_train, y_train)
preds = model.predict(X_test)
```

See the `examples/` directory for fully worked examples on each algorithm with explanations and visualizations.

---

## Design Choices
- From-scratch implementations (no scikit-learn internals)
- NumPy-first models
- Explicit data validation and clear error messages
- Reproducibility and deterministic behavior via optional random states

This package is for a class project and is not intended to replace production machine learning libraries.

---

## License
This project is released under the MIT License.