# Machine Learning Framework
Author: Evan Brown

## Package Information


## Machine Learning Algorithms
### Supervised Algorithms
- KNN

### Unsupervised Algorithms



## Installation Information
### Requirements
## Requirements

- **Python ≥ 3.9**
- Conda (recommended for environment management)

Core dependencies are managed via `pyproject.toml` and include:

- numpy
- pandas
- matplotlib
- seaborn
- requests
- pytest


### 1. To activate the environment

Call myenv/Scripts/activate
Use this environment for the notebooks' kernels in examples/

### 2. Install the package
**pip install .**
This installs the rice_ml package.

### Usage Example
>>>from rice_ml.neighbors import KNNClassifier
>>>
>>>model = KNNClassifier(k=5)
>>>model.fit(X_train, y_train)
>>>preds = model.predict(X_test)
See the *examples/* folder for fully worked examples on each algorithm with explanations and visualizations.