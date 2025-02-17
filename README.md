# calcurve

Compute and visualize calibration curves with confidence intervals for
scikit-learn binary classifiers.

DISCLAIMER: this code was generated using Windsurf IDE's Cascade agent with
some guidance as an experiment to assess the productivity of working with such
a tool.

The code is not necessarily correct, the tests are quite shallow, the analysis
in the notebook is sometimes misled or inconsident with the plots, and I did
not review them all.

I reported some feedback on the main limitations I found when working with
this tool:

- https://codeium.canny.io/feature-requests/p/do-not-let-cascade-attempt-to-generate-jupyter-notebooks-from-scratch
- https://codeium.canny.io/feature-requests/p/cascade-pretends-it-can-see-the-matplotlib-plots-embedded-in-a-jupyter-notebook


## Features

- Multiple binning strategies:
  - Quantile-based (default)
  - Uniform grid
  - Custom bin edges

- Confidence interval methods:
  - Clopper-Pearson exact intervals
  - Wilson CC intervals
  - Bootstrap resampling with interpolation

## Installation

```bash
pip install .
```

For development installation with test dependencies:

```bash
pip install -e ".[dev,test]"
```

## Usage

```python
from calcurve import CalibrationCurve
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Generate sample data
X, y = make_classification(n_samples=1000, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Train a classifier
clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred_proba = clf.predict_proba(X_test)[:, 1]

# Compute calibration curve with confidence intervals
cal_curve = CalibrationCurve(
    binning_strategy="quantile",
    n_bins=7,
    confidence_method="clopper_pearson",
    confidence_level=0.90,
)
cal_curve.fit(y_test, y_pred_proba)
cal_curve.plot()
```

## Running Tests

To run the test suite:

```bash
pytest .
```

This will run all tests and generate a coverage report.
