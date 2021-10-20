# The MIT License (MIT)
#
# Copyright (c) 2021 CNRS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from collections import Counter

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection._split import _CVIterableWrapper


class _Passthrough(BaseEstimator):
    def __init__(self):
        super().__init__()
        self.classes_ = np.array([False, True], dtype=np.bool)

    def fit(self, X, y):
        return self

    def decision_function(self, X=None):
        return X


class Calibration:
    """Probability calibration for binary classification tasks

    Parameters
    ----------
    method : {'isotonic', 'sigmoid'}, optional
        See `CalibratedClassifierCV`. Defaults to 'isotonic'.
    equal_priors : bool, optional
        Set to True to force equal priors. Default behavior is to estimate
        priors from the data itself.

    Usage
    -----
    >>> calibration = Calibration()
    >>> calibration.fit(train_score, train_y)
    >>> test_probability = calibration.transform(test_score)

    See also
    --------
    CalibratedClassifierCV

    """

    def __init__(self, equal_priors=False, method="isotonic"):
        super().__init__()
        self.method = method
        self.equal_priors = equal_priors

    def fit(self, scores, y_true):
        """Train calibration

        Parameters
        ----------
        scores : (n_samples, ) array-like
            Uncalibrated scores.
        y_true : (n_samples, ) array-like
            True labels (dtype=bool).
        """

        # to force equal priors, randomly select (and average over)
        # up to fifty balanced (i.e. #true == #false) calibration sets.
        if self.equal_priors:

            counter = Counter(y_true)
            positive, negative = counter[True], counter[False]

            if positive > negative:
                majority, minority = True, False
                n_majority, n_minority = positive, negative
            else:
                majority, minority = False, True
                n_majority, n_minority = negative, positive

            n_splits = min(50, n_majority // n_minority + 1)

            minority_index = np.where(y_true == minority)[0]
            majority_index = np.where(y_true == majority)[0]

            cv = []
            for _ in range(n_splits):
                test_index = np.hstack(
                    [
                        np.random.choice(
                            majority_index, size=n_minority, replace=False
                        ),
                        minority_index,
                    ]
                )
                cv.append(([], test_index))
            cv = _CVIterableWrapper(cv)

        # to estimate priors from the data itself, use the whole set
        else:
            cv = "prefit"

        self.calibration_ = CalibratedClassifierCV(
            base_estimator=_Passthrough(), method=self.method, cv=cv
        )
        self.calibration_.fit(scores.reshape(-1, 1), y_true)

        return self

    def transform(self, scores):
        """Calibrate scores into probabilities

        Parameters
        ----------
        scores : (n_samples, ) array-like
            Uncalibrated scores.

        Returns
        -------
        probabilities : (n_samples, ) array-like
            Calibrated scores (i.e. probabilities)
        """
        return self.calibration_.predict_proba(scores.reshape(-1, 1))[:, 1]
