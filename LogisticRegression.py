import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from statsmodels.miscmodels.ordinal_model import OrderedModel


class OrderedLogisticClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, distr="logit", method="bfgs", maxiter=500, disp=False):
        self.distr = distr
        self.method = method
        self.maxiter = maxiter
        self.disp = disp

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.model_ = OrderedModel(y, X, distr=self.distr)
        self.result_ = self.model_.fit(
            method=self.method, maxiter=self.maxiter, disp=self.disp
        )
        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        return self.result_.model.predict(self.result_.params, exog=X)

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]

    @property
    def coef_(self):
        n_features = self.model_.exog.shape[1]
        return self.result_.params[:n_features]