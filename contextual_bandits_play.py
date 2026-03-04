from bayesianbandits import Arm, NormalRegressor, ContextualAgent, ThompsonSampling
from sklearn.metrics import mean_squared_error
import numpy as np


X = np.random.normal(size=(100, 2))  # 100 samples, 2 features
y = X[:, 0] * 3.3 - X[:, 1] * 2.1 + 0.5 + np.random.normal(size=100)  # Linear combination with noise

regressor = NormalRegressor(alpha=1.0, beta=1.0)
regressor.fit(X, y)

y_ = regressor.predict(X)

mean_squared_error(y, y_), mean_squared_error(np.random.normal(loc=y.mean(), scale=y.std(), size=100), y_)


alphas = [1.0] * 4
betas = [4.0, 3.0, 2.0, 1.0]
sampled_probabilities = [
            np.random.beta(alphas[i], betas[i]) for i in range(len(alphas))
        ]

print(sampled_probabilities)

