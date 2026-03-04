from ast import Is
from sklearn.ensemble import IsolationForest

import numpy as np

X = np.random.normal(size=(1000, 10))
X[30, 3] = 1000

IF = IsolationForest(contamination=0.01)
IF.fit(X)

pred = IF.predict(X)
scores = IF.score_samples(X)
print(pred[25:35])
print(scores[25:35])