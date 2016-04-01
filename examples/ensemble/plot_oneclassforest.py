import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import OneClassRF
from sklearn.preprocessing import scale

rng = np.random.RandomState(42)

# Generate train data
X = 0.3 * rng.randn(100, 2)
X_train = np.r_[X + 2, X - 2]

# fit the model
clf = OneClassRF(n_estimators=1, max_samples=1., max_depth=4, random_state=rng)
clf.fit(X_train)
# y_pred_train = -clf.decision_function(X_train)
# y_pred_test = -clf.decision_function(X_test)
# y_pred_outliers = -clf.decision_function(X_outliers)

# plot the line, the samples, and the nearest vectors to the plane
xx, yy = np.meshgrid(np.linspace(-5, 5, 200), np.linspace(-5, 5, 200))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
#Z = scale(Z)
Z = np.log(Z)

#plt.title("OneClassRF")

levels = np.linspace(Z.min(), Z.max(), 1000)
plt.contourf(xx, yy, Z, cmap=plt.cm.Blues, levels=levels)
plt.scatter(X_train[:, 0], X_train[:, 1], c='white')
plt.axis('tight')
plt.xlim((-5, 5))
plt.ylim((-5, 5))
# plt.legend([b1, b2, c],
#            ["training observations",
#             "new regular observations", "new abnormal observations"],
#            loc="upper left")
plt.show()
