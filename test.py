#%%
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
X = np.ones((17, 2))
y = np.array([0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
groups = np.array([0, 2, 1, 1, 1, 1, 1, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11])
cv = StratifiedGroupKFold(n_splits=5)
for train_idxs, test_idxs in cv.split(X, y, groups):
    print("TRAIN:", groups[train_idxs])
    print("      ", y[train_idxs])
    print(" TEST:", groups[test_idxs])
    print("      ", y[test_idxs])
# %%
