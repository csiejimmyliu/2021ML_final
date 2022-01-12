#%%
#import
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.feature_selection import RFECV

# from yellowbrick.model_selection import RFECV
import pandas as pd
import matplotlib.pyplot as plt

#%%

train_data = pd.read_csv('./preprocessed_data/train_data.csv')
train_data_with_label = train_data.loc[train_data['Churn Category'] != -1]
X = train_data_with_label.drop(['Customer ID', 'Churn Category'], axis=1)
y = train_data_with_label[['Churn Category']]

#%%
estimator = RandomForestClassifier(max_depth=100)
selector = RFECV(estimator, step=1, cv=5)
selector = selector.fit(X, y)
selector.ranking_

# %%
col = []
for i in range(len(selector.ranking_)):
    if selector.ranking_[i] == 1:
        col.append(i)
        
#%%
X.iloc[:, col].columns

# %%
selector.grid_scores_

#%%
print("Optimal number of features : %d" % selector.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (accuracy)")
plt.plot(
    range(1, len(selector.cv_results_['mean_test_score']) + 1),
    selector.cv_results_['mean_test_score'],
)
plt.show()

#%%
y.value_counts()