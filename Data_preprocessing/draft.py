import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------
# PARAMETERS
# -------------------------------
target_name = 'high'
n_clusters = 5       # number of clusters for KMeans
pca_variance = 0.9   # percentage of variance to keep in PCA






# -------------------------------
# LOAD AND CLEAN DATA
# -------------------------------
data = pd.read_pickle('scaled_data.pkl')
data_clean = data.dropna()
print("Removed", len(data) - len(data_clean), "rows due to NaN")

X = data_clean.drop(columns=[target_name])
y = data_clean[target_name]

# -------------------------------
# SPLIT DATA
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# PCA FOR KMEANS
# -------------------------------
pca = PCA(n_components=pca_variance)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

print("PCA n_components chosen:", pca.n_components_)

# -------------------------------
# KMEANS ON PCA PROJECTION
# -------------------------------
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(X_train_pca)

train_clusters = kmeans.labels_
test_clusters = kmeans.predict(X_test_pca)

# -------------------------------
# AUGMENT ORIGINAL DATA WITH CLUSTER LABELS
# -------------------------------
X_train_aug = X_train.copy()
X_train_aug['cluster'] = train_clusters

X_test_aug = X_test.copy()
X_test_aug['cluster'] = test_clusters

# -------------------------------
# REGRESSION
# -------------------------------
reg = LinearRegression()
reg.fit(X_train_aug, y_train)
y_pred = reg.predict(X_test_aug)

# -------------------------------
# RESULTS
# -------------------------------


# -------------------------------
# OPTIONAL: CORRELATION WITH TARGET
# -------------------------------
y_series = pd.Series(y_train, name=target_name)
df_train_aug = pd.concat([X_train_aug, y_series], axis=1)
corr = df_train_aug.corr()
print("\nTop correlations with target:")
print(corr[target_name].sort_values(ascending=False))

print("MSE:", mean_squared_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))