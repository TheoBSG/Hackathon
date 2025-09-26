import pandas as pd

import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#TO MODIFY
target_name = 'high'



data = pd.read_pickle('scaled_data.pkl')  
data_clean = data.dropna()

print("we removed ", len(data)-len(data_clean), " because of NaN")

X=data_clean.drop(columns=[target_name]) 
y=data_clean[target_name]


print(data_clean.head())

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)




from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Dimensionality reduction
pca = PCA(n_components=0.9)  # keep 95% of variance
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

print("Chosen n_components:", pca.n_components_)

# Regression
reg = LinearRegression()
reg.fit(X_train_pca, y_train)
y_pred = reg.predict(X_test_pca)



# If y_train is a numpy array
y_series = pd.Series(y_train, name=target_name)

# Concatenate with features
df_train = pd.concat([pd.DataFrame(X_train, columns=X.columns), y_series], axis=1)

# Correlation of all features with target
corr = df_train.corr()
print(corr[target_name].sort_values(ascending=False))

print("MSE:", mean_squared_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))