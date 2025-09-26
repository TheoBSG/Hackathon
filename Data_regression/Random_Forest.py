import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_diabetes

#TO MODIFY
target_name = 'high'


data = pd.read_pickle('scaled_data.pkl')  
data_clean = data.dropna()

print("we removed ", len(data)-len(data_clean), " elements because of NaN")

X=data_clean.drop(columns=[target_name]) 
y=data_clean[target_name]


print(data_clean.head())

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


rf_regressor = RandomForestRegressor(
    n_estimators=50,           # Reduce number of trees
    max_depth=15,              # Allow deeper trees since you have lots of data
    min_samples_split=50,      # Require more samples to split (reduces overfitting)
    min_samples_leaf=20,       # Larger leaf size
    max_features=0.3,          # Fewer features per split
    n_jobs=-1,                 # Use all cores
    verbose=1,
    random_state=42
)
# Create and train the Random Forest regressor
# rf_regressor = RandomForestRegressor(
#     n_estimators=100,
#     verbose=1, 
#     random_state=42,
#     max_depth=10
# )

# Train the model
rf_regressor.fit(X_train, y_train)

# Make predictions
y_pred = rf_regressor.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

print(data.head())