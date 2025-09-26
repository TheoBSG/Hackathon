import pandas as pd

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




from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(X_train,y_train)

y_pred=model.predict(X_test)


from sklearn.metrics import r2_score,mean_squared_error

print("R2_score is",r2_score(y_test,y_pred))
print("Mean Squared error is", mean_squared_error(y_test,y_pred))

print("Columns in X:", X.columns.tolist())
print("Target name:", target_name)