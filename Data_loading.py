import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


#TO MODIFY
nom_file = 'all_stocks_5yr.csv'
target_name = 'high'

#TO MODIFY
feature_one_hot_name=[] #for features wit not a lot of categories (dont take too much space in memory)
feature_label=["date","Name"] #too much categories not enough space on the computer




data = pd.read_csv(nom_file)

print(data.head())

# Automatically detect non-numerical columns
categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

print("Categorical columns:", categorical_cols)




data_encoded = data.copy()

# SOLUTION 1: Simple OneHot Encoding with pandas (recommended)
# data_encoded[feature_one_hot_name] = pd.get_dummies(data[feature_one_hot_name], drop_first=True)

for feature in feature_one_hot_name:
    # Get one-hot encoded columns
    one_hot = pd.get_dummies(data[feature], prefix=feature, drop_first=True)
    # Drop the original column
    data_encoded = data_encoded.drop(feature, axis=1)
    # Concatenate the one-hot encoded columns
    data_encoded = pd.concat([data_encoded, one_hot], axis=1)

#Solution 2: Label encoding for categorical columns
for col in feature_label:
    data_encoded[col] = data_encoded[col].astype('category').cat.codes


# for col in feature_label:
#     data_encoded[col] = data_encoded[col].astype('category').cat.codes



#Solution 3: frequency encoding (issue unicity: same data can have same value if they have the same frequency) )
# categorical_cols = data.select_dtypes(include=['object']).columns
# for col in categorical_cols:
#     freq = data[col].value_counts()
#     data[col] = data[col].map(freq).astype('int32')
# data_encoded=data.copy()



print("First 5 rows:")
print(data_encoded.head())

columns_to_keep = [col for col in data_encoded.columns if col not in ['Unnamed: 0', 'index']]
data_encoded = data_encoded[columns_to_keep]


# x = data_encoded.drop(columns=[target_name]) 
x=data_encoded
numeric_cols_without_target = [col for col in numeric_cols if col != target_name]

# Scale numerical columns
scaler = StandardScaler()
x_scaled = x.copy()  # Start with a copy of the original
x_scaled[numeric_cols_without_target] = scaler.fit_transform(x[numeric_cols_without_target])

# Concatenate scaled features with target variable
# data_scaled = pd.concat([x_scaled, y], axis=1)

data_scaled=x_scaled

print(data_scaled.head())

# Categorical columns remain unchanged automatically
# Now add the target variable
data_scaled = x_scaled





data_scaled.to_pickle('scaled_data.pkl')


data_scaled.to_csv('data.csv')

data.to_pickle('original_data.pkl')


print("Execution completed successfully!")