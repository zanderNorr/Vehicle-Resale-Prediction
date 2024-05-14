from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('LogTransformed_final_data.csv') 
body_style = ['SUV', 'Sedan', 'Convertible', 'Coupe', 'Wagon', 'Hatchback', 'Truck', 'Van']
print('Ridge Regression Results of R^2:')
for style in body_style:
    new_df = df[df['body'].isin([style])]
    new_df = new_df[new_df['year'].isin([i for i in range(2000, 2016)])]
    new_df = new_df[new_df['sellingprice'] > 0]
    new_df['logprice'] = np.log(new_df['sellingprice'])
    X = new_df[['model', 'odometer', 'year', 'condition', 'state', 'color']]

    y = new_df[['logprice']]


    # One-hot encode categorical variables
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['model', 'state', 'color'])
        ],
        remainder='passthrough')

    X_transformed = preprocessor.fit_transform(X)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.25, random_state=42)

    ridge_reg = Ridge(alpha=1)

    ridge_reg.fit(X_train, y_train)

    y_pred = ridge_reg.predict(X_test)
    train_score = ridge_reg.score(X_train, y_train)
    test_score = ridge_reg.score(X_test, y_test)
    print(f"R^2 Score of {style}: {r2_score(y_test, y_pred):4f}")
    x = np.exp(y_test)
    y = np.exp(y_pred)
    mae = mean_absolute_error(x, y)
    print(f'Mean Absolute Error: {mae:4f}')
    plt.figure(figsize=(10, 5))
    plt.scatter(x, y)
    plt.xlabel('Actual values')
    plt.ylabel('Predicted values')
    plt.title(f'Actual vs. Predicted values for {style}')
    plt.axis('equal')
    plt.axline((0, 0), slope=1, color = 'red')
    plt.grid(True)
    plt.show()
    print(f'Training score: {train_score:4f}')
    print(f'Test score: {test_score:4f}')