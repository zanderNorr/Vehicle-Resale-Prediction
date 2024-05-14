from flask import Flask, request, jsonify, render_template
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
app = Flask(__name__)
df = pd.read_csv('C:\\Users\\znorr\\OneDrive\\Desktop\\DataProject\\cleaned_final_data.csv')

# X = df[['make', 'odometer', 'year', 'condition']]
# y = df['sellingprice']

new_df = df[df['year'].isin([i for i in range(2000, 2016)])]
new_df = new_df[new_df['sellingprice'] > 0]
new_df['logprice'] = np.log(new_df['sellingprice'])
X = new_df[['make', 'body', 'odometer', 'year', 'condition']]

y = new_df[['logprice']]


    # One-hot encode categorical variables
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['make', 'body'])
    ],
    remainder='passthrough')

ridge_reg = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', Ridge(alpha=1))
])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

ridge_reg.fit(X_train, y_train)

joblib.dump(ridge_reg, 'ridge_model.pkl')
# y_pred = ridge_reg.predict(X_test)
# train_score = ridge_reg.score(X_train, y_train)
# test_score = ridge_reg.score(X_test, y_test)
# print(f"R^2 Score: {r2_score(y_test, y_pred):4f}")
# x = np.exp(y_test)
# y = np.exp(y_pred)
# mae = mean_absolute_error(x, y)
# print(f'Mean Absolute Error: {mae:4f}')
# Define the home route
model = joblib.load('ridge_model.pkl')
@app.route('/')
def home():
    return render_template('index.html')

# Define the predict route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the form
        data = {
            'make': [request.form['make']],
            'body': [request.form['body']],
            'odometer': [int(request.form['odometer'])],
            'year': [int(request.form['year'])],
            'condition': [int(request.form['condition'])],
        }
        inputs = pd.DataFrame(data)
        prediction_log = model.predict(inputs)
        prediction_exp = np.exp(prediction_log[0])
        return render_template('index.html', prediction=f'Predicted Output: ${prediction_exp[0]:.2f}')
    except Exception as e:
        return render_template('index.html', prediction=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)
