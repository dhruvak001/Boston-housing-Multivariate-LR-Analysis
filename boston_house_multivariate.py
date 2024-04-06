import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the Boston Housing dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"
header = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df = pd.read_csv(url, delim_whitespace=True, names=header)

data_boston=df

# (a) Checking for any missing values in the dataset and handle them appropriately.
missing_values_boston = data_boston.isnull().sum()
print("Missing values in the Boston Housing dataset:")
print(missing_values_boston)

columns_with_missing_values = ['CRIM', 'ZN', 'INDUS', 'CHAS','AGE', 'LSTAT']
# Impute missing values with mean
for column in columns_with_missing_values:
    data_boston[column].fillna(data_boston[column].mean(), inplace=True)

# (b) Normalize the 'RM' and 'MEDV' columns using min-max normalization

min_max_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
data_boston[['RM', 'MEDV']] = data_boston[['RM', 'MEDV']].apply(min_max_scaler)
# Min-max normalization for 'CRIM', 'NOX', 'PTRATIO', 'DIS'
columns_to_normalize = ['CRIM', 'NOX', 'PTRATIO', 'DIS']
data_boston[columns_to_normalize] = data_boston[columns_to_normalize].apply(min_max_scaler)

# Display the normalized data
display(data_boston)
display(data_boston.isnull().sum())

# Split the dataset into training and testing sets (80-20 split)
X = data_boston[['RM']]
y = data_boston['MEDV']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Multivariate Linear Regression Implementation
def hypothesis(theta, X):
    return X.dot(theta)

def mean_squared_error(predictions, targets):
    return (((predictions - targets) ** 2)/1).mean()

def gradient_descent(theta, X, y, learning_rate, epochs):
    for _ in range(epochs):
        predictions = hypothesis(theta, X)
        error = predictions - y
        theta = theta - learning_rate * X.T.dot(error) / len(y)
    return theta

# Initialize parameters and perform Gradient Descent
theta_initial = np.zeros(X_train.shape[1] + 1)
X_train = np.c_[np.ones(X_train.shape[0]), X_train]  # Add a column of ones for the bias term
theta_final = gradient_descent(theta_initial, X_train, y_train, learning_rate=0.01, epochs=1000)

# Plot the regression line on the scatter plot (using two features for visualization)
plt.scatter(X_test['RM'], y_test, label='Actual', alpha=0.7)
plt.scatter(X_test['RM'], hypothesis(theta_final, np.c_[np.ones(X_test.shape[0]), X_test]), label='Predicted', alpha=0.7)
plt.title('Regression Line for RM')
plt.xlabel('RM')
plt.ylabel('MEDV')
plt.legend()
plt.show()

# Evaluation on the test set
X_test = np.c_[np.ones(X_test.shape[0]), X_test]  # Add a column of ones for the bias term
test_predictions = hypothesis(theta_final, X_test)

# Compute Mean Squared Error and Absolute Error
mse = mean_squared_error(test_predictions, y_test)
absolute_error = (abs(test_predictions - y_test).mean())

print("\nMean Squared Error on Test Set:", mse)
print("Mean Absolute Error on Test Set:", absolute_error)
