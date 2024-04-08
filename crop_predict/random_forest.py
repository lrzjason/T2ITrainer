# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Assuming 'data' is a pandas DataFrame with the required features and target variable
# Features: image_center_x, image_center_y, feature_center_x, feature_center_y, aesthetic_score, image_width, image_height
# Target variable: cropping_method

# Load your dataset into a pandas DataFrame
data = pd.read_csv('your_dataset.csv')  # Uncomment and modify with your dataset path

# Split the data into features and target variable
X = data[['image_center_x', 'image_center_y', 'feature_center_x', 'feature_center_y', 'aesthetic_score', 'image_width', 'image_height']]
y = data['cropping_method']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model to the training data
rf.fit(X_train, y_train)

# Predict the cropping method on the test set
y_pred = rf.predict(X_test)

# Evaluate the model using Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"The Mean Squared Error of the Random Forest model is: {mse:.2f}")

# Note: The above code assumes that 'cropping_method' is a numerical value.
# If 'cropping_method' is categorical, you will need to use RandomForestClassifier instead of RandomForestRegressor
# and also encode the 'cropping_method' column using techniques like Label Encoding or One Hot Encoding.
