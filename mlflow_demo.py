import mlflow
import mlflow.sklearn

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load Iris dataset
X, y = datasets.load_iris(return_X_y=True)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define model parameters
params = {
    "solver": "lbfgs",
    "max_iter": 1000
}

# Create the model
model = LogisticRegression(**params)

# Create MLflow experiment
mlflow.set_experiment("MLflow_Quickstart")

# Enable automatic logging
mlflow.sklearn.autolog()

# Train the model
with mlflow.start_run():
    model.fit(X_train, y_train)

print("Model and metrics logged automatically!")
