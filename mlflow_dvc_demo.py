import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load DVC-tracked dataset
data = pd.read_csv("data/train.csv")

X = data[["feature1", "feature2"]]
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model parameters
n_estimators = 100
max_depth = 5

model = RandomForestClassifier(
    n_estimators=n_estimators,
    max_depth=max_depth,
    random_state=42
)

# Get DVC dataset version
with open("data/train.csv.dvc", "r") as f:
    dvc_file = f.read()

dvc_version = dvc_file.split("md5:")[1].split("\n")[0].strip()

# Create MLflow experiment
mlflow.set_experiment("MLflow_DVC_Experiment")

with mlflow.start_run():

    # Log DVC dataset version
    mlflow.log_param("dataset_version", dvc_version)

    # Log model parameters
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    # Train
    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)

    # Log accuracy
    mlflow.log_metric("accuracy", accuracy)

    # Log model
    mlflow.sklearn.log_model(
    model,
    "random_forest_model",
    serialization_format="pickle")
print("MLflow + DVC experiment completed!")
print("DVC dataset version:", dvc_version)
print("Accuracy:", accuracy)
