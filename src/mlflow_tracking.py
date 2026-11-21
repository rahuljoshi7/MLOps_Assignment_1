
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from data_preprocessing import load_and_preprocess_data


# Step 1: Load and preprocess the Iris dataset
X_train, X_test, y_train, y_test = load_and_preprocess_data()


# Step 2: Set MLflow experiment
mlflow.set_experiment("Iris_RandomForest_Experiment")


# Step 3: Run two Random Forest experiments
for n_trees in [50, 100]:

    with mlflow.start_run(
        run_name=f"RandomForest_{n_trees}_Trees"
    ):

        # Initialize model
        model = RandomForestClassifier(
            n_estimators=n_trees,
            random_state=42
        )

        # Train model
        model.fit(X_train, y_train)

        # Make predictions
        predictions = model.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, predictions)

        # Log model parameters
        mlflow.log_param("n_estimators", n_trees)
        mlflow.log_param("random_state", 42)

        # Log model evaluation metric
        mlflow.log_metric("accuracy", accuracy)

        # Log trained model to MLflow
        mlflow.sklearn.log_model(
            sk_model=model,
            name="random_forest_model",
            skops_trusted_types=[
                "sklearn.tree._tree.Tree"
            ]
        )

        # Display results
        print("--------------------------------")
        print(f"Experiment: RandomForest_{n_trees}_Trees")
        print(f"Number of Trees: {n_trees}")
        print(f"Accuracy: {accuracy:.4f}")
        print("Model logged successfully!")
        print("--------------------------------")


print("\nAll MLflow experiments completed successfully!")
