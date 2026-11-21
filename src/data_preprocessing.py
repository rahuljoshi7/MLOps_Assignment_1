
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def load_and_preprocess_data():
    # Load Iris dataset
    iris = load_iris()

    # Convert features into DataFrame
    X = pd.DataFrame(
        iris.data,
        columns=iris.feature_names
    )

    # Target variable
    y = pd.Series(iris.target, name="target")

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Dataset loaded successfully!")
    print("Total samples:", len(X))
    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))
    print("Number of features:", X.shape[1])

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    load_and_preprocess_data()
