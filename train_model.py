import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

DATA_PATH = "data/telecom_churn.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "best_churn_model.pkl")

FEATURE_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "InternetService",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]

TARGET_COLUMN = "Churn"  


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}. Please place telecom_churn.csv in data/ folder.")
    df = pd.read_csv(path)

    missing_cols = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"These required columns are missing in your dataset: {missing_cols}")

    df = df[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    return df


def preprocess_and_split(df: pd.DataFrame):
    y = df[TARGET_COLUMN].map({"No": 0, "Yes": 1})
    if y.isna().any():
        raise ValueError("Target column 'Churn' must contain only 'Yes' and 'No' values.")

    X = df[FEATURE_COLUMNS]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    numeric_features = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
    categorical_features = [
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "InternetService",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
    ]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return X_train, X_test, y_train, y_test, preprocessor


def build_models(preprocessor: ColumnTransformer):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42
        ),
    }

    pipelines = {}
    for name, clf in models.items():
        pipelines[name] = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", clf),
            ]
        )
    return pipelines


def evaluate_models(pipelines, X_train, y_train, X_test, y_test):
    results = []

    for name, pipe in pipelines.items():
        print(f"\nTraining {name}...")
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        results.append({
            "model": name,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "pipeline": pipe,
        })

        print(f"{name} - "
              f"Accuracy: {acc:.4f}, "
              f"Precision: {prec:.4f}, "
              f"Recall: {rec:.4f}, "
              f"F1: {f1:.4f}")

    return results


def plot_model_comparison(results, output_path="models/model_comparison.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    model_names = [r["model"] for r in results]
    f1_scores = [r["f1"] for r in results]

    plt.figure(figsize=(6, 4))
    plt.bar(model_names, f1_scores)
    plt.ylabel("F1 Score")
    plt.title("Model Comparison (F1 Score)")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved model comparison plot to {output_path}")


def save_best_model(results):
    best = max(results, key=lambda r: r["f1"])
    best_name = best["model"]
    best_f1 = best["f1"]
    best_pipeline = best["pipeline"]

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_pipeline, MODEL_PATH)
    print(f"\nBest model: {best_name} (F1 = {best_f1:.4f})")
    print(f"Saved to {MODEL_PATH}")


def main():
    print("Loading data...")
    df = load_data(DATA_PATH)

    print("Splitting and preprocessing...")
    X_train, X_test, y_train, y_test, preprocessor = preprocess_and_split(df)

    print("Building models...")
    pipelines = build_models(preprocessor)

    print("Training and evaluating models...")
    results = evaluate_models(pipelines, X_train, y_train, X_test, y_test)

    print("\nPlotting model comparison...")
    plot_model_comparison(results)

    print("Saving best model...")
    save_best_model(results)

    print("\nDone!")


if __name__ == "__main__":
    main()
