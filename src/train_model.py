import os
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score


def load_data(path="vitamin_deficiency_disease_dataset_20260123.csv"):
    df = pd.read_csv(path)
    return df


def build_pipeline(X):
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=[object]).columns.tolist()

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, numeric_cols),
        ("cat", cat_pipeline, categorical_cols),
    ])

    clf = RandomForestClassifier(
        n_estimators=200, random_state=42, class_weight="balanced_subsample", n_jobs=-1
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", clf),
    ])

    return pipeline


def main():
    print("Loading data...")
    df = load_data()

    # target and features
    target = "disease_diagnosis"
    if target not in df.columns:
        raise RuntimeError(f"Target column '{target}' not found in CSV")

    X = df.drop(columns=[target, "symptoms_list"], errors="ignore")
    y = df[target].astype(str)

    print("Preparing label encoder...")
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    print("Building pipeline...")
    pipeline = build_pipeline(X)

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    print("Cross-validation scores (accuracy)...")
    try:
        scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1)
        print(f"CV accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
    except Exception:
        print("CV failed (likely due to categorical handling on small folds). Skipping CV.")

    print("Training final model...")
    pipeline.fit(X_train, y_train)

    print("Evaluating on test set...")
    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, target_names=le.classes_, zero_division=0)
    print(f"Test accuracy: {acc:.4f}\n")
    print(report)

    # save model
    out_dir = os.path.join(os.getcwd(), "models")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "vitamin_pipeline.joblib")

    joblib.dump({"pipeline": pipeline, "label_encoder": le}, out_path)
    print(f"Saved pipeline and label encoder to {out_path}")

    # save metrics summary
    metrics = {"test_accuracy": float(acc), "classes": le.classes_.tolist()}
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import joblib
from sklearn.metrics import classification_report, accuracy_score


DATA_PATH = "vitamin_deficiency_disease_dataset_20260123.csv"
OUT_DIR = "models"
os.makedirs(OUT_DIR, exist_ok=True)


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    return df


def prepare_features(df):
    df = df.copy()
    if "symptoms_list" in df.columns:
        df = df.drop(columns=["symptoms_list"])  # free-text

    target = "disease_diagnosis"
    y = df[target].astype(str).values
    X = df.drop(columns=[target])

    # Define feature groups (use columns present)
    categorical = [
        c
        for c in [
            "gender",
            "smoking_status",
            "alcohol_consumption",
            "exercise_level",
            "diet_type",
            "sun_exposure",
            "income_level",
            "latitude_region",
        ]
        if c in X.columns
    ]

    # numeric: choose obvious numeric columns
    numeric = [
        c
        for c in X.columns
        if c not in categorical and X[c].dtype.kind in "bifcu"
    ]

    # Imputers and transformers
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric),
            ("cat", categorical_transformer, categorical),
        ]
    )

    return X, y, preprocessor


def train():
    df = load_data()
    X, y_raw, preprocessor = prepare_features(df)

    # encode target
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("clf", clf)])

    print("Training model...")
    pipeline.fit(X_train, y_train)

    # evaluation
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0)

    print(f"Test accuracy: {acc:.4f}")
    print(report)

    # cross val
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
    print(f"5-fold CV accuracy: {scores.mean():.4f} +/- {scores.std():.4f}")

    # save pipeline + label encoder + metadata
    model_path = os.path.join(OUT_DIR, "vitamin_pipeline.joblib")
    joblib.dump({"pipeline": pipeline, "label_encoder": le}, model_path)
    print(f"Saved model to {model_path}")

    meta = {
        "test_accuracy": float(acc),
        "cv_mean_accuracy": float(scores.mean()),
        "cv_std_accuracy": float(scores.std()),
        "classes": le.classes_.tolist(),
    }
    with open(os.path.join(OUT_DIR, "training_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    train()
