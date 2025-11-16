import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from pathlib import Path


def evaluate_models(X_train, X_test, y_train, y_test, X_full, y_full, model_dir, prefix=None):
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=200, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', n_jobs=-1),
        "SVC": SVC(probability=True),
        "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500)
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

        if prefix is not None:
            final_model = models[name]
            final_model.fit(X_full, y_full)

            model_path = model_dir / f"{prefix}_{name}.joblib"
            joblib.dump(final_model, model_path)
            print(f"Saved FULL model in {model_path}")

        acc = accuracy_score(y_test, preds)
        f1_macro = f1_score(y_test, preds, average="macro")
        f1_weighted = f1_score(y_test, preds, average="weighted")
        # top3 = top_k_accuracy_score(y_test, probs, k=3) if probs is not None else None

        results[name] = {
            "accuracy": acc,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            # "top3_accuracy": top3
        }
        print(f"{name} done: acc={acc:.3f}, f1_macro={f1_macro:.3f}") # , top3={top3:.3f if top3 else 0.0}

    return results