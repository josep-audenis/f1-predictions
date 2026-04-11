import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import json
from datetime import datetime

def train_production_models():
   
    print("=" * 70)
    print("PRODUCTION MODEL TRAINING (100% DATA)")
    print("=" * 70)
    print()
    
    data_dir = Path("data/processed")
    features_file = data_dir / f"features_pre_race_2018-{datetime.now().year}.csv"
    
    if not features_file.exists():
        print(f"Features file not found: {features_file}")
        print("\nRun feature generation first:")
        print("  python -m src.features.build_predictive_features")
        return
    
    print(f"Loading features from: {features_file}")
    df = pd.read_csv(features_file)
    print(f"Loaded {len(df)} rows")
    print()
    
    for use_quali in [True, False]:
        model_type = "quali" if use_quali else "pre-quali"
        print(f"\n{'='*70}")
        print(f"Training {model_type.upper()} models with 100% data")
        print(f"{'='*70}\n")
        
        df_model = df.copy()
        
        df_model["target"] = (df_model["Position"] == 1).astype(int)
        
        df_model = df_model.fillna(df_model.mean(numeric_only=True))
        
        cat_cols = ["TeamName", "GrandPrix", "Driver"]
        for col in cat_cols:
            df_model[col] = LabelEncoder().fit_transform(df_model[col].astype(str))
        
        X = df_model.drop(columns=["Position", "target"])
        y = df_model["target"]
        
        if not use_quali:
            quali_features = [
                "grid_position", "is_top10_start", "grid_vs_team_avg",
                "driver_avg_quali_last5", "team_avg_quali_last5"
            ]
            X = X.drop(columns=quali_features, errors="ignore")
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        models = {
            "LogisticRegression": LogisticRegression(
                max_iter=1000, 
                class_weight='balanced',
                random_state=42
            ),
            "RandomForest": RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            "GradientBoosting": GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42
            )
        }
        
        model_dir = Path("models/top1") / model_type
        model_dir.mkdir(parents=True, exist_ok=True)
                
        for name, model in models.items():
            print(f"Training {name} on 100% of data ({len(X)} samples)...")
            
            model.fit(X_scaled, y)
            
            model_file = model_dir / f"{name}_model.pkl"
            joblib.dump(model, model_file)

    
    print("\n" + "=" * 70)
    print("PRODUCTION MODELS TRAINED SUCCESSFULLY")
    print("=" * 70)
    print("\nAll models trained on 100% of available data")
    print("Models saved in: models/top1/")
    print("  - quali/")
    print("  - pre-quali/")

if __name__ == "__main__":
    train_production_models()