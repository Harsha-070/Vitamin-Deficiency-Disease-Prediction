import os
import json
import joblib
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression


def load_data(path="vitamin_deficiency_disease_dataset_20260123.csv"):
    df = pd.read_csv(path)
    return df


def engineer_features(X):
    """Create derived features based on vitamin and health indicators"""
    X_eng = X.copy()
    
    # 1. Vitamin deficiency scores (how far below 100%)
    X_eng['vitamin_a_deficiency'] = np.maximum(0, 100 - X['vitamin_a_percent_rda'])
    X_eng['vitamin_d_deficiency'] = np.maximum(0, 100 - X['vitamin_d_percent_rda'])
    X_eng['vitamin_b12_deficiency'] = np.maximum(0, 100 - X['vitamin_b12_percent_rda'])
    X_eng['folate_deficiency'] = np.maximum(0, 100 - X['folate_percent_rda'])
    X_eng['vitamin_c_deficiency'] = np.maximum(0, 100 - X['vitamin_c_percent_rda'])
    X_eng['iron_deficiency'] = np.maximum(0, 100 - X['iron_percent_rda'])
    X_eng['calcium_deficiency'] = np.maximum(0, 100 - X['calcium_percent_rda'])
    
    # 2. Serum level critical ranges (most important biomarkers)
    X_eng['vitamin_d_critical'] = (X['serum_vitamin_d_ng_ml'] < 20).astype(int)
    X_eng['vitamin_b12_critical'] = (X['serum_vitamin_b12_pg_ml'] < 200).astype(int)
    X_eng['folate_critical'] = (X['serum_folate_ng_ml'] < 5).astype(int)
    
    # 3. Multiple deficiency risk
    deficiency_cols = ['vitamin_a_deficiency', 'vitamin_d_deficiency', 'vitamin_b12_deficiency', 
                       'folate_deficiency', 'vitamin_c_deficiency']
    X_eng['multiple_deficiency_risk'] = (X_eng[deficiency_cols] > 20).sum(axis=1)
    X_eng['total_deficiency_score'] = X_eng[deficiency_cols].sum(axis=1)
    
    # 4. Nutrient absorption pattern
    X_eng['nutrient_variability'] = X[['vitamin_a_percent_rda', 'vitamin_c_percent_rda', 
                                        'vitamin_d_percent_rda', 'vitamin_e_percent_rda', 
                                        'vitamin_b12_percent_rda', 'folate_percent_rda', 
                                        'calcium_percent_rda', 'iron_percent_rda']].std(axis=1)
    
    # 5. Hemoglobin risk (anemia indicator)
    X_eng['hemoglobin_low'] = (X['hemoglobin_g_dl'] < 12).astype(int)
    X_eng['hemoglobin_very_low'] = (X['hemoglobin_g_dl'] < 10).astype(int)
    
    # 6. Symptom severity score (important!)
    symptom_cols = ['has_night_blindness', 'has_fatigue', 'has_bleeding_gums', 'has_bone_pain',
                    'has_muscle_weakness', 'has_numbness_tingling', 'has_memory_problems', 'has_pale_skin']
    X_eng['total_symptoms'] = X[symptom_cols].sum(axis=1)
    X_eng['has_neurological_symptoms'] = (X['has_numbness_tingling'] | X['has_memory_problems']).astype(int)
    X_eng['has_bleeding_symptoms'] = X['has_bleeding_gums'].astype(int)
    X_eng['has_bone_muscle_symptoms'] = (X['has_bone_pain'] | X['has_muscle_weakness']).astype(int)
    
    # 7. Lifestyle risk factors
    X_eng['sedentary'] = (X['exercise_level'] == 'Sedentary').astype(int)
    X_eng['no_sun_exposure'] = (X['sun_exposure'] == 'Low').astype(int)
    X_eng['vegan_diet'] = (X['diet_type'] == 'Vegan').astype(int)
    X_eng['vegetarian_diet'] = (X['diet_type'] == 'Vegetarian').astype(int)
    X_eng['poor_absorption_risk'] = X_eng['sedentary'] + X_eng['vegan_diet'] + X_eng['no_sun_exposure']
    
    # 8. Age-related factors
    X_eng['age_senior'] = (X['age'] >= 70).astype(int)
    X_eng['age_young'] = (X['age'] < 30).astype(int)
    
    # 9. BMI risk
    X_eng['bmi_obese'] = (X['bmi'] > 30).astype(int)
    X_eng['bmi_underweight'] = (X['bmi'] < 18.5).astype(int)
    
    # 10. Socioeconomic factors
    X_eng['low_income'] = (X['income_level'] == 'Low').astype(int)
    
    # 11. Habit risk
    X_eng['smoking_current'] = (X['smoking_status'] == 'Current').astype(int)
    X_eng['heavy_alcohol'] = (X['alcohol_consumption'] == 'Heavy').astype(int)
    
    return X_eng


def build_enhanced_pipeline(X):
    """Build enhanced preprocessing without feature engineering (applied separately)"""
    
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    # Robust numeric preprocessing
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
    ])
    
    # Categorical preprocessing
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop='if_binary')),
    ])
    
    preprocessor = ColumnTransformer([
        ("num", num_pipeline, numeric_cols),
        ("cat", cat_pipeline, categorical_cols),
    ], remainder='drop')
    
    # Base learners (diverse)
    rf = RandomForestClassifier(
        n_estimators=400, max_depth=16, min_samples_split=4, min_samples_leaf=1,
        random_state=42, class_weight='balanced_subsample', n_jobs=-1, bootstrap=True
    )
    
    gb = GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.03, max_depth=8, min_samples_split=4,
        min_samples_leaf=1, random_state=42, subsample=0.7
    )
    
    xgb = XGBClassifier(
        n_estimators=300, learning_rate=0.03, max_depth=8, min_child_weight=1,
        subsample=0.7, colsample_bytree=0.7, random_state=42, n_jobs=-1, 
        eval_metric='mlogloss', verbosity=0
    )
    
    # Meta learner for stacking
    meta_learner = LogisticRegression(max_iter=500, random_state=42, n_jobs=-1)
    
    # Stacking ensemble
    stacking = StackingClassifier(
        estimators=[('rf', rf), ('gb', gb), ('xgb', xgb)],
        final_estimator=meta_learner,
        cv=5, n_jobs=-1
    )
    
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", stacking),
    ])
    
    return pipeline


def main():
    print("\n" + "=" * 70)
    print("  VITAMIN DEFICIENCY DISEASE PREDICTION - ADVANCED MODEL")
    print("=" * 70)
    
    print("\n[1/8] Loading dataset...")
    df = load_data()
    print(f"      ✓ Loaded {len(df)} records with {df.shape[1]} features")
    
    target = "disease_diagnosis"
    X = df.drop(columns=[target, "symptoms_list"], errors="ignore")
    y = df[target].astype(str)
    
    print(f"\n[2/8] Target variable analysis:")
    print(f"      ✓ Classes: {list(y.unique())}")
    class_dist = y.value_counts()
    for disease, count in class_dist.items():
        print(f"        - {disease}: {count} ({count/len(y)*100:.1f}%)")
    
    print(f"\n[3/8] Encoding labels...")
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    print(f"      ✓ {len(le.classes_)} unique diseases encoded")
    
    print(f"\n[4/8] Feature engineering...")
    X_eng = engineer_features(X)
    print(f"      ✓ Created {X_eng.shape[1]} engineered features (original: {X.shape[1]})")
    
    print(f"\n[5/8] Data stratification (80/20 split)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_eng, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )
    print(f"      ✓ Training set: {len(X_train)} samples")
    print(f"      ✓ Test set: {len(X_test)} samples")
    
    print(f"\n[6/8] Building stacking ensemble...")
    print("      ✓ Base models: RandomForest (400 trees), GradientBoosting (300), XGBoost (300)")
    print("      ✓ Meta-learner: Logistic Regression")
    pipeline = build_enhanced_pipeline(X_eng)
    
    print(f"\n[7/8] Training model (this may take 3-5 minutes)...")
    print("      Training in progress...", end="", flush=True)
    pipeline.fit(X_train, y_train)
    print(" ✓ Complete")
    
    print(f"\n[8/8] Evaluation & Results:")
    print("      " + "-" * 60)
    
    # Test set performance
    preds = pipeline.predict(X_test)
    
    acc = accuracy_score(y_test, preds)
    f1_macro = f1_score(y_test, preds, average='macro', zero_division=0)
    f1_weighted = f1_score(y_test, preds, average='weighted', zero_division=0)
    precision = precision_score(y_test, preds, average='macro', zero_division=0)
    recall = recall_score(y_test, preds, average='macro', zero_division=0)
    
    print(f"\n      📊 TEST SET METRICS:")
    print(f"      ├─ Accuracy:          {acc*100:6.2f}%")
    print(f"      ├─ F1-Score (macro):  {f1_macro:6.4f}")
    print(f"      ├─ F1-Score (weighted):{f1_weighted:6.4f}")
    print(f"      ├─ Precision:         {precision:6.4f}")
    print(f"      └─ Recall:            {recall:6.4f}")
    
    print(f"\n      📋 CLASSIFICATION REPORT:")
    print("      " + "-" * 60)
    report = classification_report(y_test, preds, target_names=le.classes_, zero_division=0)
    for line in report.split('\n'):
        print(f"      {line}")
    
    # Save model
    out_dir = os.path.join(os.getcwd(), "models")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "vitamin_pipeline.joblib")
    
    print(f"\n[SAVING] Model to {out_path}...")
    joblib.dump({
        "pipeline": pipeline,
        "label_encoder": le,
        "feature_names": X_eng.columns.tolist()
    }, out_path)
    print("      ✓ Model saved")
    
    # Save metrics
    metrics = {
        "accuracy": float(acc),
        "f1_score_macro": float(f1_macro),
        "f1_score_weighted": float(f1_weighted),
        "precision": float(precision),
        "recall": float(recall),
        "classes": le.classes_.tolist(),
        "model_type": "StackingClassifier(RF+GB+XGB → LogisticRegression)",
        "features_count": X_eng.shape[1],
        "training_samples": len(X_train),
        "test_samples": len(X_test)
    }
    
    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print("\n" + "=" * 70)
    print("  ✅ TRAINING COMPLETE - MODEL READY FOR PREDICTIONS")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
