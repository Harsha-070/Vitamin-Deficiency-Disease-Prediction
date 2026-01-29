import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

MODEL_PATH = os.path.join("models", "vitamin_pipeline.joblib")

st.set_page_config(page_title="Vitamin Deficiency Predictor", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        max-width: 1200px;
        margin: 0 auto;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


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


st.title("🏥 Vitamin Deficiency & Disease Prediction")
st.markdown("Predict diseases based on health parameters, symptoms, and vitamin levels")

if not os.path.exists(MODEL_PATH):
    st.error("❌ Model not found. Run training first: `python src/train_model_v2.py`")
else:
    saved = joblib.load(MODEL_PATH)
    pipeline = saved["pipeline"]
    le = saved["label_encoder"]
    
    # Load dataset for reference
    try:
        df_ref = pd.read_csv("vitamin_deficiency_disease_dataset_20260123.csv")
        feature_cols = df_ref.drop(columns=["disease_diagnosis", "symptoms_list"], errors="ignore").columns.tolist()
    except:
        feature_cols = []
    
    tab1, tab2, tab3 = st.tabs(["📋 Manual Input", "📤 Batch Upload", "📊 Dataset Sample"])
    
    # ===== TAB 1: Manual Input =====
    with tab1:
        st.markdown("### Enter Patient Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Demographics**")
            age = st.slider("Age (years)", 18, 90, 45)
            gender = st.selectbox("Gender", ["Male", "Female"])
            bmi = st.slider("BMI", 15.0, 40.0, 25.0, step=0.1)
            
            st.markdown("**Lifestyle**")
            smoking_status = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
            alcohol_consumption = st.selectbox("Alcohol Consumption", ["None", "Light", "Moderate", "Heavy"])
            exercise_level = st.selectbox("Exercise Level", ["Sedentary", "Light", "Moderate", "Active"])
            
        with col2:
            st.markdown("**Diet & Environment**")
            diet_type = st.selectbox("Diet Type", ["Omnivore", "Vegetarian", "Pescatarian", "Vegan"])
            sun_exposure = st.selectbox("Sun Exposure", ["Low", "Moderate", "High"])
            income_level = st.selectbox("Income Level", ["Low", "Middle", "High"])
            latitude_region = st.selectbox("Latitude Region", ["Low", "Mid", "High"])
        
        # Vitamin levels
        st.markdown("### Vitamin & Nutrient Levels (% RDA)")
        col3, col4, col5, col6 = st.columns(4)
        
        with col3:
            vitamin_a = st.number_input("Vitamin A %", 10.0, 200.0, 100.0)
            vitamin_c = st.number_input("Vitamin C %", 10.0, 250.0, 100.0)
            vitamin_d = st.number_input("Vitamin D %", 5.0, 250.0, 100.0)
            vitamin_e = st.number_input("Vitamin E %", 10.0, 220.0, 100.0)
        
        with col4:
            vitamin_b12 = st.number_input("Vitamin B12 %", 10.0, 860.0, 100.0)
            folate = st.number_input("Folate %", 10.0, 200.0, 100.0)
            calcium = st.number_input("Calcium %", 10.0, 210.0, 100.0)
            iron = st.number_input("Iron %", 10.0, 200.0, 100.0)
        
        # Blood test values
        st.markdown("### Blood Test Values")
        col7, col8, col9, col10 = st.columns(4)
        
        with col7:
            hemoglobin = st.number_input("Hemoglobin (g/dL)", 9.0, 18.0, 14.0, step=0.1)
            vitamin_d_serum = st.number_input("Serum Vitamin D (ng/mL)", 5.0, 100.0, 30.0)
        
        with col8:
            vitamin_b12_serum = st.number_input("Serum B12 (pg/mL)", 100.0, 900.0, 400.0)
            folate_serum = st.number_input("Serum Folate (ng/mL)", 2.0, 30.0, 10.0)
        
        # Symptoms
        st.markdown("### Symptoms (Select all that apply)")
        col11, col12, col13, col14 = st.columns(4)
        
        with col11:
            has_night_blindness = st.checkbox("Night Blindness")
            has_fatigue = st.checkbox("Fatigue")
        
        with col12:
            has_bleeding_gums = st.checkbox("Bleeding Gums")
            has_bone_pain = st.checkbox("Bone Pain")
        
        with col13:
            has_muscle_weakness = st.checkbox("Muscle Weakness")
            has_numbness_tingling = st.checkbox("Numbness/Tingling")
        
        with col14:
            has_memory_problems = st.checkbox("Memory Problems")
            has_pale_skin = st.checkbox("Pale Skin")
        
        symptoms_count = sum([has_night_blindness, has_fatigue, has_bleeding_gums, has_bone_pain,
                             has_muscle_weakness, has_numbness_tingling, has_memory_problems, has_pale_skin])
        
        # Predict button
        if st.button("🔍 Predict Disease", type="primary"):
            input_data = pd.DataFrame({
                'age': [age],
                'gender': [gender],
                'bmi': [bmi],
                'smoking_status': [smoking_status],
                'alcohol_consumption': [alcohol_consumption],
                'exercise_level': [exercise_level],
                'diet_type': [diet_type],
                'sun_exposure': [sun_exposure],
                'income_level': [income_level],
                'latitude_region': [latitude_region],
                'vitamin_a_percent_rda': [vitamin_a],
                'vitamin_c_percent_rda': [vitamin_c],
                'vitamin_d_percent_rda': [vitamin_d],
                'vitamin_e_percent_rda': [vitamin_e],
                'vitamin_b12_percent_rda': [vitamin_b12],
                'folate_percent_rda': [folate],
                'calcium_percent_rda': [calcium],
                'iron_percent_rda': [iron],
                'hemoglobin_g_dl': [hemoglobin],
                'serum_vitamin_d_ng_ml': [vitamin_d_serum],
                'serum_vitamin_b12_pg_ml': [vitamin_b12_serum],
                'serum_folate_ng_ml': [folate_serum],
                'symptoms_count': [symptoms_count],
                'has_night_blindness': [int(has_night_blindness)],
                'has_fatigue': [int(has_fatigue)],
                'has_bleeding_gums': [int(has_bleeding_gums)],
                'has_bone_pain': [int(has_bone_pain)],
                'has_muscle_weakness': [int(has_muscle_weakness)],
                'has_numbness_tingling': [int(has_numbness_tingling)],
                'has_memory_problems': [int(has_memory_problems)],
                'has_pale_skin': [int(has_pale_skin)],
                'has_multiple_deficiencies': [int(symptoms_count >= 2)],
            })
            
            try:
                # Apply feature engineering
                input_eng = engineer_features(input_data)
                
                # Make prediction
                pred_class = pipeline.predict(input_eng)[0]
                pred_disease = le.inverse_transform([pred_class])[0]
                proba = pipeline.predict_proba(input_eng)[0]
                confidence = max(proba) * 100
                
                st.markdown("---")
                st.markdown("### 🎯 Prediction Result")
                col_result1, col_result2 = st.columns(2)
                
                with col_result1:
                    st.metric("Predicted Disease", pred_disease)
                
                with col_result2:
                    st.metric("Confidence", f"{confidence:.1f}%")
                
                st.markdown("**Disease Probabilities:**")
                prob_df = pd.DataFrame({
                    'Disease': le.classes_,
                    'Probability': proba
                }).sort_values('Probability', ascending=False)
                st.bar_chart(prob_df.set_index('Disease'))
                
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
    
    # ===== TAB 2: Batch Upload =====
    with tab2:
        st.markdown("### Upload CSV for Batch Predictions")
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
        
        if uploaded_file is not None:
            df_upload = pd.read_csv(uploaded_file)
            st.write("**Uploaded data preview:**")
            st.dataframe(df_upload.head())
            
            if st.button("🔍 Predict All", type="primary"):
                try:
                    # Drop target columns if present
                    df_input = df_upload.drop(columns=["disease_diagnosis", "symptoms_list"], errors="ignore")
                    
                    # Apply feature engineering
                    df_eng = engineer_features(df_input)
                    
                    preds = pipeline.predict(df_eng)
                    pred_diseases = le.inverse_transform(preds)
                    
                    df_results = df_upload.copy()
                    df_results["predicted_disease"] = pred_diseases
                    
                    st.success(f"✅ Predictions completed for {len(df_results)} rows")
                    st.dataframe(df_results)
                    
                    # Download results
                    csv = df_results.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results (CSV)",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv"
                    )
                except Exception as e:
                    st.error(f"Batch prediction error: {str(e)}")
    
    # ===== TAB 3: Dataset Sample =====
    with tab3:
        st.markdown("### Sample Data from Training Set")
        sample_size = st.slider("Number of samples to display", 1, len(df_ref), 5)
        
        sample_df = df_ref.sample(n=sample_size, random_state=42)
        st.dataframe(sample_df)
        
        # Statistics
        st.markdown("### Dataset Statistics")
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        
        with col_stat1:
            st.metric("Total Records", len(df_ref))
        
        with col_stat2:
            st.metric("Disease Classes", len(df_ref['disease_diagnosis'].unique()))
        
        with col_stat3:
            st.metric("Features", len(df_ref.columns) - 2)
        
        st.markdown("**Disease Distribution:**")
        disease_counts = df_ref['disease_diagnosis'].value_counts()
        st.bar_chart(disease_counts)
