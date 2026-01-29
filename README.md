# Vitamin Deficiency / Disease Prediction

Quick start:

1. Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Train the model (saves to `models/vitamin_pipeline.joblib`):

```bash
python src/train_model.py
```

3. Run the Streamlit app:

```bash
streamlit run app.py
```

Files added:
- `src/train_model.py` — training and model export
- `models/vitamin_pipeline.joblib` — created after training
- `app.py` — Streamlit UI (existing)
# Vitamin Deficiency Disease Prediction

This project trains a Random Forest classifier to predict vitamin-deficiency-related diseases from a patient dataset and exposes a simple Streamlit app for inference.

Quick start

1. Create a Python environment and install requirements:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Train the model:

```bash
python src/train_model.py
```

3. Run the Streamlit app:

```bash
streamlit run streamlit_app.py
```

Model and metadata are saved to the `models/` folder.
