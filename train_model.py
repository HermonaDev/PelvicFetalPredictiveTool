import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import joblib

# Load the dataset
df = pd.read_csv('pelvic_fetal_data.csv')

# Define features and target
X = df[['pelvic_inlet_cm', 'pelvic_outlet_cm', 'fetal_head_cm', 'fetal_weight_g', 'maternal_age', 'parity']]
y = df['delivery_outcome']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize XGBoost model with tuned parameters
model = XGBClassifier(
    n_estimators=500,           # More trees for better learning
    learning_rate=0.01,         # Slower learning for stability
    max_depth=7,                # Deeper trees for complex patterns
    random_state=42,
    eval_metric='logloss'
)

# Train model (no early stopping to avoid API issues)
model.fit(X_train, y_train)

# Evaluate model
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of vaginal delivery (class 1)
auc = roc_auc_score(y_test, y_pred_proba)
print(f"AUC-ROC Score: {auc:.3f}")

# Save the model
joblib.dump(model, 'xgb_model.pkl')
print("Model saved as 'xgb_model.pkl'")