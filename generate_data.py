import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 500

# Generate synthetic data
data = {
    'pelvic_inlet_cm': np.random.uniform(10, 14, n_samples),  # Pelvic inlet diameter (cm)
    'pelvic_outlet_cm': np.random.uniform(8, 12, n_samples),  # Pelvic outlet diameter (cm)
    'fetal_head_cm': np.random.uniform(30, 36, n_samples),    # Fetal head circumference (cm)
    'fetal_weight_g': np.random.uniform(2500, 4500, n_samples),  # Fetal weight (grams)
    'maternal_age': np.random.randint(18, 45, n_samples),     # Maternal age (years)
    'parity': np.random.randint(0, 4, n_samples),             # Number of previous births
    # Simulated outcome: 1 for vaginal delivery, 0 for cesarean
    'delivery_outcome': np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
}

# Create DataFrame
df = pd.DataFrame(data)

# Add simple logic for realism: cesarean more likely if pelvic inlet is small or fetal head is large
for i in range(n_samples):
    if df.loc[i, 'pelvic_inlet_cm'] < 11 or df.loc[i, 'fetal_head_cm'] > 35:
        df.loc[i, 'delivery_outcome'] = 0 if np.random.random() < 0.6 else 1

# Save to CSV
df.to_csv('pelvic_fetal_data.csv', index=False)
print("Dataset generated and saved as 'pelvic_fetal_data.csv'")