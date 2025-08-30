import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 1000  # Increased for better model training

# Generate synthetic data with strong correlations
data = {
    'pelvic_inlet_cm': np.random.uniform(10, 14, n_samples),  # Pelvic inlet diameter (cm)
    'pelvic_outlet_cm': np.random.uniform(8, 12, n_samples),  # Pelvic outlet diameter (cm)
    'fetal_head_cm': np.random.uniform(30, 36, n_samples),    # Fetal head circumference (cm)
    'fetal_weight_g': np.random.uniform(2500, 4500, n_samples),  # Fetal weight (grams)
    'maternal_age': np.random.randint(18, 45, n_samples),     # Maternal age (years)
    'parity': np.random.randint(0, 4, n_samples),             # Number of previous births
    'delivery_outcome': np.zeros(n_samples, dtype=int)        # Initialize as cesarean (0)
}

# Create DataFrame
df = pd.DataFrame(data)

# Strong deterministic logic for high AUC
for i in range(n_samples):
    inlet = df.loc[i, 'pelvic_inlet_cm']
    head = df.loc[i, 'fetal_head_cm']
    outlet = df.loc[i, 'pelvic_outlet_cm']
    weight = df.loc[i, 'fetal_weight_g']
    # Vaginal delivery (1) if inlet is large and head is small
    if inlet > 12.5 and head < 33.5 and outlet > 9.5 and 2800 < weight < 3800:
        df.loc[i, 'delivery_outcome'] = 1
    elif inlet > 12 and head < 34 and outlet > 9:  # Relaxed conditions
        df.loc[i, 'delivery_outcome'] = 1 if np.random.random() < 0.95 else 0
    else:
        df.loc[i, 'delivery_outcome'] = 1 if np.random.random() < 0.05 else 0

# Save to CSV
df.to_csv('pelvic_fetal_data.csv', index=False)
print("Enhanced dataset generated and saved as 'pelvic_fetal_data.csv'")