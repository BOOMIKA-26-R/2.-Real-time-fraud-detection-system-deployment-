import pandas as pd
import numpy as np

# Simulate 1000 transactions
np.random.seed(42)
data = {
    'distance_from_home': np.random.exponential(scale=10, size=1000),
    'purchase_price_ratio': np.random.normal(loc=1.0, scale=0.5, size=1000),
    'is_online_order': np.random.choice([0, 1], size=1000),
    'used_pin_number': np.random.choice([0, 1], size=1000, p=[0.1, 0.9]),
}
# Fraud logic: High distance + High price ratio + Online usually = Fraud
data['is_fraud'] = ((data['distance_from_home'] > 50) & 
                    (data['purchase_price_ratio'] > 3.0) & 
                    (data['is_online_order'] == 1)).astype(int)

pd.DataFrame(data).to_csv('fraud_data.csv', index=False)
print("Fraud dataset created.")
