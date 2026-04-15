import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv('fraud_data.csv')
X = df.drop('is_fraud', axis=1)
y = df['is_fraud']

# Fraud is rare, so we split carefully
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

model = RandomForestClassifier(n_estimators=100, class_weight='balanced')
model.fit(X_train, y_train)

joblib.dump(model, 'fraud_model.pkl')
print(f"Fraud model saved. Accuracy: {model.score(X_test, y_test):.2f}")
