import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import os

# Create dummy data
df = pd.DataFrame({
    'feature1': [1, 2, 3, 4],
    'feature2': [2, 4, 6, 8],
    'label': [0, 0, 1, 1]
})

X = df[['feature1', 'feature2']]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)

os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/model.pkl")