import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score

# 1. Load the dataset
df = pd.read_csv('diabetic_data.csv')

# 2. Replace '?' with NaN
df = df.replace('?', np.nan)

# 3. Drop irrelevant columns
drop_cols = ['encounter_id', 'patient_nbr', 'weight', 'payer_code', 'medical_specialty']
df = df.drop(columns=drop_cols, errors='ignore')

# 4. Simplify the target variable
df['readmitted'] = df['readmitted'].apply(lambda x: 1 if x in ['<30', '>30'] else 0)

# 5. Convert specific string categories to numbers (like >200, <30)
# We will use a simple mapping for these medical categories
df['max_glu_serum'] = df['max_glu_serum'].replace({'>200': 200, '>300': 300, 'Norm': 100, 'NaN': 0})
df['A1Cresult'] = df['A1Cresult'].replace({'>7': 7, '>8': 8, 'Norm': 5, 'NaN': 0})

# 6. Handle Categorical variables with Label Encoding
le = LabelEncoder()
# Get all columns that are of type 'object' (string) and encode them
categorical_cols = df.select_dtypes(include=['object']).columns

for col in categorical_cols:
    df[col] = df[col].astype(str) 
    df[col] = le.fit_transform(df[col])

# 7. Split data into features (X) and target (y)
X = df.drop(columns=['readmitted'])
y = df['readmitted']

# 8. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 9. Apply SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("--- Data Shape before SMOTE ---")
print(f"Train: {y_train.value_counts().to_dict()}")
print("--- Data Shape after SMOTE ---")
print(f"Train: {y_train_resampled.value_counts().to_dict()}")

# 10. Train XGBoost Model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X_train_resampled, y_train_resampled)

# 11. Evaluate on the test set
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 12. Plot Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Not Readmitted (0)', 'Readmitted (1)'],
            yticklabels=['Not Readmitted (0)', 'Readmitted (1)'])

plt.title('Confusion Matrix - XGBoost Model')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
# 13. Plot Feature Importance
plt.figure(figsize=(10, 8))

# Get feature importances from the XGBoost model
importances = model.feature_importances_
features = X.columns

# Sort them in descending order
indices = np.argsort(importances)[::-1]
top_n = 15  # Show top 15 features

plt.title('Top 15 Feature Importances - XGBoost Model', fontsize=14)
plt.barh(range(top_n), importances[indices[:top_n]], color='teal', align='center')
plt.yticks(range(top_n), [features[i] for i in indices[:top_n]])
plt.xlabel('Relative Importance')
plt.gca().invert_yaxis()  # Invert y-axis to have the most important at the top
plt.tight_layout()
plt.show()