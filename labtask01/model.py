import pandas as pd
import numpy as np
import csv
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

# Load data
train = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# Detect id and target columns
id_col = next((c for c in ["PassengerId", "Id", "ID"] if c in test_data.columns), None)
target_col = next((c for c in ["SalePrice", "Price", "Target"] if c in train.columns), None)
if target_col is None:
    target_col = train.columns[-1]  
    print(f"Warning: target column not found by name; using '{target_col}' as target.")

y_train = train[target_col]
# Build feature DataFrames 
drop_cols = [c for c in ["PassengerId", "Id", "Ticket", "Name"] if c in train.columns]
X_train = train.drop(columns=drop_cols + [target_col], errors="ignore")
X_test = test_data.drop(columns=[c for c in ["Ticket", "Name"] if c in test_data.columns], errors="ignore")
# ensure id_col not dropped from test_data
if id_col in X_test.columns and id_col in drop_cols:
    X_test = test_data.drop(columns=[c for c in ["Ticket", "Name"] if c in test_data.columns], errors="ignore")

# Basic feature engineering
for df in (X_train, X_test):
    if "Cabin" in df.columns:
        df["CabinDeck"] = df["Cabin"].astype(str).str[0].replace("n", "Missing")
        df.drop(columns=["Cabin"], inplace=True, errors="ignore")
    if set(["SibSp", "Parch"]).issubset(df.columns):
        df["FamilySize"] = df["SibSp"].fillna(0) + df["Parch"].fillna(0) + 1

# Separate numeric and categorical
num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

# Impute numeric with median and categorical with mode
num_imputer = SimpleImputer(strategy="median")
X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
X_test[num_cols] = num_imputer.transform(X_test[num_cols])

for col in cat_cols:
    mode = X_train[col].mode().iloc[0] if not X_train[col].mode().empty else ""
    X_train[col].fillna(mode, inplace=True)
    if col in X_test.columns:
        X_test[col].fillna(mode, inplace=True)

#Encode
X_train = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
X_test = pd.get_dummies(X_test, columns=[c for c in cat_cols if c in X_test.columns], drop_first=True)

# A train and test
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# Model 
rf = RandomForestRegressor(random_state=42)
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 8, 16]
}
grid = GridSearchCV(rf, param_grid, cv=5, scoring="neg_mean_squared_error", n_jobs=-1)
grid.fit(X_train, y_train)
best = grid.best_estimator_

# RMSE
cv_scores = cross_val_score(best, X_train, y_train, cv=5, scoring="neg_mean_squared_error", n_jobs=-1)
rmse = np.sqrt(-cv_scores).mean()
print(f"CV RMSE: {rmse:.4f}")

# Fit and predict
best.fit(X_train, y_train)
preds = best.predict(X_test)

# Evaluate model
is_binary = set(y_train.dropna().unique()).issubset({0, 1})
if is_binary:
    acc_cv = cross_val_score(best, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1)
    print(f"CV Accuracy: {acc_cv.mean():.4f}")
    best.fit(X_train, y_train)
    train_pred = best.predict(X_train)
    print(f"Train Accuracy: {accuracy_score(y_train, train_pred):.4f}")
else:
    neg_mse_cv = cross_val_score(best, X_train, y_train, cv=5, scoring="neg_mean_squared_error", n_jobs=-1)
    rmse_cv = np.sqrt(-neg_mse_cv).mean()
    r2_cv = cross_val_score(best, X_train, y_train, cv=5, scoring="r2", n_jobs=-1).mean()
    print(f"CV RMSE: {rmse_cv:.4f}")
    print(f"CV R2: {r2_cv:.4f}")
    best.fit(X_train, y_train)
    train_pred = best.predict(X_train)
    print(f"Train RMSE: {np.sqrt(mean_squared_error(y_train, train_pred)):.4f}")

# Prepare submission
if id_col is None:
    raise ValueError("Could not find an ID column in test data (PassengerId or Id). Add one and rerun.")
submission_id = test_data[id_col].values
with open("submission.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([id_col, target_col])
    for pid, p in zip(submission_id, preds):
        writer.writerow([pid, float(p)])
print("Saved submission.csv")

