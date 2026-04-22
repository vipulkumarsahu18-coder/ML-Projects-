import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score

data = {
    "feature1": [1, 2, 3, 4, 5, 6, 7, 8],
    "feature2": [2, 1, 4, 3, 6, 5, 8, 7],
    "feature3": [5, 3, 6, 2, 8, 4, 9, 1],
    "class_label": [0, 0, 0, 1, 1, 1, 0, 1],
    "reg_target": [10, 15, 20, 25, 30, 35, 40, 45]
}

df = pd.DataFrame(data)

X = df[["feature1", "feature2", "feature3"]]
y_class = df["class_label"]
y_reg = df["reg_target"]

X_train, X_test, y_train_c, y_test_c = train_test_split(X, y_class, test_size=0.25, random_state=42)
_, _, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.25, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train_c)
pred_class = clf.predict(X_test)

print("Classification Accuracy:", accuracy_score(y_test_c, pred_class))

reg = RandomForestRegressor(n_estimators=100, random_state=42)
reg.fit(X_train, y_train_r)
pred_reg = reg.predict(X_test)

print("Regression R2 Score:", r2_score(y_test_r, pred_reg))

sample = pd.DataFrame([[3, 4, 5]], columns=["feature1", "feature2", "feature3"])

print("Class Prediction:", clf.predict(sample))
print("Regression Prediction:", reg.predict(sample))
