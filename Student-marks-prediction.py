import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

data = {
    "hours_studied": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "attendance": [50, 55, 60, 65, 70, 75, 80, 85, 90, 95],
    "marks": [35, 40, 50, 55, 60, 70, 75, 80, 90, 95]
}

df = pd.DataFrame(data)

X = df[["hours_studied", "attendance"]]
y = df["marks"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

sample = pd.DataFrame({
    "hours_studied": [6],
    "attendance": [80]
})

print("Predicted Marks:", model.predict(sample))
