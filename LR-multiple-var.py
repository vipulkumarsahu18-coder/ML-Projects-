import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

data = {
    "area": [1000, 1500, 1800, 2000, 2400, 3000, 3500, 4000],
    "bedrooms": [2, 3, 3, 3, 4, 4, 5, 5],
    "bathrooms": [1, 2, 2, 2, 3, 3, 4, 4],
    "age": [10, 5, 8, 3, 12, 7, 2, 1],
    "price": [3000000, 4500000, 5000000, 5200000, 6500000, 8000000, 9500000, 11000000]
}

df = pd.DataFrame(data)

X = df[["area", "bedrooms", "bathrooms", "age"]]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

sample = pd.DataFrame({
    "area": [2500],
    "bedrooms": [3],
    "bathrooms": [2],
    "age": [5]
})

print("Predicted Price:", model.predict(sample))
