import pandas as pd

data = {
    "Name": ["A", "B", "C", "D"],
    "City": ["Delhi", "Mumbai", "Delhi", "Chennai"],
    "Gender": ["Male", "Female", "Female", "Male"]
}

df = pd.DataFrame(data)

dummies = pd.get_dummies(df, columns=["City", "Gender"])

print(dummies)
