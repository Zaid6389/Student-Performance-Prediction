import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, accuracy_score

# ----------------------
# DATASET
# ----------------------
data = {
    "hours": [1,2,3,4,5,6,7,8],
    "marks": [10,20,30,40,50,60,70,80],
    "result": [0,0,0,0,1,1,1,1]
}

df = pd.DataFrame(data)

# ----------------------
# LINEAR REGRESSION
X = df[["hours"]]
y = df["marks"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

lr = LinearRegression()
lr.fit(X_train, y_train)

print("Predicted Marks for 6 hours:", lr.predict([[6]]))

# Graph
plt.scatter(df["hours"], df["marks"])
plt.plot(df["hours"], lr.predict(X))
plt.show()

# Error
print("MAE:", mean_absolute_error(y_test, lr.predict(X_test)))

# LOGISTIC REGRESSION

y_class = df["result"]

X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.2)

log = LogisticRegression()
log.fit(X_train, y_train)

print("Pass/Fail Prediction:", log.predict([[6]]))
print("Probability:", log.predict_proba([[6]]))

# Accuracy
print("Accuracy:", accuracy_score(y_test, log.predict(X_test)))
