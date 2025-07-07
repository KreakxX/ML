
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import  mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

path = r"C:\Users\Henri\Videos\Studnets\Student_Performance.csv"
data = pd.read_csv(path)
print(data.columns)
print(data.head())

x = data[["Hours Studied","Previous Scores","Sleep Hours"]]
# x = data[["Hours Studied","Previous Scores","Sleep Hours","Sample Question Papers Practiced"]]
data["target"] = data["Performance Index"]
y = data["target"]


model = LinearRegression()
# Gradient Descent for best parameters

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2 , random_state=5)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(mae)
print(r2)