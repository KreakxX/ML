from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import  mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
import pandas as pd

housing = datasets.fetch_california_housing()

# Datenverarbeitung

# loading Data
dt = pd.DataFrame(data=housing.data, columns=housing.feature_names)
dt['target'] = housing['target']
dt.info()
dt.head()
dt = dt.drop_duplicates()   # duplicates removen
dt = dt.dropna()        # leere Zeilen removen


# Choosing Target and parameter for the model
X = dt[["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]]
y = dt['target']



# Model Setup RandomForrestClassifier, LinearRegression weil numerischen EingabeWerte
model = RandomForestRegressor( random_state=10, n_estimators=200) # n_estimators gleich Anzahl der bäume    # LinearRegression model nicht gut, weil zu viele Parameter also RandomForrestRegressor

# Data in Test und Training 80/20 split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2 , random_state=10)

# Training

model.fit(X_train, y_train)

# Evalutaten mit beispiel

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(mae)
print(r2)


# LinearRegression
# trys to match linear path between factors like if the size goes up the price also goes up (useCase when you have linear Connections)

# RandomForrestRegressor
# Decision Trees, like if the size is bigger than 100 m2 than go this way till the bottom -> take average like 100.000$
# not for linear connections esspecially more for compley connections between the attributs and not for linear stuff