from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import  mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier 
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# trend predicten
sp500 =  yf.Ticker("^GSPC")
sp500 = sp500.history(period="max")
sp500 = sp500.loc["2010-01-01":]    # ab 2010 machen

#Index(['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits'], dtype='object')  -> cols

print(sp500.head()) # spalten top level anzeigen
print(sp500.index)  # gets the index of each thing
print(sp500.columns) # for info

sp500["Tomorrow"] = sp500["Close"].shift(-1) # shifting the position in the array to get next day
sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)   # and check if tommorow is higher or not return 0 or 1
sp500["Close"]       # todays price
sp500["Volume"]      # volume today   
sp500["Daily Change"] = sp500["Close"] - sp500["Open"]    # how the Close minuse the Open is Changed the diffences 
sp500["Return 1d"] = sp500["Close"].pct_change()    # rendite    # new cols etc 

sp500 = sp500.dropna()     # drop all NaN values
sp500 = sp500.drop_duplicates() # drop all duplicate

up = sp500[sp500["Target"] == 1]  # get all the up values
down = sp500[sp500["Target"] == 0]   # get all the down values 
balanced = pd.concat([up.sample(len(down)), down]).sample(frac=1)  # get a balance of both like 50 samples of down and 50 samples of up

train = balanced.iloc[:int(0.8 * len(balanced))]  # 80 % training
test = balanced.iloc[int(0.8 * len(balanced)):]  # 20 % testing

features = ["Close", "Volume", "Daily Change", "Return 1d"] # features for leveling

model = RandomForestClassifier(n_estimators=100) # basic RandomForrestClassifier
model.fit(train[features],train["Target"]) # training

sp500["Predicted"] = model.predict(sp500[features])  # making the predicted value
sp500["Correct"] = sp500["Predicted"] == sp500["Target"]   # correct when Predicted and Target matched
sp500["Correct"].astype(int).rolling(100).mean().plot(title="Prediction Accuracy (Rolling)")

# visualize it
plt.show()

# to be continued Regression model gets input like the trend and predicts the "exact" price 