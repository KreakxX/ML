import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import pandas as pd

sp500 =  yf.Ticker("^GSPC")

sp500 = sp500.history(period="max")

print(sp500)

index = sp500.index
print(index)

sp500.plot.line(y="Close", use_index=True) # for Graph but this is the wrong development area

del sp500["Dividends"] # deleting cols we dont need
del sp500["Stock Splits"]


sp500["Tomorrow"] = sp500["Close"].shift(-1)    # The Close Price of tommorrow

sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)  # check if the Price tommorrow is greater than the price of today


sp500 = sp500.loc["1990-01-01":].copy() # only take the cols that are at least after 1990-01-01 because old data is not as useful as it seems

rf = RandomForestClassifier(n_estimators=100, min_samples_split=100,random_state=1)  # min_samples_split Good for overfitting and n_estimators higher == better but till the limit is reached

train = sp500.iloc[:-100] # all except last 100 rows are train
test = sp500.iloc[-100:] # and only the last 100 rows are test

predictors = ["Close", "Volume", "Open", "High", "Low"]

rf.fit(train[predictors],train["Target"])

preds = rf.predict(test[predictors])
preds = pd.Series(preds, index=test.index)
print(preds)

precision = precision_score(test["Target"],preds)
print(precision)  # 66% of the time we were correct



# new Predictors

horizons = [2,5,60,250,1000]

new_predictors = []

for horizon in horizons :
    rolling_averages = sp500.rolling(horizon).mean()
    ration_col = f"Close_Ratio_{horizon}"

    sp500[ration_col] = sp500["Close"] / rolling_averages["Close"]

    trend_col = f"Trend_{horizon}"
    sp500[trend_col]= sp500.shift(1).rolling(horizon).sum()["Target"]
    new_predictors += [ration_col, trend_col]



    print(sp500.head())
    sp500 = sp500.dropna()

    train = sp500.iloc[:-100]
    test = sp500.iloc[-100:]

    model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)

    def predict(train, test, predictors, model):
        model.fit(train[predictors], train["Target"])
        preds = model.predict_proba(test[predictors])[:,1]
        preds[preds >= .6] = 1 # if the model is about 60% sure it will go about, then it returns 1 for price will go up else 0
        preds[preds < .6] = 0
        return precision_score(test["Target"], preds)

    print(predict(train,test,new_predictors,model))