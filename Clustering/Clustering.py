from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

np.random.seed(42)
n_customers = 1000


data = pd.DataFrame({
    'purchase_frequency': np.random.poisson(8, n_customers),  # amount of orders per year
    'avg_order_value': np.random.gamma(2, 50, n_customers),   # average order value
    'days_since_last_purchase': np.random.exponential(30, n_customers),  # days till last order
    'total_spent': np.random.gamma(3, 200, n_customers)       # all expenses
})

scaler = StandardScaler()   # neccessary that every feature has the same weight like income, moneyspend
data_scaled = scaler.fit_transform(data) # scale every data

kmeans = KMeans(n_clusters=3, random_state=42)  # n clusters is for the amount of groups in this example group in the 3 groups
clusters = kmeans.fit_predict(data)

data['cluster'] = clusters
print(data)
