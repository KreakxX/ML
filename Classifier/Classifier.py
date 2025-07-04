from sklearn import datasets
import pandas as pd

housing = datasets.fetch_california_housing()

# loading Data
dt = pd.DataFrame(data=housing,columns=housing.feature_names)
dt['target'] = housing['target']
dt.info()
dt.head()
