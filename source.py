import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import pickle

df=sns.load_dataset('mpg')
df.isnull().sum()
df.dropna(inplace=True)
X= df[['displacement','horsepower','weight','acceleration']]
Y=df.mpg
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=42)
from ctypes import LibraryLoader
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train,y_train)
model.score(X_test,y_test)
from sklearn.tree import DecisionTreeRegressor
model2= DecisionTreeRegressor(criterion='poisson',random_state=0)
model2.fit(X_train,y_train)
model2.score(X_test,y_test)

filename='mpg_regression.sav'
pickle.dump(model, open(filename,'wb'))
