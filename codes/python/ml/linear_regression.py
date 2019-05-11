import sys

from sklearn.datasets import load_boston
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

boston = load_boston()
datax = boston.data
datay = boston.target

print(datax.shape)

trainx, testx, trainy, testy = train_test_split(datax, datay, test_size=0.3, random_state=1)

lr = LinearRegression()
lr.fit(trainx, trainy)
lr_coef = lr.coef_
print(type(lr_coef))
print(lr_coef.shape)
print(f'lr coef is: {lr_coef}')
sys.exit()

rd = Ridge(alpha=10)
rd.fit(trainx, trainy)
rd_coef = rd.coef_
print(f'rd coef is: {rd_coef}')

ls = Lasso()
ls.fit(trainx, trainy)
ls_coef = ls.coef_
print(f'ls coef is: {ls_coef}')

en = ElasticNet()
en.fit(trainx, trainy)
en_coef = en.coef_
print(f'en coef is: {en_coef}')
