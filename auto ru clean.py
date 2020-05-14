import pandas as pd
import numpy as np
from sklearn import preprocessing, model_selection
import sklearn
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error


data = pd.read_csv('new_data_99_06_03_13_04.csv')

data = data.drop(['Таможня', 'description'], axis='columns', inplace=False)
le = preprocessing.LabelEncoder()


categorical_columns = data.columns[data.dtypes == 'object']

for column in categorical_columns:
    data[column] = le.fit_transform(list(data[column]))

data = data.dropna()


predict = 'Price'

X = np.array(data.drop([predict], 1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, shuffle=1)

model = CatBoostRegressor(learning_rate=0.5)
model.fit(x_train, y_train)
accuracy = model.score(x_test, y_test)
print(accuracy)

predictions_2 = model.predict(x_test)
mae_2 = mean_absolute_error(predictions_2, y_test)
print("Mean Absolute Error:", mae_2)