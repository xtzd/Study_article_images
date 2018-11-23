import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.externals import joblib

# Choose GBDT Regression model as baseline
# my_model = GradientBoostingRegressor()


# Training Step
def my_train_func(station):
    global my_model
    train_data = pd.read_csv('train-dataset/point_date_' + station + '.csv')
    train_data_Y = train_data['actualPowerGeneration']

    # Drop some non-relative factors
    drop_columns = ['longitude', 'latitude', 'RadiationHorizontalPlane', 'Temperature', 'actualPowerGeneration',
                    'Humidity', 'atmosphericPressure', 'windDirection', 'scatteredRadiation']
    train_data_X = train_data.drop(axis=1, columns=drop_columns)

    train_data_X['month'] = pd.to_datetime(train_data_X.Time).dt.month
    train_data_X['day'] = pd.to_datetime(train_data_X.Time).dt.day
    train_data_X['hour'] = pd.to_datetime(train_data_X.Time).dt.hour
    train_data_X = train_data_X.drop(axis=1, columns=['Time'])

    # Validation
    X_train, X_test, Y_train, Y_test = train_test_split(train_data_X, train_data_Y, test_size=0.2, random_state=40)

    myGBR = GradientBoostingRegressor()
    myGBR.fit(X_train, Y_train)
    Y_pred = myGBR.predict(X_test)

    # Output model to global variation
    # my_model = myGBR
    _ = joblib.dump(myGBR, 'model/model.pkl', compress=9)




    print('Training completed. MSE on validation set is {}'.format(mean_squared_error(Y_test, Y_pred)))
    print('Factors below are used: \n{}'.format(list(X_train.columns)))


def my_spredict_func(input_file, output_file):

    # Clean test data
    columns = 'Time,longitude,latitude,directRadiation,scatterdRadiation,windSpeed,airTransparency,airDensity'
    columns = list(columns.split(','))
    test_data = pd.read_csv('test-dataset/' + input_file, names=columns)
    drop_columns = ['longitude', 'latitude', 'airTransparency', 'airDensity']
    test_data = test_data.drop(axis=1, columns=drop_columns)

    test_data['month'] = pd.to_datetime(test_data.Time).dt.month
    test_data['day'] = pd.to_datetime(test_data.Time).dt.day
    test_data['hour'] = pd.to_datetime(test_data.Time).dt.hour
    test_data['min'] = pd.to_datetime(test_data.Time).dt.minute

    # Find the time point we need to start with

    test_data = test_data.sort_values(by='Time')

    # Find the latest time point
    time_point = test_data[test_data['hour'] == 0][test_data['min'] == 0].index.tolist()[0]

    test_data = test_data.drop(axis=1, columns=['Time', 'min'])

    # Simply fill the NaN values, need more discussion

    test_data = test_data.fillna(method='ffill')
    test_data = test_data.fillna(0)

    test_data = test_data.iloc[time_point - 48:time_point + 47]

    my_model = joblib.load('model/model.pkl')

    result = my_model.predict(test_data)
    print('Short prediction of power generation in nearest 96 time points:\n{}'.format(result))
    np.savetxt('output/' + output_file, result)




def my_sspredict_func(input_file, output_file):
    columns = 'Time,longitude,latitude,directRadiation,scatterdRadiation,windSpeed,airTransparency,airDensity'
    columns = list(columns.split(','))
    test_data = pd.read_csv('test-dataset/' + input_file, names=columns)
    drop_columns = ['longitude', 'latitude', 'airTransparency', 'airDensity']
    test_data = test_data.drop(axis=1, columns=drop_columns)

    test_data['month'] = pd.to_datetime(test_data.Time).dt.month
    test_data['day'] = pd.to_datetime(test_data.Time).dt.day
    test_data['hour'] = pd.to_datetime(test_data.Time).dt.hour

    test_data = test_data.sort_values(by='Time')

    test_data = test_data.drop(axis=1, columns=['Time'])

    test_data = test_data.fillna(method='ffill')
    test_data = test_data.fillna(0)

    test_data = test_data[:16]

    my_model = joblib.load('model/model.pkl')

    result = my_model.predict(test_data)
    print('Super short prediction of power generation in nearest 16 time points:\n{}'.format(result))
    np.savetxt('output/' + output_file, result)


