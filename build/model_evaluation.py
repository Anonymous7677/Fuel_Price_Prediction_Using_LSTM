from numpy import mean
from numpy import std
from model_building import *
import matplotlib.pyplot as plt


'''
Parameters to be evaluated and validated:
- RON97:
    1. input data = [[RON97, RM/USD], [RON97, 0PEC, RM/USD], [RON97, WTI, OPEC, RM/USD]
- RON95, Diesel:
    1. input data = [[Diesel, RON95]]
- All three:
    2. n_steps = [1, 2, 3]
    2. n_neurons = [2, 4, 6, 8, 10]
    5. lrates = [1E-1, 1E-2, 1E-3]
    6. n_epochs = [50, 100, 150]
    7. activ_func['relu', 'sigmoid']
'''


# FOR RON95 MODEL
# define model filename

RON95_DIESEL_MODEL = "95D"
# define list of hyperparameters
# config = [n_steps, n_neurons, activ_func, learning rates, n_epochs]
config = [1, 8, 'relu', 0.1, 50]
ix = ['RON95/litre', 'Diesel/litre']
iy = ['RON95/litre', 'Diesel/litre']
'''
# FOR RON97 MODEL
# define model filename
RON97_MODEL = "win2_97"
# define list of hyperparameters
# config = [n_steps, n_neurons, activ_func, learning rates, n_epochs]
config = [2, 8, 'relu', 0.01, 100]
ix = ['RON97/litre', 'OPEC/litre', 'MYR/USD']
iy = ['RON97/litre']
'''
# get necessary column data
X, y = get_dataset(ix, iy)
# frame variable into sequence (t-n, ... t-1), (t+3, ...t+n)
X, y = series_to_supervised(X, y)
# train-val-test split, scaling to range(0, 1), turn data into multisteps
n_steps = config[0]
X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(X, y, n_steps)
n_inputs, n_outputs = X_train.shape[2], y_train.shape[1]
print(y_test.shape[0])
# define, fit and evaluate model for n times
single_yhats, single_scores, ensemble_yhats, ensemble_scores = repeat_evaluation(
    config, n_inputs, n_outputs, X_train, y_train, X_val, y_val, X_test, y_test, model_name=RON95_DIESEL_MODEL, n_repeats=1)
# calculate mean and standard deviation of the rmse on all runs
m_single_error = mean(single_scores)
std_single_error = std(single_scores)
m_ensemble_error = mean(ensemble_scores)
std_ensemble_error = std(ensemble_scores)
m_ensemble_yhats = mean(ensemble_yhats, axis=0).round(2)

print('Total RMSE for single model is %.3f(%.3f)' %
      (m_single_error, std_single_error))
print('Total RMSE for ensemble model is %.3f(%.3f)' %
      (m_ensemble_error, std_ensemble_error))
for i in range(len(y_test)):
    print('RON 95: Actual= %.2f, Prediction= %.2f (%.2f)' %
          (y_test[i][0], m_ensemble_yhats[i][0], (y_test[i][0] - m_ensemble_yhats[i][0])))
    print('Diesel: Actual= %.2f, Prediction= %.2f (%.2f)' %
          (y_test[i][1], m_ensemble_yhats[i][1], (y_test[i][1] - m_ensemble_yhats[i][1])))
