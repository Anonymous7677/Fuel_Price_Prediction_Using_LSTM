from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from math import exp
from numpy import array
from numpy import average
from numpy import mean
from numpy import sqrt
from numpy import std
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from keras.models import load_model
from keras.models import clone_model
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

BASE_DIR = Path(__file__).resolve().parent


def get_dataset(ix, iy):
    file = str(BASE_DIR) + '/dataset/Final_Dataset.csv'

    dataset = pd.read_csv(file, index_col='Date')
    X = dataset[list(ix)].values
    y = dataset[list(iy)].values

    return X, y


def series_to_supervised(X, y, n_in=3):
    df = pd.DataFrame(X)
    cols = list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -3):
        cols.append(df.shift(i))
    agg = pd.concat(cols, axis=1)
    # drop rows with NaN values
    agg.dropna(inplace=True)

    # forecast sequence (t+3, ... t+n)
    y = y[n_in:]

    return agg.values, y


def split_dataset(dataset, split_size=0.1):
    '''
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=split_size, random_state=1, shuffle=False)

    return X_train, y_train, X_test, y_test
    '''

    split_train_val = len(dataset) - int(split_size * len(dataset) * 2)
    split_val_test = len(dataset) - int(split_size * len(dataset))

    train_data = dataset[:split_train_val]
    val_data = dataset[split_train_val:split_val_test]
    test_data = dataset[split_val_test:]

    return train_data, val_data, test_data


def rescale_X_dataset(X_train, X_val, X_test):
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_val_scaled, X_test_scaled


def split_sequences(X, y, n_steps):
    Xs, ys = [], []
    for i in range(len(X)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(X):
            break
        # gather input and output parts of the pattern
        Xs.append(X[i:end_ix])
        ys.append(y[end_ix-1])

    return array(Xs), array(ys)


def prepare_data(X, y, n_steps):
    # split data
    # X_train, y_train, X_test, y_test = split_dataset(X, y)
    X_train, X_val, X_test = split_dataset(X)
    y_train, y_val, y_test = split_dataset(y)
    # rescale input data
    X_train, X_val, X_test = rescale_X_dataset(X_train, X_val, X_test)
    # turn 1D data into time series data
    X_train, y_train = split_sequences(X_train, y_train, n_steps)
    X_val, y_val = split_sequences(X_val, y_val, n_steps)
    X_test, y_test = split_sequences(X_test, y_test, n_steps)

    return X_train, y_train, X_val, y_val, X_test, y_test


def measure_rmse(actual, predicted):
    return sqrt(mean_squared_error(actual, predicted))


def get_file(epoch_no, model_name):
    filename = str(BASE_DIR) + '/saved_model/' + model_name + \
        '_model_' + str(epoch_no) + '.h5'

    return filename


def get_model(config, n_inputs, n_outputs, loss='mse'):
    # unpack config
    n_steps, n_nodes, activ_func, lrates, _ = config

    model = Sequential()
    model.add(LSTM(n_nodes, input_shape=(n_steps, n_inputs),
              activation=activ_func))
    model.add(Dense(n_outputs))
    model.compile(loss=loss, optimizer=Adam(learning_rate=lrates))

    return model


def fit_model(model, X_train, y_train, X_val, y_val, config, model_name, evaluate_status=False, epochs=1, verbose=2):
    # unpack config
    _, _, _, _, n_epochs = config
    n_save_after = n_epochs - 10

    all_models_history = list()
    for i in range(n_epochs):
        # fit model
        if evaluate_status == True:
            model_history = model.fit(
                X_train, y_train, epochs=epochs, verbose=verbose)
        else:
            model_history = model.fit(
                X_train, y_train, epochs=epochs, verbose=verbose, validation_data=(X_val, y_val))
        all_models_history.append(model_history)
        # save the model after defined epoch numbers
        if i >= n_save_after:
            model.save(get_file(i, model_name))

    return all_models_history


def load_all_models(config, model_name):
    # unpack config
    _, _, _, _, n_epochs = config
    n_start = n_epochs - 10
    n_end = n_epochs

    all_models = list()
    for epoch in range(n_start, n_end):
        # get filename with current epoch
        filename = get_file(epoch, model_name)
        # load model from file
        model = load_model(filename)
        # add to list of members
        all_models.append(model)
        print('>loaded %s' % filename)

    return all_models


def model_weight_ensemble(members, weights, config, loss='mse'):
    # unpack config
    _, _, _, lrates, _ = config
    # determine how many layers need to be averaged
    n_layers = len(members[0].get_weights())

    # create a set of average model weights
    avg_model_weights = list()
    for layer in range(n_layers):
        # collec this layer from each model
        layer_weights = array([model.get_weights()[layer]
                              for model in members])
        # weighted average of weights for this layer
        avg_layer_weights = average(layer_weights, axis=0, weights=weights)
        # store average layer weights
        avg_model_weights.append(avg_layer_weights)

    # create a new model with the same structure
    model = clone_model(members[0])
    # set weights in the new
    model.set_weights(avg_model_weights)
    model.compile(loss=loss, optimizer=Adam(learning_rate=lrates))

    return model


def evaluate_n_members(members, n_members, X_data, y_data, config):
    # reverse loaded models so we build the ensemble with the last models first
    members = list(reversed(members))
    # select a subset of members
    subset = members[:n_members]
    # prepare an array of equal weights
    # option 1:
    # weights = [1.0 / n_members for _ in range(1, n_members+1)]
    # option 2:
    # weights = [i/n_members for i in range(n_members, 0, -1)]
    # option 3:
    alpha = 2.0
    weights = [exp(-i/alpha) for i in range(1, n_members+1)]
    # create a new model with the weighted average of all model weights
    model = model_weight_ensemble(subset, weights, config)
    # make predictions and evaluate error
    yhats = model.predict(X_data, verbose=2)
    error = measure_rmse(y_data, yhats)

    return yhats, error


def evaluate_standalone_ensemble_models(members, X_data, y_data, config):
    single_yhats, single_scores, ensemble_yhats, ensemble_scores = list(), list(), list(), list()
    for i in range(1, len(members)+1):
        # evaluate model with i members
        ensemble_yhat, ensemble_score = evaluate_n_members(
            members, i, X_data, y_data, config)

        # evaluate the i'th model standalone
        single_yhat = members[i-1].predict(X_data, verbose=2)
        single_score = measure_rmse(y_data, single_yhat)
        # summarize this step
        print(single_yhat, ensemble_yhat)
        print('> %d: single=%.3f, ensemble=%.3f' %
              (i, single_score, ensemble_score))
        ensemble_yhats.append(ensemble_yhat)
        ensemble_scores.append(ensemble_score)
        single_yhats.append(single_yhat)
        single_scores.append(single_score)

    return single_yhats, single_scores, ensemble_yhats, ensemble_scores


def repeat_evaluation(config, n_inputs, n_outputs, X_train, y_train, X_val, y_val, X_test, y_test, model_name, evaluate_status=False, n_repeats=30):
    # convert config to a key
    key = str(config)

    # fit and evaluate the model n times
    for i in range(n_repeats):
        print('Runtimes: %d' % (i+1))
        # remove '#' to define and fit the model
        # model = get_model(config, n_inputs, n_outputs)
        # history = fit_model(model, X_train, y_train, X_val, y_val, config, model_name, evaluate_status)
        members = load_all_models(config, model_name)
        print('Loaded %d models' % len(members))
        # evaluate different numbers of ensembles
        if evaluate_status == True:
            single_yhats, single_scores, ensemble_yhats, ensemble_scores = evaluate_standalone_ensemble_models(
                members, X_val, y_val, config)
        else:
            single_yhats, single_scores, ensemble_yhats, ensemble_scores = evaluate_standalone_ensemble_models(
                members, X_test, y_test, config)

    print('> Model[%s]' % key)

    return single_yhats, single_scores, ensemble_yhats, ensemble_scores
