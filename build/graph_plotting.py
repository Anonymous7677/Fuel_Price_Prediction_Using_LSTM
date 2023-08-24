import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

pd.set_option('display.expand_frame_repr', False)


def plot_loss_over_epochs(model_history):
    plt.plot([history.history['loss'] for history in model_history])
    plt.plot([history.history['val_loss'] for history in model_history])
    plt.title('Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validate'], loc='upper left')
    plt.show()


def plot_time_series(dataset):
    dataset['Date'] = pd.to_datetime(dataset['Date'])
    fig, axs = plt.subplots(7)
    for i in range(len(dataset.columns) - 1):
        axs[i-1].plot(dataset['Date'], dataset[dataset.columns[i+1]].values)
        axs[i-1].set_title(dataset.columns[i+1],
                           loc='right', pad=-10, fontsize=8)
    fig.savefig('time_series_stationarity.png')


'''
def get_dataset_full():
    dataset = pd.read_csv('dataset/Final_Dataset.csv', index_col='Date')
    pd.set_option('display.expand_frame_repr', False)
    corr = dataset.corr()

    b = X[X.columns[0]].values
    c = y[y.columns[0]].values
    covariance = cov(c, b)
    print(covariance)
    corr, _ = pearsonr(c, b)
    print('Pearsons correlation: %.3f' % corr)
    corr, _ = spearmanr(c, b)
    print('Spearmans correlation: %.3f' % corr)

    return corr
'''


# method = pearson, spearman
def plot_correlation_matrix(method):
    dataset = pd.read_csv('dataset/Final_Dataset.csv')
    corr = dataset.corr(method=method)

    figure = plt.figure(dpi=300)
    axes = figure.add_subplot(111)
    caxes = axes.matshow(corr)
    figure.colorbar(caxes)
    axes.set_xticklabels(dataset, fontsize=8, rotation=30)
    axes.set_yticklabels(dataset, fontsize=8)
    # plt.savefig('corr_' + method + '.png')
    # plt.show()

    return corr


def plot_correlation_matrix():
    dataset = pd.read_csv('dataset/Final_Dataset.csv', index_col='Date')

    plt.subplot(3, 4, 1)
    plt.scatter(dataset[dataset.columns[3]], dataset[dataset.columns[0]])
    plt.ylabel(dataset.columns[0])
    plt.subplot(3, 4, 2)
    plt.scatter(dataset[dataset.columns[4]], dataset[dataset.columns[0]])
    plt.subplot(3, 4, 3)
    plt.scatter(dataset[dataset.columns[5]], dataset[dataset.columns[0]])
    plt.subplot(3, 4, 4)
    plt.scatter(dataset[dataset.columns[6]], dataset[dataset.columns[0]])
    plt.subplot(3, 4, 5)
    plt.scatter(dataset[dataset.columns[3]], dataset[dataset.columns[1]])
    plt.ylabel(dataset.columns[1])
    plt.subplot(3, 4, 6)
    plt.scatter(dataset[dataset.columns[4]], dataset[dataset.columns[1]])
    plt.subplot(3, 4, 7)
    plt.scatter(dataset[dataset.columns[5]], dataset[dataset.columns[1]])
    plt.subplot(3, 4, 8)
    plt.scatter(dataset[dataset.columns[6]], dataset[dataset.columns[1]])
    plt.subplot(3, 4, 9)
    plt.scatter(dataset[dataset.columns[3]], dataset[dataset.columns[2]])
    plt.ylabel(dataset.columns[2])
    plt.xlabel(dataset.columns[3])
    plt.subplot(3, 4, 10)
    plt.scatter(dataset[dataset.columns[4]], dataset[dataset.columns[2]])
    plt.xlabel(dataset.columns[4])
    plt.subplot(3, 4, 11)
    plt.scatter(dataset[dataset.columns[5]], dataset[dataset.columns[2]])
    plt.xlabel(dataset.columns[5])
    plt.subplot(3, 4, 12)
    plt.scatter(dataset[dataset.columns[6]], dataset[dataset.columns[2]])
    plt.xlabel(dataset.columns[6])
    plt.savefig('scatter_plot.png')


# plot loss vs number of ensemble members and standalone
def plot_loss_over_members(members, single_scores, ensemble_scores):
    x_axis = [i for i in range(1, len(members)+1)]
    plt.plot(x_axis, single_scores, marker='o', linestyle='None')
    plt.plot(x_axis, ensemble_scores, marker='o')
    plt.show()


def plot_loss_over_features(model_history, lrates, i):
    plot_no = 420 + (i+1)
    plt.subplot(plot_no)
    plt.plot([history.history['loss']
             for history in model_history], label='train')
    plt.plot([history.history['val_loss']
             for history in model_history], label='validate')
    plt.title('lrate=' + str(lrates[i]), pad=-50)
