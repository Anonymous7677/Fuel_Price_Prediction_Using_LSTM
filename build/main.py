from build.model_building import *
from build.data_collection import *
from build.data_preprocessing import *

'''
    This file is specific for running in app
'''

BASE_DIR = Path(__file__).resolve().parent

rawdata_filepath = str(BASE_DIR) + '/dataset/collection/'
preprocessed_filepath = str(BASE_DIR) + '/dataset/preprocessed/'
# define location of original dataset
brent_oridata = rawdata_filepath + 'RBRTEd.xls'
wti_oridata = rawdata_filepath + 'RWTCd.xls'
opec_oridata = rawdata_filepath + 'basketDayArchives.xml'
forex_oridata = rawdata_filepath + 'exchange-rates.csv'
fuel_oridata = rawdata_filepath + 'Fuel_Price.csv'
# define location of processed dataset
brent_newdata = preprocessed_filepath + 'Brent.csv'
wti_newdata = preprocessed_filepath + 'WTI.csv'
opec_newdata = preprocessed_filepath + 'OPEC.csv'
forex_newdata = preprocessed_filepath + 'Exchange_Rates.csv'
fuel_newdata = preprocessed_filepath + 'Fuel_Price.csv'
merged_file = preprocessed_filepath + 'Weekly_Average_Price.csv'
# define location of finalized dataset
final_dataset = str(BASE_DIR) + '/dataset/Final_Dataset.csv'
history = str(BASE_DIR) + '/pred_history.csv'

# obtain download links for datasets from websites
FUEL_URL = 'https://www.comparehero.my/transportation/articles/latest-petrol-price-ron95-ron97-diesel'
BRENT_URL = 'https://www.eia.gov/dnav/pet/hist_xls/RBRTEd.xls'
WTI_URL = 'https://www.eia.gov/dnav/pet/hist_xls/RWTCd.xls'
OPEC_URL = 'https://www.opec.org/basket/basketDayArchives.xml'


RON95_DIESEL_MODEL = "95D"
RON97_MODEL = "win2_97"


def get_latest_dataset():
    _ = get_fuel_dataset(FUEL_URL, fuel_oridata)
    _ = get_oil_dataset(BRENT_URL, brent_oridata)
    _ = get_oil_dataset(WTI_URL, wti_oridata)
    _ = get_oil_dataset(OPEC_URL, opec_oridata)
    _ = get_forex_dataset(forex_oridata)

    return None


def update_unified_dataset():
    # convert opec xml file to csv
    _ = convert_xml_to_csv(opec_oridata, opec_newdata)
    # convert all the dataset with daily price to weekly average
    files = [[fuel_oridata, fuel_newdata],
             [brent_oridata, brent_newdata],
             [wti_oridata, wti_newdata],
             [opec_newdata, opec_newdata]]
    for i in range(len(files)):
        convert_daily_to_weekly_average(files[i][0], files[i][1])
    # combine all the datasets into one single dataset
    _ = merge_files(fuel_newdata, wti_newdata, brent_newdata,
                    opec_newdata, forex_oridata, merged_file)
    # convert crude oils with barrel and usd unit to litre and myr unit
    _ = convert_oil_price_to_myr(merged_file, final_dataset)

    return None


def predict_ron95_diesel():
    #
    config = [1, 8, 'relu', 0.1, 50]
    #
    ix = ['RON95/litre', 'Diesel/litre']
    iy = ['RON95/litre', 'Diesel/litre']
    X, y = get_dataset(ix, iy)
    # get necessary column data
    X, y = get_dataset(ix, iy)
    # frame variable into sequence (t-n, ... t-1), (t+3, ...t+n)
    X, y = series_to_supervised(X, y)
    # train-val-test split, scaling to range(0, 1), turn data into multisteps
    n_steps = config[0]
    _, _, _, _, X_test, y_test = prepare_data(X, y, n_steps)
    X_test = X_test[-3:]
    # Load saved trained model
    members = load_all_models(config, RON95_DIESEL_MODEL)
    print('Loaded %d models' % len(members))
    ensemble_yhats = list()
    for i in range(1, len(members)+1):
        members = list(reversed(members))
        # select a subset of members
        subset = members[:i]
        alpha = 2.0
        weights = [exp(-i/alpha) for i in range(1, i+1)]
        # create a new model with the weighted average of all model weights
        model = model_weight_ensemble(subset, weights, config)
        # make predictions and evaluate error
        yhats = model.predict(X_test, verbose=0)
        ensemble_yhats.append(yhats)

    prediction = mean(ensemble_yhats, axis=0).round(2)
    pred_95, pred_diesel = list(), list()
    for i in range(len(prediction)):
        pred_95.append(prediction[i][0])
        pred_diesel.append(prediction[i][1])

    return pred_95, pred_diesel


def predict_ron97():
    #
    config = [2, 8, 'relu', 0.01, 100]
    #
    ix = ['RON97/litre', 'OPEC/litre', 'MYR/USD']
    iy = ['RON97/litre']
    X, y = get_dataset(ix, iy)
    # get necessary column data
    X, y = get_dataset(ix, iy)

    n_steps = config[0]
    _, _, _, _, X_test, y_test = prepare_data(X, y, n_steps)
    X_test = X_test[-3:]
    print(X_test)
    # Load saved trained model
    members = load_all_models(config, RON97_MODEL)
    print('Loaded %d models' % len(members))
    ensemble_yhats = list()
    for i in range(1, len(members)+1):
        members = list(reversed(members))
        # select a subset of members
        subset = members[:i]
        # prepare an array of equal weights
        alpha = 2.0
        weights = [exp(-i/alpha) for i in range(1, i+1)]
        # create a new model with the weighted average of all model weights
        model = model_weight_ensemble(subset, weights, config)
        # make predictions and evaluate error
        yhats = model.predict(X_test, verbose=0)
        ensemble_yhats.append(yhats)

    prediction = mean(ensemble_yhats, axis=0).round(2)
    prediction = prediction.reshape(-1)

    return prediction


def save_predictions(r95, r97, diesel):
    data = pd.read_csv(final_dataset)
    data['Date'] = pd.to_datetime(data['Date'])
    lastdate = data['Date'].iloc[-1]
    lastdate = pd.date_range(lastdate, periods=4, freq='W')
    lastdate = lastdate.strftime('%Y-%m-%d')
    newdate = lastdate[1:].values

    newdata = {
        'Date': newdate,
        'RON95/litre: Prediction': r95,
        'RON97/litre: Prediction': r97,
        'Diesel/litre: Prediction': diesel,
    }

    df = pd.DataFrame(newdata)
    # append data frame to CSV file
    df2 = pd.read_csv(history)
    a = check_date(df2, newdate[-1])

    if (a == False):
        df.to_csv(history, mode='a', index=False, header=False)
    # with open(history, 'r+') as hist: if newdate[-1] not in hist:

    return newdate


def check_date(df, date):
    return date in df['Date'].values
