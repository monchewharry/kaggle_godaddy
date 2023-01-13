import numpy as np
import pandas as pd

# score functions


def smape(y_true, y_pred):
    smap = np.zeros(len(y_true))
    num = np.abs(y_true - y_pred)
    dem = ((np.abs(y_true) + np.abs(y_pred)) / 2)

    pos_ind = (y_true != 0) | (y_pred != 0)

    smap[pos_ind] = num[pos_ind] / dem[pos_ind]
    return 100 * np.mean(smap)


# define the vsmape function which calculates and returns the
# element-wise SMAPE values of y_true and y_pred in an array
def vsmape(y_true, y_pred):
    # create an array to store the SMAPE values
    smap = np.zeros(len(y_true))

    num = np.abs(y_true - y_pred)
    dem = ((np.abs(y_true) + np.abs(y_pred)) / 2)

    pos_ind = (y_true != 0) | (y_pred != 0)
    smap[pos_ind] = num[pos_ind] / dem[pos_ind]

    return 100 * smap

# data preprocessing
# preprocessing


def get_rawdata(train, test):
    "concat train and test with labels, add dcount,county_id,state_id"
    train['istest'] = 0
    test['istest'] = 1
    raw = pd.concat((train, test)).sort_values(['cfips', 'row_id']).reset_index(drop=True)

    raw['first_day_of_month'] = pd.to_datetime(raw["first_day_of_month"])
    raw['county'] = raw.groupby('cfips')['county'].ffill()
    raw['state'] = raw.groupby('cfips')['state'].ffill()
    raw["year"] = raw["first_day_of_month"].dt.year

    raw["month"] = raw["first_day_of_month"].dt.month

    # date ids in each county
    raw["dcount"] = raw.groupby(['cfips'])['row_id'].cumcount()

    raw['county_i'] = (raw['county'] + raw['state']).factorize()[0]

    # encode the object as an enumerated type or categorical variable
    raw['state_i'] = raw['state'].factorize()[0]
    return raw


def replace_outliers(raw, threshold=0.2):
    outliers = []
    cnt = 0

    for o in raw.cfips.unique():

        indices = (raw['cfips'] == o)
        tmp = raw.loc[indices].copy().reset_index(drop=True)
        var = tmp.microbusiness_density.values.copy()

        # loop through the values in reverse order, starting from index 37 and ending at index 2
        for i in range(37, 2, -1):
            # calculate a threshold value as 20% of the mean of the first i values
            thr = threshold * np.mean(var[:i])
            # calculate the absolute difference between the current and previous values
            difa = abs(var[i] - var[i - 1])

            # if the difference is greater than or equal to the threshold
            if (difa >= thr):
                var[:i] *= (var[i] / var[i - 1])
                outliers.append(o)
                cnt += 1

        # set the first value to be 99% of the second value
        var[0] = var[1] * 0.99

        raw.loc[indices, 'microbusiness_density'] = var

    outliers = np.unique(outliers)
    print(f"number of unique cfips with outliers:{len(outliers)}, total number of outliers:{cnt}")
    return raw


def abs_dif(raw):
    "create abs dif column"
    lag = 1

    # Create a new column in the raw dataframe that is the microbusiness density value of the previous month
    # (shifted by lag number of periods) for each cfips. Use backfill (bfill) to fill in the first value for
    # each cfips with the value from the second row.
    raw[f'mbd_lag_{lag}'] = raw.groupby('cfips')['microbusiness_density'].shift(lag).bfill()

    # Create a new column in the raw dataframe that is the difference between the current month's
    # microbusiness density and the previous month's microbusiness density, as a percentage.
    # Fill missing values with 1 and clip values between 0 and infinity.
    raw['dif'] = (raw['microbusiness_density'] / raw[f'mbd_lag_{lag}']).fillna(1).clip(0, None) - 1

    # Replace all 0 values in the new 'dif' column with 0.
    raw.loc[(raw[f'mbd_lag_{lag}'] == 0), 'dif'] = 0

    # Replace all values in the new 'dif' column that are positive and have a corresponding value of 0 in
    # the previous month's 'microbusiness_density' column with 1.
    raw.loc[(raw[f'microbusiness_density'] > 0) & (raw[f'mbd_lag_{lag}'] == 0), 'dif'] = 1

    # Take the absolute value of all values in the new 'dif' column.
    raw['dif'] = raw['dif'].abs()
    return raw


# convert target


def new_target(raw):
    "convert target variable"
    # same as raw.microbusiness_density.pct_change(1)
    raw['target'] = raw.groupby('cfips')['microbusiness_density'].shift(-1)
    raw['target'] = raw['target'] / raw['microbusiness_density'] - 1

    # Set the target value for rows with cfips value of 28055 to 0
    raw.loc[raw['cfips'] == 28055, 'target'] = 0.0

    # Set the target value for rows with cfips value of 48269 to 0
    raw.loc[raw['cfips'] == 48269, 'target'] = 0.0
    return raw


def lastactive(raw):
    "Create a new column in the dataframe called 'lastactive' at dcount 28, for within trainset train/valid split."

    raw['lastactive'] = raw.groupby('cfips')['active'].transform('last')

    # Select rows from the dataframe where the 'dcount' column is equal to 28 == 2021-12-1.
    # and group the resulting dataframe by the 'cfips' column
    # Apply the 'last()' function to the 'microbusiness_density' column and store the resulting Series in 'dt'
    dt = raw.loc[raw.dcount == 28].groupby('cfips')['microbusiness_density'].agg('last')

    # Create a new column in the dataframe called 'lasttarget' by mapping the 'cfips' column to the 'dt' Series
    raw['lasttarget'] = raw['cfips'].map(dt)

    return raw


# feature engineering


def build_features(raw, target='microbusiness_density', target_act='active_tmp', lags=6):
    "feature engineering"
    feats = []
    for lag in range(1, lags):
        raw[f'mbd_lag_{lag}'] = raw.groupby('cfips')[target].shift(lag)
        raw[f'act_lag_{lag}'] = raw.groupby('cfips')[target_act].diff(lag)
        feats.append(f'mbd_lag_{lag}')
        feats.append(f'act_lag_{lag}')

    lag = 1
    for window in [2, 4, 6]:
        raw[f'mbd_rollmea{window}_{lag}'] = raw.groupby('cfips')[f'mbd_lag_{lag}'].transform(
            lambda s: s.rolling(window, min_periods=1).sum())
        # raw[f'mbd_rollmea{window}_{lag}'] = raw[f'mbd_lag_{lag}'] - raw[f'mbd_rollmea{window}_{lag}']
        feats.append(f'mbd_rollmea{window}_{lag}')

    return raw, feats
