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


def replace_outliers(raw):
    lag = 1  # set the lag value to 1

    # 'mbd_lag_1' with the shifted values of the 'microbusiness_density' column, back-filled within each 'cfips' group
    raw[f'mbd_lag_{lag}'] = raw.groupby('cfips')['microbusiness_density'].shift(lag).bfill()

    # 'dif' with the absolute difference between the current and lag 'microbusiness_density' values, normalized by the lag value
    raw['dif'] = (raw['microbusiness_density'] / raw[f'mbd_lag_{lag}']).fillna(1).clip(0, None) - 1

    # set the 'dif' values to 0 where the lag 'microbusiness_density' is 0
    raw.loc[(raw[f'mbd_lag_{lag}'] == 0), 'dif'] = 0
    # set the 'dif' values to 1 where the current 'microbusiness_density' is
    # positive and the lag 'microbusiness_density' is 0
    raw.loc[(raw[f'microbusiness_density'] > 0) & (raw[f'mbd_lag_{lag}'] == 0), 'dif'] = 1
    # take the absolute value of the 'dif' column
    raw['dif'] = raw['dif'].abs()

    outliers = []
    cnt = 0

    for o in raw.cfips.unique():
        # get the indices for rows with the current 'cfips' value
        indices = (raw['cfips'] == o)

        # create a temporary copy of the data for the current 'cfips' value
        tmp = raw.loc[indices].copy().reset_index(drop=True)

        var = tmp.microbusiness_density.values.copy()

        # loop through the values in reverse order, starting from index 37 and ending at index 2
        for i in range(37, 2, -1):
            # calculate a threshold value as 20% of the mean of the first i values
            thr = 0.20 * np.mean(var[:i])

            # calculate the absolute difference between the current and previous values
            difa = abs(var[i] - var[i - 1])

            # if the difference is greater than or equal to the threshold
            if (difa >= thr):
                var[:i] *= (var[i] / var[i - 1])
                outliers.append(o)
                cnt += 1

        # set the first value to be 99% of the second value
        var[0] = var[1] * 0.99

        # update the 'microbusiness_density' values in the original data with the
        # modified values for the current 'cfips' value
        raw.loc[indices, 'microbusiness_density'] = var

    outliers = np.unique(outliers)
    print(f"number of unique cfips with outliers:{len(outliers)}, total number of outliers:{cnt}")
    return raw

# convert target


def new_target(raw):
    # same as raw.microbusiness_density.pct_change(1)
    raw['target'] = raw.groupby('cfips')['microbusiness_density'].shift(-1)
    raw['target'] = raw['target'] / raw['microbusiness_density'] - 1

    # Set the target value for rows with cfips value of 28055 to 0
    raw.loc[raw['cfips'] == 28055, 'target'] = 0.0

    # Set the target value for rows with cfips value of 48269 to 0
    raw.loc[raw['cfips'] == 48269, 'target'] = 0.0
    return raw

# feature engineering


def build_features(raw, target='microbusiness_density', target_act='active_tmp', lags=6):
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
