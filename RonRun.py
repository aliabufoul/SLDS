import pickle
import random
import math
import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import electrocardiogram
from scipy.signal import find_peaks
from scipy import signal
import os
from operator import itemgetter
from scipy.interpolate import interp1d
from numpy.linalg import matrix_power
from numpy import linalg
from mpl_toolkits.mplot3d import Axes3D

from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from pypr.stattest.ljungbox import *
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf
import scipy.stats

from collections import namedtuple


def print_prediction_results(prediction):
    print("*******************************************************************************************************")
    print("Estimation Method: ", prediction.model.reg_method)
    print("T=\n", prediction.model.T)
    print("D=\n", prediction.model.D)
    print("Training Period: [%s, %s]" % (prediction.model.start_train_point, prediction.model.end_train_point))
    print("Training based on %s data" % (prediction.model.patient_id))

    print("")
    print("Testing the model based on %s data" % (prediction.patient_id))
    print('Mean squared error: %.2f'
          % mean_squared_error(prediction.data_test, prediction.data_pred))
    # The coefficient of determination: 1 is perfect prediction
    print('Coefficient of determination: %.2f'
          % r2_score(prediction.data_test, prediction.data_pred))
    print("The Noise Is White: ", white_noise(prediction.data_test, prediction.data_pred))

    fig, ax1 = plt.subplots(figsize=(18, 3))
    ax1.set_title('Heart Rate')
    ax1.set_xlabel('Time [sec]')
    ax1.set_ylabel('Heart Rate [bpm]')
    ax1.plot(prediction.time, prediction.data_test[0], linewidth=3)
    ax1.plot(prediction.time, prediction.data_pred[0], linewidth=3)
    ax1.legend(['Data', 'Predicted'], loc='best')

    ax2 = ax1.twinx()
    color = 'tab:gray'
    ax2.set_ylabel('Error [bpm]', color=color)
    ax2.plot(prediction.time, prediction.data_test[0] - prediction.data_pred[0], '--', color=color, linewidth=0.5)
    ax2.tick_params(axis='y', labelcolor=color)
    plt.show()

    fig, ax1 = plt.subplots(figsize=(18, 3))
    ax1.set_title('Systolic Blood Pressure')
    ax1.set_xlabel('Time [sec]')
    ax1.set_ylabel('Systolic Blood Pressure [mmHg]')
    ax1.plot(prediction.time, prediction.data_test[1], linewidth=3)
    ax1.plot(prediction.time, prediction.data_pred[1], linewidth=3)
    ax1.legend(['Data', 'Predicted'], loc='best')

    ax2 = ax1.twinx()
    color = 'tab:gray'
    ax2.set_ylabel('Error [mmHg]', color=color)
    ax2.plot(prediction.time, prediction.data_test[1] - prediction.data_pred[1], ':', color=color, linewidth=0.5)
    ax2.tick_params(axis='y', labelcolor=color)
    plt.show()

    fig, ax1 = plt.subplots(figsize=(18, 3))
    ax1.set_title('Respiration Rate')
    ax1.set_xlabel('Time [sec]')
    ax1.set_ylabel('Respiration Rate [Hz]')
    ax1.plot(prediction.time, prediction.data_test[2], linewidth=3)
    ax1.plot(prediction.time, prediction.data_pred[2], linewidth=3)
    ax1.legend(['Data', 'Predicted'], loc='best')

    ax2 = ax1.twinx()
    color = 'tab:gray'
    ax2.set_ylabel('Error [Hz]', color=color)
    ax2.plot(prediction.time, prediction.data_test[2] - prediction.data_pred[2], '--', color=color, linewidth=0.5)
    ax2.tick_params(axis='y', labelcolor=color)
    plt.show()

    '''
    #autocorelation graphs

    #plot_acf(prediction.data_test[0] - prediction.data_pred[0], lags=20, alpha=0.1)
    #plot_acf(prediction.data_test[1] - prediction.data_pred[1], lags=20, alpha=0.1)
    #plot_acf(prediction.data_test[2] - prediction.data_pred[2], lags=20, alpha=0.1)
    '''

    print("*******************************************************************************************************")


def rr(start, end, patient_id, print1, print2):
    if patient_id == 'simulator':
        time = pickle.load(open('samples/simulated_time', 'rb'))
        rr = pickle.load(open('samples/simulated_processed_rr', 'rb'))
        return time[(int)(start * F):(int)(end * F)], rr[(int)(start * F):(int)(end * F)]

    debug = print1 or print2
    if debug == False and os.path.exists('pickles/' + str(patient_id) + '-rr-processed') == True:
        time = pickle.load(open('pickles/' + str(patient_id) + '-time-processed', 'rb'))
        rr = pickle.load(open('pickles/' + str(patient_id) + '-rr-processed', 'rb'))
        return time[(int)(start * F):(int)(end * F)], rr[(int)(start * F):(int)(end * F)]

    if debug == False:
        start_ = 0
        end_ = 2300
    else:
        start_ = start
        if start != 0:
            start_ = start - MARGIN
        end_ = end + MARGIN
    resp = []
    filename = patient_id + "-MDC_RESP-62.5.csv"
    if print1 or print2:
        print(filename + ':')
        print("sampling frequency is ", F, " Hz")
    with open('samples/' + filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            resp.append(float(row[1]))

    resp_small = np.array(resp[(int)(start_ * 62.5):(int)(end_ * 62.5)])
    peaks, _ = find_peaks(resp_small, distance=31, height=1150, prominence=1)
    peaks_sec = (peaks + start_ * 62.5) * 0.016
    time = np.arange(start_, end_, 0.016)

    if print1:
        plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
        plt.plot(time, resp_small)
        plt.plot(peaks_sec, resp_small[peaks], "x")
        plt.title('Respiration')
        plt.xlabel('Time [sec]')
        plt.show()

    diffs = np.diff(peaks_sec)
    peaks_sec = peaks_sec[:len(diffs)] + diffs / 2
    rr = 60 / diffs

    rr_interp_func = interp1d(peaks_sec, rr)
    time = np.arange(start_ + MARGIN, end_ - MARGIN + 1 / F, 1 / F)

    if print2:
        plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
        plt.plot(peaks_sec, rr)
        plt.plot(time, rr_interp_func(time))
        plt.title('Respiration Rate')
        plt.xlabel('Time [sec]')
        plt.ylabel('[Hz]')
        plt.legend(['data', 'resampled'], loc='best')
        plt.show()

    if debug == False:
        with open('pickles/' + str(patient_id) + '-rr-processed', 'wb') as rr_file:
            pickle.dump(rr_interp_func(time), rr_file)
        return time[(int)(start * F):(int)(end * F)], rr[(int)(start * F):(int)(end * F)]


def hr(start, end, patient_id, print1, print2):
    if patient_id == 'simulator':
        time = pickle.load(open('samples/simulated_time', 'rb'))
        hr = pickle.load(open('samples/simulated_processed_hr', 'rb'))
        return time[(int)(start * F):(int)(end * F)], hr[(int)(start * F):(int)(end * F)]

    debug = print1 or print2
    if debug == False and os.path.exists('pickles/' + str(patient_id) + '-hr-processed') == True:
        time = pickle.load(open('pickles/' + str(patient_id) + '-time-processed', 'rb'))
        hr = pickle.load(open('pickles/' + str(patient_id) + '-hr-processed', 'rb'))
        return time[(int)(start * F):(int)(end * F)], hr[(int)(start * F):(int)(end * F)]

    if debug == False:
        start_ = 0
        end_ = 2300
    else:
        start_ = start
        if start != 0:
            start_ = start - MARGIN
        end_ = end + MARGIN
    ecg = []
    filename = patient_id + "-MDC_ECG_ELEC_POTL_II-500.csv"
    if print1 or print2:
        print(filename + ':')
        print("sampling frequency is ", F, " Hz")
    with open('samples/' + filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            ecg.append(float(row[1]))

    ecg_small = np.array(ecg[start_ * 500:end_ * 500])
    peaks, _ = find_peaks(ecg_small, height=8270, distance=155)  # 60/(193*0.002)
    peaks_sec = (peaks + start_ * 500) * 0.002
    time = np.arange(start_, end_, 0.002)

    if print1:
        plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
        plt.plot(time, ecg_small)
        plt.plot(peaks_sec, ecg_small[peaks], "x")
        plt.title('ECG')
        plt.xlabel('Time [sec]')
        plt.show()

    diffs = np.diff(peaks_sec)
    peaks_sec = peaks_sec[:len(diffs)] + diffs / 2
    hr = 60 / diffs

    hr_interp_func = interp1d(peaks_sec, hr)
    time = np.arange(start_ + MARGIN, end_ - MARGIN + 1 / F, 1 / F)

    if print2:
        plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
        plt.plot(peaks_sec, hr)
        plt.plot(time, hr_interp_func(time))
        plt.title('Heart Rate')
        plt.xlabel('Time [sec]')
        plt.ylabel('[bpm]')
        plt.legend(['data', 'resampled'], loc='best')
        plt.show()

    if debug == False:
        with open('pickles/' + str(patient_id) + '-hr-processed', 'wb') as hr_file:
            pickle.dump(hr_interp_func(time), hr_file)
        return time[(int)(start * F):(int)(end * F)], hr[(int)(start * F):(int)(end * F)]



def bp(start, end, patient_id, print1, print2):
    if patient_id == 'simulator':
        time = pickle.load(open('samples/simulated_time', 'rb'))
        bp = pickle.load(open('samples/simulated_processed_bp', 'rb'))
        return time[(int)(start * F):(int)(end * F)], bp[(int)(start * F):(int)(end * F)]

    debug = print1 or print2
    if debug == False and os.path.exists('pickles/' + str(patient_id) + '-bp-processed') == True:
        time = pickle.load(open('pickles/' + str(patient_id) + '-time-processed', 'rb'))
        bp = pickle.load(open('pickles/' + str(patient_id) + '-bp-processed', 'rb'))
        return time[(int)(start * F):(int)(end * F)], bp[(int)(start * F):(int)(end * F)]

    if debug == False:
        start_ = 0
        end_ = 2300
    else:
        start_ = start
        if start != 0:
            start_ = start - MARGIN
        end_ = end + MARGIN
    bp = []
    filename = patient_id + "-MDC_PRESS_BLD_ART_ABP-125.csv"
    if print1 or print2:
        print(filename + ':')
        print("sampling frequency is ", F, " Hz")
    with open('samples/' + filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            bp.append((0.0625 * float(row[1])) - 40)

    bp_small = np.array(bp[start_ * 125:end_ * 125])
    peaks, _ = find_peaks(bp_small, prominence=1, distance=42)
    peaks_sec = (peaks + start_ * 125) * 0.008
    sys = bp_small[peaks]
    time = np.arange(start_, end_, 0.008)

    if print1:
        plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
        plt.plot(time, bp_small)
        plt.plot(peaks_sec, sys, "x")
        plt.title('Blood Pressure')
        plt.xlabel('Time [sec]')
        plt.show()

    sys_interp_func = interp1d(peaks_sec, sys)
    time = np.arange(start_ + MARGIN, end_ - MARGIN + 1 / F, 1 / F)

    if print2:
        plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
        plt.plot(peaks_sec, sys)
        plt.plot(time, sys_interp_func(time))
        plt.title('Systolic Blood Pressure')
        plt.xlabel('Time [sec]')
        plt.legend(['data', 'resampled'], loc='best')
        plt.show()

    if debug == False:
        with open('pickles/' + str(patient_id) + '-time-processed', 'wb') as time_file:
            pickle.dump(time, time_file)
        with open('pickles/' + str(patient_id) + '-bp-processed', 'wb') as bp_file:
            pickle.dump(sys_interp_func(time), bp_file)
        return time[(int)(start * F):(int)(end * F)], bp[(int)(start * F):(int)(end * F)]


def simulator():
    for file in os.listdir("./pickles"):
        if file.find("simulator"):
            os.remove("pickles/" +file)


    # build a list of 3 random models
    Simulator_Model = namedtuple('Simulator_Model', ['id', 'T', 'D'])
    models = []

    #mu = np.array([[np.random.uniform(80, 100)], [np.random.uniform(90, 110)], [np.random.uniform(20, 35)]])

    for j in range(3):
        A = np.random.randn(3, 3)
        eigvals1 = linalg.eigvals(A)
        lambda_ = np.max(np.abs(eigvals1))
        delta_ = np.random.uniform(1.01 ,1.1)
        A = A/ (delta_ * lambda_)

        mu = np.array([[np.random.uniform(80, 100)], [np.random.uniform(90, 110)], [np.random.uniform(20, 35)]])

        model = Simulator_Model(j, A, mu)
        models.append(model)

    # simulate a data for 3 periods (each period based on a randomly chosen model from the list)
    period_length_array = np.arange(30, 50)  # period_length/10

    s0 = np.random.randn(3, 1)
    s_n = s0
    x_n = s_n + mu

    hr = []
    bp = []
    rr = []
    hr.append(x_n[0])
    bp.append(x_n[1])
    rr.append(x_n[2])
    total_length = 0
    for j in range(3):
        model = random.choice(models)
        length = random.choice(period_length_array)
        total_length = total_length + length
        print("model " + str(model.id) + " from " + str((int)((total_length - length) / F)) + " to " + str(
            (int)(total_length / F)))
        print("T = \n", model.T)
        print("mu = \n", model.D)
        print("")

        for i in range(length):
            #s_n = np.dot(model.T, s_n) + np.multiply(0.01*model.D, np.random.randn(3, 1))
            s_n = np.dot(model.T, s_n) + np.random.randn(3, 1)
            x_n = s_n + model.D
            hr.append(x_n[0])
            bp.append(x_n[1])
            rr.append(x_n[2])
    data = np.row_stack((np.array(hr).transpose(), np.array(bp).transpose(), np.array(rr).transpose()))
    hr = data[0]
    bp = data[1]
    rr = data[2]
    time = np.arange(0, (int)(len(hr) / F), (int)(1 / F))

    # plot the simulated data
    plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
    plt.plot(time, hr, label='H.R')
    plt.title('Physiological signals from simulator')
    plt.xlabel('Time [sec]')
    #plt.ylabel('[bpm]')
    #plt.show()

    #plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
    plt.plot(time, bp, label='B.P')
    #plt.title('Systolic Blood Pressure')
    #plt.xlabel('Time [sec]')
    #plt.show()

    #plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
    plt.plot(time, rr, label='resp')
    #plt.title('Respiratory Rate')
    #plt.xlabel('Time [sec]')
    plt.legend()
    plt.grid()
    plt.show()

    # save the simulated data in files
    with open('samples/simulated_processed_hr', 'wb') as file:
        pickle.dump(hr, file)
    with open('samples/simulated_processed_bp', 'wb') as file:
        pickle.dump(bp, file)
    with open('samples/simulated_processed_rr', 'wb') as file:
        pickle.dump(rr, file)
    with open('samples/simulated_time', 'wb') as file:
        pickle.dump(time, file)


def all_models(start, end, patient_id):
    models = []
    for start_ in np.arange(start, end, 30):
        for end_ in np.arange(start_ + 100, end + 1, 70):
            for reg_method in [3]:
                models.append(estimate_T(start_, end_, patient_id, reg_method))
    return models


Period = namedtuple('Period', ['start', 'end'])


def all_periods(start, end):
    periods = []
    for start_ in np.arange(start, end, 100):
        end_ = start_ + 100
        period = Period(start_, end_)
        periods.append(period)
    return periods


def models_periods_table(start, end, patient_id):
    if os.path.exists('pickles/table' + str(patient_id) + '-' + str(start) + '-' + str(end)) == True:
        table = pickle.load(open('pickles/table' + str(patient_id) + '-' + str(start) + '-' + str(end), 'rb'))
        models = pickle.load(open('pickles/models' + str(patient_id) + '-' + str(start) + '-' + str(end), 'rb'))
        periods = pickle.load(open('pickles/periods' + str(patient_id) + '-' + str(start) + '-' + str(end), 'rb'))
        return table, models, periods

    models = all_models(start, end, patient_id)
    periods = all_periods(start, end)

    table = np.empty((len(periods), len(models)), dtype=object)
    print(table.shape)

    for i in range(len(models)):
        for j in range(len(periods)):
            prediction = test_T(models[i], periods[j].start, periods[j].end, patient_id)
            table[j][i] = (prediction.data_test, prediction.data_pred)
    # RON COMMENTED:
    """
    with open('pickles/table' + str(patient_id) + '-' + str(start) + '-' + str(end), 'wb') as table_file:
        pickle.dump(table, table_file)
    with open('pickles/models' + str(patient_id) + "-" + str(start) + "-" + str(end), 'wb') as models_file:
        pickle.dump(models, models_file)
    with open('pickles/periods' + str(patient_id) + "-" + str(start) + "-" + str(end), 'wb') as periods_file:
        pickle.dump(periods, periods_file)
    """
    return table, models, periods


def translate_table(table, models, periods, ac_limit, mean_limit):
    clean_table = np.empty((len(periods), len(models)), dtype=object)
    for i in range(len(models)):
        for j in range(len(periods)):
            (data_test, data_pred) = table[j][i]
            error = data_test - data_pred

            left_rank_mean = right_rank_mean = rank_mean = extreme_mean = 0
            rank_ac = 0  # autocorelation

            for signal in [0, 1, 2]:
                if np.abs(np.mean(error[signal])) <= mean_limit:
                    rank_mean = rank_mean + 1
                if np.abs(np.mean(error[signal])) <= 0.4:
                    extreme_mean = extreme_mean + 1
                if np.abs(np.mean(error[signal][0:(int)(len(error[signal]) / 2)])) <= 0.7:
                    left_rank_mean = left_rank_mean + 1
                if np.abs(np.mean(error[signal][(int)(len(error[signal]) / 2):])) <= 0.7:
                    right_rank_mean = right_rank_mean + 1

                h, pV, Q, cV = lbqtest(error[signal], range(1, min(20, len(error[signal]))), alpha=ac_limit)
                if not any(h):
                    rank_ac = rank_ac + 1

            if (
                    rank_ac == 3 and rank_mean == 3) or extreme_mean == 3:  # and left_rank_mean == 3 and right_rank_mean == 3:
                clean_table[j][i] = 1 - np.minimum(1, np.round(np.mean(np.abs(error)) / 100, 5))
            else:
                clean_table[j][i] = 0
    return clean_table


def clean_table(table, models, periods):
    if len(periods) == 1:
        return table
    clean_table = np.empty((len(periods), len(models)), dtype=object)
    for i in range(len(models)):
        for j in range(len(periods)):
            clean_table[j][i] = table[j][i]
            if table[j][i] > 0:
                if (j == 0 and table[1][i] == 0) or (j == (len(periods) - 1) and table[len(periods) - 2][i] == 0) or (
                        (j != 0) and (j != (len(periods) - 1)) and (table[j - 1][i] == 0) and (table[j + 1][i] == 0)):
                    clean_table[j][i] = 0
    return clean_table


def popular_model(table, models, periods):
    Period = namedtuple('Period', ['start', 'end'])
    models_popularity = np.zeros(len(models))
    model_periods = []
    for i in range(len(models)):
        pre_fit = 0
        for j in range(len(periods)):
            if pre_fit == 1 and table[j][i] > 0:
                models_popularity[i] = models_popularity[i] + 1
            if table[j][i] > 0:
                models_popularity[i] = models_popularity[i] + table[j][i]
                pre_fit = 1
            else:
                pre_fit = 0

    print("models_popularity", models_popularity)
    popular_index = np.argmax(models_popularity)
    print("popular_index", popular_index)
    to_delete = []
    for i in range(len(periods)):
        if table[i][popular_index] > 0:
            if len(model_periods) == 0 or model_periods[-1].end != periods[i].start:
                model_periods.append(periods[i])
            else:
                last_period_start = model_periods[-1].start
                model_periods.pop(-1)
                model_periods.append(Period(last_period_start, periods[i].end))
            to_delete.append(i)
    for i in reversed(to_delete):
        table = np.delete(table, i, axis=0)
        periods.pop(i)
    return table, periods, models[popular_index], model_periods


Model_Periods = namedtuple('Model_Periods', ['model', 'model_periods'])


def estimate_T(start_train_point, end_train_point, patient_id, reg_method):
    Model = namedtuple('Model',
                       ['T', 'D', 'start_train_point', 'end_train_point', 'patient_id', 'reg_method', 'scaler'])
    D = np.array([[0], [0], [0]])

    reg1 = reg2 = reg3 = None
    scaler = None

    time, bp_train = bp(start_train_point, end_train_point, patient_id, 0, 0)
    time, hr_train = hr(start_train_point, end_train_point, patient_id, 0, 0)
    time, rr_train = rr(start_train_point, end_train_point, patient_id, 0, 0)

    # scaling
    data_train = np.column_stack((hr_train, bp_train, rr_train))
    scaler = StandardScaler()
    scaler.fit(data_train)
    data_train_scaled = scaler.transform(data_train)
    bp_train = data_train_scaled[:, 1]
    hr_train = data_train_scaled[:, 0]
    rr_train = data_train_scaled[:, 2]

    # x is the 3 signals data from 0 to n-1
    x = np.column_stack((hr_train, bp_train, rr_train))
    x = np.delete(x, -1, axis=0)

    # y_i is the data from 1 to n for the i signal
    y_bp = np.delete(bp_train, 0)
    y_rr = np.delete(rr_train, 0)
    y_hr = np.delete(hr_train, 0)

    if reg_method == 1:
        hr_coef = np.dot(np.linalg.inv(np.dot(x.transpose(), x)), np.dot(x.transpose(), y_hr))
        bp_coef = np.dot(np.linalg.inv(np.dot(x.transpose(), x)), np.dot(x.transpose(), y_bp))
        rr_coef = np.dot(np.linalg.inv(np.dot(x.transpose(), x)), np.dot(x.transpose(), y_rr))

    else:
        if reg_method == 2:
            reg1 = linear_model.LinearRegression(fit_intercept=False)
            reg2 = linear_model.LinearRegression(fit_intercept=False)
            reg3 = linear_model.LinearRegression(fit_intercept=False)

        if reg_method == 3:
            reg1 = linear_model.LassoCV(cv=5, fit_intercept=False)
            reg2 = linear_model.LassoCV(cv=5, fit_intercept=False)
            reg3 = linear_model.LassoCV(cv=5, fit_intercept=False)

        if reg_method == 4:
            reg1 = linear_model.RidgeCV(fit_intercept=False)
            reg2 = linear_model.RidgeCV(fit_intercept=False)
            reg3 = linear_model.RidgeCV(fit_intercept=False)

        reg1.fit(x, y_hr)
        hr_coef = reg1.coef_
        hr_intercept = reg1.intercept_

        reg2.fit(x, y_bp)
        bp_coef = reg2.coef_
        bp_intercept = reg2.intercept_

        reg3.fit(x, y_rr)
        rr_coef = reg3.coef_
        rr_intercept = reg3.intercept_

        D = np.row_stack((hr_intercept, bp_intercept, rr_intercept))

    D = scaler.mean_.reshape(3, 1)
    T = np.row_stack((hr_coef, bp_coef, rr_coef))
    model = Model(T, D, start_train_point, end_train_point, patient_id, reg_method, scaler)

    return model


def white_noise(data_test, data_pred):
    error = data_test - data_pred
    left_rank_mean = right_rank_mean = rank_mean = 0
    rank_ac = 0  # autocorelation

    for signal in [0, 1, 2]:
        for epsilon, bonous in [(0.2, 1), (0.4, 10), (0.6, 100), (0.8, 1000), (1, 10000), (1.4, 100000), (1.8, 1000000),
                                (2.2, 10000000)]:
            if np.abs(np.mean(error[signal])) < epsilon:
                rank_mean = rank_mean + bonous
            if np.abs(np.mean(error[signal][0:(int)(len(error[signal]) / 2)])) < epsilon:
                left_rank_mean = left_rank_mean + bonous
            if np.abs(np.mean(error[signal][(int)(len(error[signal]) / 2):])) < epsilon:
                right_rank_mean = right_rank_mean + bonous

        for alpha_, bonous in [(0.15, 1), (0.10, 10), (0.08, 100), (0.05, 1000), (0.03, 10000), (0.02, 100000),
                               (0.01, 1000000)]:
            h, pV, Q, cV = lbqtest(error[signal], range(1, min(20, len(error[signal]))), alpha=alpha_)
            if not any(h):
                rank_ac = rank_ac + bonous
    return (rank_mean, rank_ac, left_rank_mean, right_rank_mean)

def test_T(model, start_point, end_point, patient_id):
    Prediction = namedtuple('Prediction', ['time', 'data_test', 'data_pred', 'patient_id', 'model', 'white_noise_rank'])

    time, bp_test = bp(start_point, end_point, patient_id, 0, 0)
    time, hr_test = hr(start_point, end_point, patient_id, 0, 0)
    time, rr_test = rr(start_point, end_point, patient_id, 0, 0)

    # scaling
    data_test = np.column_stack((hr_test, bp_test, rr_test))
    scaler = StandardScaler()
    scaler.fit(data_test)
    data_test_scaled = scaler.transform(data_test)

    data_pred = np.dot(model.T, data_test_scaled.transpose())  # + np.repeat(model.D, len(time), axis=1)

    # reverse scaling
    data_pred = model.scaler.inverse_transform(data_pred.transpose())
    data_pred = data_pred.transpose()

    time = np.delete(time, 0)
    data_test = np.delete(data_test.transpose(), 0, axis=1)
    data_pred = np.delete(data_pred, -1, axis=1)

    prediction = Prediction(time, data_test, data_pred, patient_id, model, white_noise(data_test, data_pred))

    return prediction


def split_to_models(start, end, patient_id):
    best_rank = -np.inf
    best_results = None
    ac_limit_list = [0.1, 0.05, 0.03]
    mean_limit_list = [0.9, 1.8, 2.3, 2.8]
    for ac_limit in ac_limit_list:
        for mean_limit in mean_limit_list:
            result_rank = 0.4 * (ac_limit - min(ac_limit_list)) / (max(ac_limit_list) - min(ac_limit_list)) + 1 - (
                        mean_limit - min(mean_limit_list)) / (max(mean_limit_list) - min(mean_limit_list))
            print("ac_limit, mean_limit = ", ac_limit, mean_limit)
            table, models, periods = models_periods_table(start, end, patient_id)
            table = translate_table(table, models, periods, ac_limit, mean_limit)
            table = clean_table(table, models, periods)

            results = []
            for i in range(3):
                table, periods, popular_model_, model_periods = popular_model(table, models, periods)
                if len(model_periods) == 0:
                    break;
                else:
                    results.append(Model_Periods(popular_model_, model_periods))

            # result_rank = 0.01*result_rank + ((end-start)-len(periods)*100)/len(results)
            result_rank = 0.1 * result_rank + 1 - len(results) / 3
            result_rank = 0.25 * result_rank + ((end - start) - len(periods) * 100) / (end - start)

            if result_rank > best_rank:
                best_rank = result_rank
                best_results = results

    for result in best_results:
        for period in result.model_periods:
            print_prediction_results(test_T(result.model, period.start, period.end, patient_id))


# configs:
F = 0.1
FIG_WIDTH=18
FIG_HEIGHT=20
MARGIN=10

# run simulator:
simulator()

# run algorithm:

split_to_models(0, 300, "simulator")