'''
module for performing statistical inference 
'''

import mne 
import numpy as np 
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_val_score


def prep_input(data, avg=True):

    '''
    function for preparing data by averaging

    Parameters: data -> mne.EpochsArray containing epoched data
                avg -> whether to average data or not, default is True

    Returns: avg_raw_data -> averaged raw data
             labels -> session labels

    '''
    if not isinstance(data, mne.EpochsArray):
        raise Exception('data must be a valid EpochsArray')

    raw_data = data.get_data()
    raw_data = raw_data.transpose(0, 2, 1)

    if avg:
        avg_raw_data = np.mean(raw_data, axis=1)
    else:
        raise Exception('currently only average behavior is supported'
                        'must be set to True')

    events = data.events
    if events.shape[1] != 3:
        raise Exception('events must have shape: N x 3')
    
    labels = events[:, 2]
    return avg_raw_data, labels


def linear_regression(data, fit=True, avg=True, train_score=True, **kwargs):

    '''
    function for performing linear regression on data

    Parameters: data -> mne.EpochsArray containing epoched data
                fit -> bool, whether to fit the model to the data or not
                avg -> bool, whether to average the data before fitting, default True
                train_score -> bool, whether to evaluate score on training data or not
                kwargs -> additional (sklearn) keyword arguments for the model

    Returns: lin_reg / lin_reg_fit / score -> fit (or fit and scored) models 
                                                based on arguments

    '''

    lin_reg = LinearRegression(**kwargs)
    if not fit:
        return lin_reg

    avg_raw_data, labels = prep_input(data, avg=avg)
    lin_reg_fit = lin_reg.fit(avg_raw_data, labels)

    if train_score:
        score = lin_reg_fit.score(avg_raw_data, labels)
        return lin_reg_fit, score

    return lin_reg_fit


def ridge_regression(data, fit=True, avg=True, train_score=True, **kwargs):

    '''
    function for performing ridge regression on data

    Parameters: data -> mne.EpochsArray containing epoched data
                fit -> bool, whether to fit the model to the data or not
                avg -> bool, whether to average the data before fitting, default True
                train_score -> bool, whether to evaluate score on training data or not
                kwargs -> additional (sklearn) keyword arguments for the model

    Returns: ridge_reg / ridge_reg_fit / score -> fit (or fit and scored) models 
                                                based on arguments

    '''

    ridge_reg = RidgeClassifier(**kwargs)
    if not fit:
        return ridge_reg

    avg_raw_data, labels = prep_input(data, avg=avg)
    ridge_reg_fit = ridge_reg.fit(avg_raw_data, labels)

    if train_score:
        score = ridge_reg_fit.score(avg_raw_data, labels)
        return ridge_reg_fit, score

    return ridge_reg_fit


def logistic_regression(data, fit=True, avg=True, train_score=True, **kwargs):

    '''
    function for performing logistic regression on data

    Parameters: data -> mne.EpochsArray containing epoched data
                fit -> bool, whether to fit the model to the data or not
                avg -> bool, whether to average the data before fitting, default True
                train_score -> bool, whether to evaluate score on training data or not
                kwargs -> additional (sklearn) keyword arguments for the model

    Returns: log_reg / log_reg_fit / score -> fit (or fit and scored) models 
                                                based on arguments
    '''

    log_reg = LogisticRegression(**kwargs)
    if not fit:
        return log_reg

    avg_raw_data, labels = prep_input(data, avg=avg)
    log_reg_fit = log_reg.fit(avg_raw_data, labels)

    if train_score:
        score = log_reg_fit.score(avg_raw_data, labels)
        return log_reg_fit, score
    
    return log_reg_fit


def lda(data, fit=True, avg=True, train_score=True, **kwargs):

    '''
    function for performing linear discriminant analysis on data

    Parameters: data -> mne.EpochsArray containing epoched data
                fit -> bool, whether to fit the model to the data or not
                avg -> bool, whether to average the data before fitting, default True
                train_score -> bool, whether to evaluate score on training data or not
                kwargs -> additional (sklearn) keyword arguments for the model

    Returns: lda_clf / lda_clf_fit / score -> fit (or fit and scored) models 
                                                based on arguments

    '''

    lda_clf = LinearDiscriminantAnalysis(**kwargs)
    if not fit:
        return lda_clf

    avg_raw_data, labels = prep_input(data, avg=avg)
    lda_clf_fit = lda_clf.fit(avg_raw_data, labels)

    if train_score:
        score = lda_clf_fit.score(avg_raw_data, labels)
        return lda_clf_fit, score
 
    return lda_clf_fit


def qda(data, fit=True, avg=True, train_score=True, **kwargs):

    '''
    function for performing quadratic discriminant analysis on data

    Parameters: data -> mne.EpochsArray containing epoched data
                fit -> bool, whether to fit the model to the data or not
                avg -> bool, whether to average the data before fitting, default True
                train_score -> bool, whether to evaluate score on training data or not
                kwargs -> additional (sklearn) keyword arguments for the model

    Returns: qda_clf / qda_clf_fit / score -> fit (or fit and scored) models 
                                                based on arguments
    '''

    qda_clf = QuadraticDiscriminantAnalysis(**kwargs)
    if not fit:
        return qda_clf

    avg_raw_data, labels = prep_input(data, avg=avg)
    qda_clf_fit = qda_clf.fit(avg_raw_data, labels)

    if train_score:
        score = qda_clf_fit.score(avg_raw_data, labels)
        return qda_clf_fit, score
    
    return qda_clf_fit


def svm(data, fit=True, avg=True, train_score=True, **kwargs):

    '''
    function for fitting support vector machines to data

    Parameters: data -> mne.EpochsArray containing epoched data
                fit -> bool, whether to fit the model to the data or not
                avg -> bool, whether to average the data before fitting, default True
                train_score -> bool, whether to evaluate score on training data or not
                kwargs -> additional (sklearn) keyword arguments for the model

    Returns: svm_clf / svm_clf_fit / score -> fit (or fit and scored) models 
                                                based on arguments

    '''

    svm_clf = SVC(**kwargs)
    if not fit:
        return svm_clf

    avg_raw_data, labels = prep_input(data, avg=avg)

    svm_clf_fit = svm_clf.fit(avg_raw_data, labels)

    if train_score:
        score = svm_clf_fit.score(avg_raw_data, labels)
        return svm_clf_fit, score
    
    return svm_clf_fit


def cross_validation(algorithm, data, accuracy_est=True, **kwargs):

    '''
    function for performing cross validation of algorithm on data

    Parameters: algorithm -> sklearn algorithm for training
                data -> mne.EpochsArray containing epoched data
                accuracy_est -> bool, whether to estimate accuracy of fit model on data
                kwargs -> additional (sklearn) keyword arguments for the model

    Returns: scores / accuracy_lower / accuracy_upper -> scores and accuracy
                                                        bounds based on arguments

    '''
    
    if not isinstance(data, mne.EpochsArray):
        raise Exception('data must be a valid EpochsArray')

    avg_features, labels = prep_input(data)
    algo_fit = algorithm(data, fit=False)
    scores = cross_val_score(algo_fit, avg_features, labels, **kwargs)

    if accuracy_est:
        score_mean = scores.mean()
        score_std = scores.std()
        accuracy_lower = score_mean - score_std * 2
        accuracy_upper = score_mean + score_std * 2
        return scores, (accuracy_lower, accuracy_upper)

    return scores

