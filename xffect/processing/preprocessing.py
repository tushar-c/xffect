'''
module for preprocessing data
'''

import mne
import mne.decoding 
import numpy as np 
from sklearn.decomposition import FastICA, PCA
from scipy import linalg
from . import data_io
 

def csp(epoched_data, labels, info_obj, n_components=2, 
        transform_into='csp_space'):

    '''
    function for applying csp (common spatial patterns) to the data

    Parameters: epoched_data -> EpochsArray containing the epoched data
                labels -> numpy array or python list containing class labels
                info_obj -> mne.Info object

                n_components -> no. of filters (sorted eigenvectors) to choose
                transform_into -> the space to which the data be projected

    Returns: proj_epochs -> epoched data with the projected data after csp
    '''

    if not isinstance(epoched_data, mne.epochs.EpochsArray):
        raise Exception('epoched_data must be a valid EpochsArray')

    if not isinstance(labels, (np.ndarray, list)):
        raise Exception('labels must be either an ndarray or a list')

    if not isinstance(info_obj, mne.io.meas_info.Info):
        raise Exception('info_obj must be a valid Info object')

    csp_object = mne.decoding.CSP(n_components=n_components, 
                                    transform_into=transform_into)
    projected_data = csp_object.fit_transform(epoched_data.get_data(), labels)

    n_info = data_io.create_info(n_components, info_obj['sfreq'])

    proj_epochs = data_io.create_epochs(projected_data, n_info, 
                                epoched_data.events, epoched_data.event_id)

    return proj_epochs


def pca(epoched_data, info_obj, n_components=2, collapse_epochs=False, 
        whiten=False, transform=True):

    '''
    function for applying pca (principal component analysis) to the data


    Parameters: epoched_data -> EpochsArray containing the epoched data
                info_obj -> mne.Info object containing metadata
                n_components -> no. of components (sorted eigenvectors) to use

                collapse_epochs -> bool, whether to make one giant epoch or not
                whiten -> whether to whiten the data
                transform -> whether to apply transform or return components

    Returns: proj_epochs -> mne.EpochsArray or list based on value of transform
    '''

    if not isinstance(epoched_data, mne.epochs.EpochsArray):
        raise Exception('epoched_data must be a valid EpochsArray')

    if not isinstance(info_obj, mne.io.meas_info.Info):
        raise Exception('info_obj must be a valid Info object')

    X = epoched_data.get_data()

    if X.shape[1] < n_components:
        raise Exception('n_components cannot be greater than n_channels')

    n_epochs = X.shape[0]

    if collapse_epochs:
        print('collapsing epochs...')
        X_lst = [X[i] for i in range(n_epochs)]
        X = np.concatenate(X_lst)

    else:
        print('preserving epochs...')

    pca_obj = PCA(n_components=n_components, whiten=whiten)
    n_info = data_io.create_info(n_components, info_obj['sfreq'])

    if transform:
        pt = pca_obj.fit_transform
        epoch_wise_transforms = [pt(X[i].T).T for i in range(n_epochs)]
        data = np.array(epoch_wise_transforms)
        pca_proj_epochs = data_io.create_epochs(data, n_info, 
                            epoched_data.events, epoched_data.event_id)
        return pca_proj_epochs

    else:
        epoch_wise_fits = [pca_obj.fit(X[i].T) for i in range(n_epochs)]
        return epoch_wise_fits


def ica(epoched_data, info_obj, n_components=2, whiten=True, max_iter=250,
         random_state=0, transform=True, collapse_epochs=False):

     '''
        function for applying ica (independent component analysis) to the data
        with FastICA


        Parameters: epoched_data -> EpochsArray containing the epoched data
                    info_obj -> mne.Info object
                    n_components -> no. of components (sorted eigenvectors) to use

                    whiten -> whether to whiten the data
                    max_iter -> maximum FastICA iterations for convergence

                    random_state -> the random seed to be used
                    transform -> whether to apply transform or return components
                    collapse_epochs -> bool, whether to make one giant epoch or not

        Returns: proj_epochs -> mne.EpochsArray or list based on value of transform
    '''

    if not isinstance(epoched_data, mne.epochs.EpochsArray):
        raise Exception('epoched_data must be a valid EpochsArray')

    if not isinstance(info_obj, mne.io.meas_info.Info):
        raise Exception('info_obj must be a valid Info object')

    X = epoched_data.get_data()

    if X.shape[1] < n_components:
        raise Exception('n_components cannot be greater than n_channels')

    n_epochs = X.shape[0]

    if collapse_epochs:
        print('collapsing epochs...')
        X_lst = [X[i] for i in range(n_epochs)]
        X = np.concatenate(X_lst)

    else:
        print('preserving epochs...')


    ica_obj = FastICA(n_components=n_components, whiten=whiten, 
                    max_iter=max_iter, random_state=random_state)
    n_info = data_io.create_info(n_components, info_obj['sfreq'])

    if transform:
        ft = ica_obj.fit_transform
        epoch_wise_transforms = [ft(X[i].T).T for i in range(n_epochs)]
        data = np.array(epoch_wise_transforms)
        ica_proj_epochs = data_io.create_epochs(data, n_info, 
                        epoched_data.events, epoched_data.event_id)
        return ica_proj_epochs

    else:
        epoch_wise_fits = [ica_obj.fit(X[i]) for i in range(n_epochs)]
        return epoch_wise_fits
    
    