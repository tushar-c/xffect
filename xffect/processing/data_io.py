'''
module for creating mne data types from numpy arrays and for going the other way round
'''

import numpy as np 
import mne 


def create_info(channel_names, sampling_rate, *args, 
                    channel_types=None, montage=None, **kwargs):

    '''

    function for creating mne.Info object from signal metadata

    Parameters: channel_names -> list of names or an integer denoting number of channels
                sampling_rate -> the sampling rate of the signal
                channel_types -> the modality within each channel (EEG, ECG, EMG, etc.)
                montage -> whether to follow a certain head montage of the electrodes

                *args -> additional positional arguments
                **kwargs -> additional keyword arguments

    Returns: info_obj -> an mne Info object 

    '''

    if channel_names:
        if not isinstance(channel_names, (list, int)):
            raise Exception('channel_names must be a list of strings or an int')

    if channel_types:
        if not isinstance(channel_types, list):
            raise Exception('channel_types must be a list')
    
    if channel_names and channel_types:
        if isinstance(channel_names, list):
            assert len(channel_names) == len(channel_types)

    if sampling_rate:
        if not isinstance(sampling_rate, float):
            raise Exception('sampling_rate must be a float')

    if montage:
        if not isinstance(montage, list):
            raise Exception('montage must be a list')

    
    info_obj = mne.create_info(channel_names, sampling_rate, channel_types, 
                                montage, *args, **kwargs)

    return info_obj


def create_raw(info_obj, data=None):

    '''
    function for creating mne.RawArray from mne.Info object

    Parameters: info_obj -> a valid mne.Info object
                data -> a valid numpy array or list of data

    Returns: mne RawArray containing the data and parameters from the info_obj
    '''


    if not info_obj:
        raise Exception('info_obj must be a valid mne.Info object')

    else:
        n_channels = info_obj['nchan']
        sampling_rate = info_obj['sfreq']

    if data is None:
        print('warning: data is None, will use information from info_obj',
                'to generate random data...')
        data = np.random.randn(n_channels, int(sampling_rate))
    
    else:
        if isinstance(data, np.ndarray):
            data = data 
        elif isinstance(data, list):
            data = np.array(data)
        else:
            raise Exception('data must be a numpy array or a list')

    custom_raw = mne.io.RawArray(data, info_obj)
    return custom_raw


def create_epochs(data, info, events, event_id, tmin=None):
    '''
    function for epoching the data based on event occurences during experimentation

    Parameters: data -> numpy array containing the (raw) readings
                info -> mne.Info object containing meta details
                event_id -> dict with (usually) mappings from events -> ints
                events -> numpy array with event indices & transition labels

                tmin -> time from which to consider the beginning of recording

    Returns: mne.EpochsArray object that contains the epoched data
    '''


    if not isinstance(data, np.ndarray):
        raise Exception('data must be a valid ndarray')

    if not isinstance(info, mne.io.meas_info.Info):
        raise Exception('info must be a valid info object')

    if not isinstance(event_id, dict):
        raise Exception('event_id must be a valid python dict')

    if not isinstance(events, np.ndarray):
        raise Exception('events must be a valid ndarray')


    if tmin:
        tmin = float(tmin)
    
    else:
        print('warning: tmin is None, default value of -0.1 will be used')
        tmin = -0.1

    custom_epochs = mne.EpochsArray(data=data, info=info, events=events, 
                                    tmin=tmin, event_id=event_id)
    return custom_epochs


def create_evoked(data):
    '''

    function for returning averaged data from raw data

    Parameters: data -> numpy array containing the data

    Returns: evoked -> numpy array that has averaged data
    '''

    if not isinstance(data, np.ndarray):
        raise Exception('data must be a valid ndarray')
    
    evoked = data.mean(0)
    return evoked


def get_numpy(raw_obj):
    '''

    function for fetching the raw numpy array from the raw array

    Parameters: raw_obj -> mne.RawArray containing the data

    Returns: numpy array with the raw data
    '''

    if not isinstance(raw_obj, mne.io.array.array.RawArray):
        return raw_obj.get_data()
    else:
        raise Exception('raw_obj must be a raw object')


