'''
module for designing and implementing filters
'''

import mne.filter as mne_filter
import mne 
import numpy as np 


def make_filter(data, sfreq, l_freq, h_freq, method='fir', **mne_kwargs):
    '''
    function for creating a filter to apply to raw data

    Parameters: data -> a numpy array or mne.EpochsArray
                *s_freq -> sampling frequency of the data
                *l_freq -> low pass frequency
                *h_freq -> high pass frequency
                method -> the type of filter we want to build (iir or fir)
                **mne_kwargs -> additional keyword arguments for mne

                *must be a float or convertible to float data type

        
    Returns: numpy array(or dict) if method = 'fir' (or 'iir')
    '''


    sfreq = float(sfreq)
    l_freq = float(l_freq)
    h_freq = float(h_freq)

    if not isinstance(method, str):
        raise Exception('method must be a string')
    else:
        if method.lower() not in ['fir', 'iir']:
            raise Exception("method must be either 'iir' or 'fir'")

    if isinstance(data, mne.EpochsArray):    
        raw_data = data.get_data()
    elif isinstance(data, np.ndarray):
        raw_data = data 
    elif data is None:
        raw_data = None
        print('warning: data is None, no sanity checking is going to be performed...')
    else:
        raise Exception('data must be either a valid EpochsArray or ndarray')

    custom_filter = mne_filter.create_filter(raw_data, sfreq, l_freq, h_freq, method=method)

    return custom_filter

