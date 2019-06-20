import random
import time 
import mne 
import numpy as np
from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_stream


def send_data(data, tag='random_tag', modality='NA', n_channels=2, s_freq=128, 
                dtype='float32', device_id='random_id', cond='inf', breaks=0.1):

    if not isinstance(data, (np.ndarray, mne.EpochsArray)):
        raise Exception('data must be either an ndarray or EpochsArray')

    if isinstance(data, np.ndarray):
        raw_data = data 
    else:
        raw_data = data.get_data()

    if raw_data.size != n_channels:
        raise Exception('raw_data.size must be equal to n_channels')

    if modality == 'NA':
        print('warning: modality is set to NA, consider setting it to',
                'an actual modality (EEG, ECG, etc.)')

    if tag == 'random_tag':
        print("warning: tag is set to 'random_tag', consider setting",
                'it to an actual tag (such as your device name)')

    info = StreamInfo(tag, modality, n_channels, s_freq, dtype, device_id)
    outlet = StreamOutlet(info)

    print('sending data...')
    if cond == 'inf':
        print("cond set to 'inf', will send data indefinitely, until external",
                    'interruption occurs')
        while True:
            outlet.push_sample(data)
            time.sleep(breaks)


def receive_data(modality='NA', cond='inf', record_timestamp=True):
    streams = resolve_stream('type', modality)
    stream_inlet = StreamInlet(streams[0])

    if modality == 'NA':
        print('warning: modality is set to NA, consider setting it to',
                'an actual modality (EEG, ECG, etc.)')

    if cond == 'inf':
        print("cond set to 'inf', will receive data indefinitely, until external", 
                    "interruption occurs")
        while True:
            sample, timestamp = stream_inlet.pull_sample()
            if record_timestamp:
                print('sample:', sample, 'timestamp:', timestamp)
            else:
                print('sample:', sample)


