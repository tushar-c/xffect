import numpy as np 
import mne 
import os 
import tarfile
import pandas as pd
from scipy.io import loadmat
from urllib.request import urlretrieve


# eesd = eight emotion semantics dataset
SUPPORTED_DATASETS = ['eesd']


def download_file(filename='eesd', dir_name='MAS622data', 
                    extension='.tar.gz', keep_as_temp=True):

    if filename not in SUPPORTED_DATASETS:
        raise Exception('filename is not supported, supported datasets are: '
                        '{}'.format(' '.join([i for i in SUPPORTED_DATASETS])))

    if filename == 'eesd':
        file_url = 'http://affect.media.mit.edu/downloads/data-sentics/SetA.tar.gz'

    dl_file, _ = urlretrieve(file_url, filename+extension)

    if dl_file.endswith('.tar.gz'):
        filename += '.tar.gz'
        tar = tarfile.open(dl_file, "r:gz")
    else:
        raise Exception('unrecognized file format')

    tar.extractall()
    tar.close()

    cwd = os.path.abspath(os.getcwd())

    dl_file_path = os.path.join(cwd, dir_name)
    
    return dl_file_path, filename


def eight_emotion_states_dataset(extension='.mat', separated=False,
                                    with_headers=True, dir_name='MAS622data'):

    all_days_array = []

    current_files = os.listdir()

    emotion_files = [f for f in current_files if os.path.splitext(f)[1] == extension]
    N = len(emotion_files)
    
    print("files with '{}' extension found: {}".format(extension, N))
    print(emotion_files)

    for i in range(N):
        day_filename = os.path.splitext(emotion_files[i])[0]

        load_file = loadmat(emotion_files[i])

        day_array = load_file[day_filename]

        emotions = ['no_emotion', 'anger', 'hate', 'grief', 'platonic_love', 'romantic_love', 'joy', 'reverence']
        sensors = ['emg(jaw)', 'bvp', 'gsr(palm)', 'respiration']

        cols = day_array.shape[1]

        total_sensors = len(sensors)
        step_size = int(cols / total_sensors)

        sensor_slices = [i for i in range(0, cols, step_size)]
        sensor_slices.append(cols)

        sensor_dict = {i: {} for i in sensors}

        for i in range(len(sensor_slices) - 1):
            sensor_wise_slice = day_array[:, sensor_slices[i]: sensor_slices[i + 1]]
            emotions_in_slice = sensor_wise_slice.shape[1]
            assert emotions_in_slice == len(emotions)
            for e in range(len(emotions)):
                emotion = emotions[e]
                emotion_col_in_sensor = sensor_wise_slice[:, e]
                sensor_dict[sensors[i]][emotion] = emotion_col_in_sensor

        header_array = []

        for s in sensor_dict:
            for k in sensor_dict[s].keys():
                header_array.append(s + "_" + k)
        
        header_array = np.array(header_array).reshape(1, total_sensors * len(emotions))

        if with_headers:
            data_array = np.append(header_array, day_array, axis=0)
        else:
            data_array = day_array

        all_days_array.append(data_array)

        if separated:
            return sensor_dict
    
    return all_days_array


def prepare_data(filename, dl_file_path, keep_others=False, 
                    extension='.mat', del_file=True):
    curr_dir = os.path.abspath(os.getcwd())
    target_dir = dl_file_path

    os.chdir(target_dir)

    data_array = eight_emotion_states_dataset()

    all_files = os.listdir()
    to_remove = []

    for each_file in all_files:
        if os.path.splitext(each_file)[1] != extension:
            to_remove.append(each_file)

    for t in to_remove:
        os.remove(t)

    os.chdir(curr_dir)
    if del_file:
        os.remove(filename)

    return data_array


def get_file(*args, **kwargs):
    dl_file_path, filename = download_file(**kwargs)
    data = prepare_data(filename, dl_file_path, **kwargs)
    return data

