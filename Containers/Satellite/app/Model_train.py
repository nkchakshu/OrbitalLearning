import numpy as np
import os
import tensorflow as tf
# import tensorflow_federated as tff
from time import time
import asyncio
from loguru import logger
import collections
from scipy import signal, stats

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1512)])
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
  except RuntimeError as e:
    print(e)

timestamps = []
pearson = []

def weights_calc(d, classes):
    g = {}
    g_range = range(classes)
    max_element = max(d, key=d.count)
    count_m = d.count(max_element)
    for u in list(set(g_range)):
        if d.count(u) != 0:
            w = (len(d)/(5 * d.count(u)))
        else:
            w = 0.01
        g[int(u)] = w
    return g


def flatten(x):
    if isinstance(x, collections.abc.Iterable):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]


def freeze_layers(model):
    for n in range(len(model.layers)):
        model.layers[n].trainable = False
    return model


def divide_chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def normalize(data):
    with open('/mnt/Data/sd.npy', 'rb') as f:
        std = np.load(f)
    with open('/mnt/Data/mea.npy', 'rb') as f:
        mea = np.load(f)
    data = (data - mea)/(2.0 * std)
    return data


def preprocess_ydata(data):
    keys = {'STTC': 1, '0': 0, 'CD': 2, 'NORM': 0, 'HYP': 3, 'MI': 4}
    l_drt = []
    for i in list(data):
        if i:
            l_drt.append(i[0])
        else:
            l_drt.append('0')
    k_drt = []
    for d in l_drt:
        k_drt.append(keys[d])
    out = np.zeros([len(k_drt), 5])
    for m, o in enumerate(k_drt):
        out[int(m), int(o)] = 1
    return out, k_drt


log_format = "{time} | {level} | {message} | {file} | {line} | {function} | {exception}"
logger.add(sink='app/log_files/logs_x.log', format=log_format, level='DEBUG', compression='zip')

# Second stage is the only compulsory shared stage in all the satellites for a project. It will be active (trainable)
# only in the lowest orbit where either high frequency of training updates take place or a wide variety of data is
# available or both.


def second_stage(project_id):
    path = '/mnt/VolumeLocal/model_zoo_local/SecondStage_' + str(project_id) + '.h5'
    file_exists = os.path.exists(path)
    if file_exists:
        print('exists')
        model = tf.keras.models.load_model(path)
        return model
    else:
        model = tf.keras.models.Sequential(name='s_stage')
        # model.add(tf.keras.layers.GRU(32, return_sequences=True))
        # model.add(tf.keras.layers.Dropout(0.3))
        # # model.add(tf.keras.layers.GRU(64, return_sequences=True))
        # # model.add(tf.keras.layers.Dropout(0.3))
        # # model.add(tf.keras.layers.GRU(32, return_sequences=True))
        # # model.add(tf.keras.layers.Dropout(0.3))
        # model.add(tf.keras.layers.GRU(32))
        # # model.add(tf.keras.layers.Dense(256, activation='relu'))#, kernel_constraint=min_max_norm(min_value=0.2, max_value=1.0, rate=0.9)))
        # # model.add(tf.keras.layers.Dropout(0.5))
        # # model.add(tf.keras.layers.Dense(128, activation='relu'))#, kernel_constraint=min_max_norm(min_value=0.2, max_value=1.0, rate=0.9)))
        # # model.add(tf.keras.layers.Dropout(0.2))
        # # model.add(tf.keras.layers.Dense(64, activation='relu'))
        # # model.add(tf.keras.layers.Dropout(0.2))
        # model.add(tf.keras.layers.Dense(128, activation='relu'))
        # model.add(tf.keras.layers.Dropout(0.2))
        # model.add(tf.keras.layers.Dense(128, activation='relu'))
        # model.add(tf.keras.layers.Dropout(0.2))
        # model.add(tf.keras.layers.Dense(128, activation='relu'))
        # model.add(tf.keras.layers.Dropout(0.2))

        model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
        model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'))
        model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'))
        model.add(tf.keras.layers.GlobalAveragePooling1D())
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.3))
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.3))

        return model


# Local stages consists of both proximal and distal portions of the model.
def local_stages(project_id):
    n_timesteps = 1000
    n_features = 12
    proximal_path = '/mnt/VolumeLocal/model_zoo_local/Local_stages_proximal_' + str(project_id) + '.h5'
    file_exists_p = os.path.exists(proximal_path)
    if file_exists_p:
        logger.debug('proximal Hai')
        proximal_model = tf.keras.models.load_model(proximal_path)
    else:
        logger.debug('proximal nahi Hai re baba!')
        proximal_model = tf.keras.models.Sequential(name='p_stage')
        # proximal_model.add(tf.keras.layers.LSTM(128, input_shape=(1000, 12), return_sequences=True))
        # proximal_model.add(tf.keras.layers.GRU(64, return_sequences=True))
        # #proximal_model.add(tf.keras.layers.Dropout(0.4))
        # proximal_model.add(tf.keras.layers.GRU(64, return_sequences=True))
        # #proximal_model.add(tf.keras.layers.Dropout(0.4))
        proximal_model.add(tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu',
                                                  input_shape=(n_timesteps, n_features)))
        proximal_model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'))
        proximal_model.add(tf.keras.layers.Dropout(0.5))

    distal_path = '/mnt/VolumeLocal/model_zoo_local/Local_stages_distal_' + str(project_id) + '.h5'
    file_exists_d = os.path.exists(distal_path)
    if file_exists_d:
        distal_model = tf.keras.models.load_model(distal_path)
    else:
        distal_model = tf.keras.models.Sequential(name='d_stage')
        # distal_model.add(tf.keras.layers.Dense(512, activation='relu'))
        # distal_model.add(tf.keras.layers.Dropout(0.5))
        distal_model.add(tf.keras.layers.Dense(128, activation='relu'))
        # distal_model.add(tf.keras.layers.Dropout(0.2))
        distal_model.add(tf.keras.layers.Dense(64, activation='relu'))
        distal_model.add(tf.keras.layers.Dense(5, activation='softmax'))

    return proximal_model, distal_model


# all correction stage have fixed dimensions in this version of the code.
def correction_stage(project_id, stage_id, version):
    path = '/mnt/VolumeLocal/model_zoo_local/' + str(project_id) + '_' + str(stage_id) + '_' + str(
        version) + '.h5'
    file_exists = os.path.exists(path)
    if file_exists:
        model = tf.keras.models.load_model(path)
        return model
    else:
        model = tf.keras.models.Sequential(name=stage_id)
        model.add(tf.keras.layers.Dense(10, activation='relu'))
        model.add(tf.keras.layers.Dense(10, activation='relu'))
        return model


# If you want access to other metrics while training/testing change the compile parameter here.
def assembled_rocket(project_id, stage_ids, version, loss, active_correction_stage=None):
    p_stage, d_stage = local_stages(project_id)
    s_stage = second_stage(project_id)
    c_stages = []
    assembled_stages = tf.keras.models.Sequential()
    assembled_stages.add(p_stage)
    if stage_ids:
        s_stage = freeze_layers(s_stage)
        assembled_stages.add(s_stage)
        for i in stage_ids:
            stage = correction_stage(project_id, str(i), version)
            if str(i) != str(active_correction_stage):
                stage = freeze_layers(stage)
            assembled_stages.add(stage)
    else:
        assembled_stages.add(s_stage)
    assembled_stages.add(d_stage)
    assembled_stages.compile(loss=loss, optimizer='adam', metrics=['accuracy'])

    return assembled_stages


def save_stages(assembled_stages, project_id, stage_ids, version):
    path = '/mnt/VolumeLocal/model_zoo_local/'
    p_stage = assembled_stages.get_layer('p_stage')
    p_stage.save(path + 'Local_stages_proximal_' + str(project_id) + '.h5')
    s_stage = assembled_stages.get_layer('s_stage')
    s_stage.save(path + 'SecondStage_' + str(project_id) + '.h5')
    if stage_ids:
        for i in stage_ids:
            c_stage = assembled_stages.get_layer(str(i))
            c_stage.save(path + str(project_id) + '_' + str(i) + '_' + str(version) + '.h5')
    d_stage = assembled_stages.get_layer('d_stage')
    d_stage.save(path + 'Local_stages_distal_' + str(project_id) + '.h5')
    return True


# All satellites are provided with basic model from the beginning for a project.
async def train_rocket(data_t, project_id, stage_ids, epochs=100, cl_wts=None, version='1.0', loss='categorical_crossentropy'):

    rocket = assembled_rocket(project_id, stage_ids, version, loss)
    logger.debug('inT')
    # s_stage_local_before = rocket.get_layer('s_stage')
    s_stage_local_before = rocket.get_weights()
    rocket.fit(data_t['X'], data_t['Y'], epochs=epochs, class_weight=cl_wts, verbose=1)
    # s_stage_local_after = rocket.get_layer('s_stage')
    s_stage_local_after = rocket.get_weights()
    s_stage_local_before = flatten(s_stage_local_before)
    s_stage_local_after = flatten(s_stage_local_after)
    global pearson
    # p_o = signal.correlate(s_stage_local_before, s_stage_local_after, mode='valid', method='fft')
    p_o = stats.pearsonr(s_stage_local_before, s_stage_local_after)
    pearson.append(float(p_o[0]))
    logger.debug('inT2')
    save_stages(rocket, project_id, stage_ids, version)
    logger.debug('inT3')
    return True


def entropy_logs():
    global pearson
    if len(pearson)>=1:
        return float(pearson[-1])
    else:
        return 1.0


def time_logs():
    global timestamps
    if len(timestamps)>=2:
        return abs(timestamps[-1]-timestamps[-2])
    else:
        return 200.0

async def model_train():
    TimeInterval = 20
    path = '/mnt/Data/'
    chunk_size = 10
    while True:
        await asyncio.sleep(TimeInterval - time() % TimeInterval)
        file_ex = os.path.exists(path + 'Pending_X_data.npy')
        rtj_flag = os.environ['RTJ']
        rtj_id = os.environ['ID_SAT']
        if file_ex and rtj_flag == 'False':
                global timestamps
                timestamps.append(time())
                # while True:
                #     flag = np.load(path + 'open_flag.npy')
                #     if flag:
                #         sleep(10)
                #     else:
                #         break
                # f_flag = open(path + 'open_flag.npy', 'wb')
                # flag = True
                # np.save(f_flag, flag)
                # stage_ids = np.load(path + 'StagesInUse.npy')
                stage_ids = []
                # pats = np.load(path + 'Pending_patients_OL.npy')
                # split_array = np.array_split(pats, math.ceil(pats.size/chunk_size))
                data = {'X': np.load(path + 'Pending_X_data.npy', allow_pickle=True),
                        'Y': np.load(path + 'Pending_Y_data.npy', allow_pickle=True)}
                # for ft in range(len(split_array)):
                logger.debug(rtj_id)

                if rtj_id == '1':
                    noise = np.random.normal(0,0.5,data['X'].shape)
                    data['X'] += noise
                    logger.debug(noise)
                data['X'] = normalize(data['X'])
                data_batch = {'X': data['X'], 'Y': data['Y']}  # [split_array[ft]],
                data_batch['Y'], k_drt = preprocess_ydata(data_batch['Y'])
                cl_wts = weights_calc(k_drt, 5)
                logger.debug('~T')
                await train_rocket(data_batch, '1.0', stage_ids, cl_wts=cl_wts, loss='categorical_crossentropy') #  cl_wts=cl_wts
                os.remove(path + 'Pending_X_data.npy')
                os.remove(path + 'Pending_Y_data.npy')
                os.environ['RTJ'] = 'BBlink'
                # flag = False
                # np.save(f_flag, flag)
                # f_flag.close()


