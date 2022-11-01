import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.constraints import min_max_norm
# import tensorflow_federated as tff
from time import time
import asyncio
from loguru import logger

gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1512)])
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
  except RuntimeError as e:
    print(e)


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


def freeze_layers(model):
    for n in range(len(model.layers)):
        model.layers[n].trainable = False
    return model


def divide_chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


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


def normalize(data):
    std = np.std(data)
    mea = np.mean(data)
    data = (data - mea)/(2.0 * std)
    return data, std, mea

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
        proximal_model = tf.keras.models.load_model(proximal_path)
    else:
        proximal_model = tf.keras.models.Sequential(name='p_stage')
        # proximal_model.add(tf.keras.layers.LSTM(128, input_shape=(1000, 12), return_sequences=True))
        # proximal_model.add(tf.keras.layers.GRU(64, return_sequences=True))
        # # proximal_model.add(tf.keras.layers.Dropout(0.4))
        # # proximal_model.add(tf.keras.layers.GRU(64, return_sequences=True))
        # proximal_model.add(tf.keras.layers.Dropout(0.4))
        # proximal_model.add(tf.keras.layers.GRU(64, return_sequences=True))
        # proximal_model.add(tf.keras.layers.Dropout(0.4))
        # proximal_model.add(tf.keras.layers.GRU(32, return_sequences=True))
        # proximal_model.add(tf.keras.layers.Dropout(0.4))
        # proximal_model.add(tf.keras.layers.GRU(32))
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
        distal_model.add(tf.keras.layers.Dense(64, activation='relu'))#, kernel_constraint=min_max_norm(min_value=0.2, max_value=1.0, rate=0.9)))
        distal_model.add(tf.keras.layers.Dense(5, activation='softmax'))#, kernel_constraint=min_max_norm(min_value=0.2, max_value=1.0, rate=0.9)))

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
    assembled_stages.add(s_stage)
    assembled_stages.add(d_stage)
    assembled_stages.compile(loss=loss, optimizer='adam', metrics=['accuracy'])

    return assembled_stages


def save_stages(assembled_stages, project_id, stage_ids, version):
    path = '/mnt/VolumeLocal/model_zoo_local/'
    p_stage = assembled_stages.get_layer('p_stage')
    # p_stage = freeze_layers(p_stage)
    p_stage.save(path + 'Local_stages_proximal_' + str(project_id) + '.h5')
    s_stage = assembled_stages.get_layer('s_stage')
    # s_stage = freeze_layers(s_stage)
    s_stage.save(path + 'SecondStage_' + str(project_id) + '.h5')
    if stage_ids:
        for i in stage_ids:
            c_stage = assembled_stages.get_layer(str(i))
            c_stage.save(path + str(project_id) + '_' + str(i) + '_' + str(version) + '.h5')
    d_stage = assembled_stages.get_layer('d_stage')
    # d_stage = freeze_layers(d_stage)
    d_stage.save(path + 'Local_stages_distal_' + str(project_id) + '.h5')
    return True


# All satellites are provided with basic model from the beginning for a project.
async def train_rocket(data_t, project_id, stage_ids, epochs=50, cl_wts=None, version='1.0', loss='categorical_crossentropy'):

    rocket = assembled_rocket(project_id, stage_ids, version, loss)
    logger.debug('inT')
    rocket.fit(data_t['X'], data_t['Y'], epochs=epochs, class_weight=cl_wts, verbose=1)
    logger.debug('inT2')
    save_stages(rocket, project_id, stage_ids, version)
    logger.debug('inT3')
    return True


async def model_train(train_flag):
    path = '/mnt/Data/'
    if train_flag:
        stage_ids = []
        data = {'X': np.load(path + 'Base_X_data.npy', allow_pickle=True),
                'Y': np.load(path + 'Base_Y_data.npy', allow_pickle=True)}
        logger.debug('hmmm')
        data['X'], sd, mea = normalize(data['X'])
        with open(path + 'sd.npy', 'wb') as fx:
            np.save(fx, sd)
        with open(path + 'mea.npy', 'wb') as f1:
            np.save(f1, mea)
        data_batch = {'X': data['X'], 'Y': data['Y']}  # [split_array[ft]],
        data_batch['Y'], k_drt = preprocess_ydata(data_batch['Y'])
        cl_wts = weights_calc(k_drt, 5)
        logger.debug('~T')
        await train_rocket(data_batch, '1.0', stage_ids, cl_wts=cl_wts, loss='categorical_crossentropy')
    os.environ['PLANET_READY'] = "True"
    return True

