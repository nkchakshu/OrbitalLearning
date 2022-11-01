import numpy as np
import os
import tensorflow as tf
from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay
# from sklearn.preprocessing import MinMaxScaler
# import tensorflow_federated as tff
from loguru import logger
from scipy.stats import entropy
import glob


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
    return out

import collections


def flatten(x):
    if isinstance(x, collections.abc.Iterable):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]

def normalize(data):
    with open('/mnt/Data/sd.npy', 'rb') as f:
        std = np.load(f)
    with open('/mnt/Data/mea.npy', 'rb') as f:
        mea = np.load(f)
    data = (data - mea)/(2.0 * std)
    return data


def freeze_layers(model):
    for n in range(len(model.layers)):
        model.layers[n].trainable = False
    return model


def shannon_entropy(weights):

    # define distribution
    import itertools
    bhy = []

    weights = flatten(weights)
    logger.debug(weights)
    dist = [x / sum(weights) for x in weights]

    # use scipy to calculate entropy
    entropy_value = entropy(dist, base=2)

    return entropy_value


# Second stage is the only compulsory shared stage in all the satellites for a project. It will be active (trainable)
# only in the lowest orbit where either high frequency of training updates take place or a wide variety of data is
# available or both.
def second_stage(project_id):
    path = '/mnt/VolumeLocal/model_zoo_local/SecondStage_' + str(project_id) + '.h5'
    file_exists = os.path.exists(path)
    if file_exists:
        model = tf.keras.models.load_model(path)
        return model
    else:
        return FileNotFoundError


def second_stage_aux(project_id):
    path = '/mnt/VolumeLocal/model_zoo_local/SecondStage_*'

    aux_models = []
    for f in glob.glob(path):
        print(f)
        model = tf.keras.models.load_model(f)
        aux_models.append(model)
    return aux_models



# Local stages consists of both proximal and distal portions of the model.
def local_stages(project_id):
    proximal_path = '/mnt/VolumeLocal/model_zoo_local/Local_stages_proximal_' + str(project_id) + '.h5'
    file_exists_p = os.path.exists(proximal_path)
    if file_exists_p:
        proximal_model = tf.keras.models.load_model(proximal_path)
    else:
        return FileNotFoundError

    distal_path = '/mnt/VolumeLocal/model_zoo_local/Local_stages_distal_' + str(project_id) + '.h5'
    file_exists_d = os.path.exists(distal_path)
    if file_exists_d:
        distal_model = tf.keras.models.load_model(distal_path)
    else:
        return FileNotFoundError

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
        return FileNotFoundError


# If you want access to other metrics while training/testing change the compile parameter here.
# This function is for ensemble learning.
def assembled_rocket_aux(project_id, stage_ids, version, loss, active_correction_stage=None):
    p_stage, d_stage = local_stages(project_id)
    s_stage_all = second_stage_aux(project_id)
    c_stages = []
    assembled_models = []
    for i in range(len(s_stage_all)):
        s_stage = s_stage_all[i]
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
        assembled_models.append(assembled_stages)

    return assembled_models


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


# All satellites are provided with basic model from the beginning for a project.
async def rocket_predict(data_t, project_id, stage_ids, version='1.0', loss='categorical_crossentropy'):
    rocket = assembled_rocket(project_id, stage_ids, version, loss)
    logger.debug('inhula2')
    results = rocket.predict(data_t['X'])
    import matplotlib.pyplot as plt
    y_cal = tf.argmax(results, axis=1)
    y_true = tf.argmax(data_t['Y'], axis=1)
    y_cal = y_cal.numpy()
    y_true = y_true.numpy()
    ConfusionMatrixDisplay.from_predictions(y_true, y_cal)  # , normalize='true')#, display_labels = clf.classes_)
    # labels={0: 'NORM', 1: 'STTC', 2: 'CD', 3: 'HYP', 4: 'MI'}
    plt.savefig("benchmark_out.png", dpi=300)
    return results


# All satellites are provided with basic model from the beginning for a project. This function is for Ensemble Learning.
async def rocket_predict_ensemble(data_t, project_id, stage_ids, version='1.0', loss='categorical_crossentropy'):
    rockets = assembled_rocket_aux(project_id, stage_ids, version, loss)
    logger.debug('inhula2')
    results = []
    for z in range(len(rockets)):
        rocket = rockets[z]
        if z == 0:
            results = rocket.predict(data_t['X'])
        else:
            logger.debug(results)
            results += rocket.predict(data_t['X'])
    # results = results/float(len(rockets))
    import matplotlib.pyplot as plt
    y_cal = tf.argmax(results, axis=1)
    y_true = tf.argmax(data_t['Y'], axis=1)
    y_cal = y_cal.numpy()
    y_true = y_true.numpy()
    ConfusionMatrixDisplay.from_predictions(y_true, y_cal)  # , normalize='true')#, display_labels = clf.classes_)
    # labels={0: 'NORM', 1: 'STTC', 2: 'CD', 3: 'HYP', 4: 'MI'}
    plt.savefig("benchmark_out_ensemble.png", dpi=300)
    return results

# TODO: add stages into input
async def benchmark(project_id):
    path = '/mnt/Data/'
    data = {'X': np.load(path + str(project_id) + '_X_data.npy', allow_pickle=True),
            'Y': np.load(path + str(project_id) + '_Y_data.npy', allow_pickle=True)}
    # for ft in range(len(split_array)):
    data['X'] = normalize(data['X'])
    y_t = preprocess_ydata(data['Y'])
    data_batch = {'X': data['X'], 'Y': y_t}
    logger.debug('inhula1')
    b_res = await rocket_predict(data_t=data_batch, project_id=project_id, stage_ids=[])
    logger.debug('inhula3')
    return b_res

async def benchmark_ensemble(project_id):
    path = '/mnt/Data/'
    data = {'X': np.load(path + str(project_id) + '_X_data.npy', allow_pickle=True),
            'Y': np.load(path + str(project_id) + '_Y_data.npy', allow_pickle=True)}
    # for ft in range(len(split_array)):
    data['X'] = normalize(data['X'])
    y_t = preprocess_ydata(data['Y'])
    data_batch = {'X': data['X'], 'Y': y_t}
    logger.debug('inhula1')
    b_res = await rocket_predict_ensemble(data_t=data_batch, project_id=project_id, stage_ids=[])
    logger.debug('inhula3')
    return b_res

async def rocket_summary(project_id, stage_ids, version='1.0', loss='categorical_crossentropy'):
    rocket = assembled_rocket(project_id, stage_ids, version, loss)
    logger.debug(rocket.summary())
    rocket.summary(print_fn=myprint)
    return True


# TODO: add stages into input
async def summary(project_id):
    b_res = await rocket_summary(project_id=project_id, stage_ids=[])
    return b_res


def get_entropy(project_id, stage_ids, version='1.0', loss='categorical_crossentropy'):
    rocket = assembled_rocket(project_id, stage_ids, version, loss)
    weights = rocket.get_weights()
    logger.debug('Debigging entropy')
    logger.debug(weights[0])
    logger.debug(len(weights[0]))
    entropy = shannon_entropy(weights)
    return entropy


def myprint(s):
    with open('model_summary.txt', 'w') as f:
        print(s, file=f)