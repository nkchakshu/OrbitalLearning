# Import Needed Libraries
import uvicorn
import joblib
import pandas as pd
import socket
import shutil
import time
from fastapi import FastAPI, UploadFile, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel
# from test_ml_api import get_predictions
from loguru import logger
import os
import asyncio
import Model_train
import Model_predict
from scipy import linalg as la
import numpy as np
import tensorflow as tf
# import tensorflow_federated as tff

hostname = socket.gethostname()
ip_address = socket.gethostbyname(hostname)
planet_address = 'http://0.0.0.0:' + os.environ['PLANET_PORT']  # TODO: Planet port env variable needs to be set.
access_key = os.environ['ACCESS_KEY']  # Currently access key is the same for an orbital system,
# controlled by the planet.
accepted_keys = []
StagesInUse = []
OrbitID = 0.0
OrbitEntropy = 1.0
OrbitHeight = 1.0
# Initiate app instance
app = FastAPI(title='Medical informatics FL experiment', version='1.0',
              description='Client_node_medical_FL_experiment_API')


# Design the incoming feature data
class Features(BaseModel):
    variance_of_wavelet: float
    skewness_of_wavelet: float
    curtosis_of_wavelet: float
    entropy_of_wavelet: float
    retrain_model_: int


# Benchmark Data to be used
class Benchmark(BaseModel):
    Data: float
    Version: float
    Batch: float
    # Preprocessing details may be required. Placeholder- for such details to be added.


class ProjectExtras(BaseModel):
    ProjectID: str
    Field: str


class Status(BaseModel):
    Stages = list
    OrbitID: str
    OrbitEntropy: float
    OrbitHeight: int


class Project(BaseModel):
    ProjectID: str
    Version: float
    StageID: str
    StageLevel: str
    # StartDate: str
    # LastModified: str
    # LastUpdater: str
    # Key: str


class ProjectID(BaseModel):
    ProjectID: str


class RTJ(BaseModel):
    RTJ: str


class SecureKeys(BaseModel):
    AcceptedKeys: list
    AccessKey: str


# Initiate logging
log_format = "{time} | {level} | {message} | {file} | {line} | {function} | {exception}"
logger.add(sink='app/log_files/logs.log', format=log_format, level='DEBUG', compression='zip')


rtp = Model_train
@app.on_event('startup')
async def app_startup():
    global rtp
    rtp = Model_train
    asyncio.create_task(rtp.model_train())



# Api root or home endpoint
@app.get('/')
@app.get('/home')
async def read_home():
    logger.debug('User checked the root page')
    return {'message': 'Fake or Not API live!'}


# Security update endpoint
@app.post('/secure')
async def update_keys(incoming_data: SecureKeys):
    new_data = incoming_data.dict()
    logger.debug(access_key)
    logger.debug(new_data["AccessKey"])
    if new_data["AccessKey"] == access_key:
        global accepted_keys
    accepted_keys = new_data["AcceptedKeys"]
    logger.debug(accepted_keys)


# Prediction endpoint
# @app.post('/predict')
# @logger.catch()  # catch any unexpected breaks
# async def get_prediction(incoming_data: Features):
#     # retrieve incoming json data as a dictionary
#     new_data = incoming_data.dict()
#     logger.info('User sent some data for predictions')
#
#     # Make predictions based on the incoming data and saved neural net
#     preds, train_output, model_state = get_predictions(new_data)
#     logger.debug('Predictions successfully generated for the user')
#     logger.debug(preds)
#     logger.debug(train_output)
#     logger.debug(model_state)
#
#     # Return the predicted class and the predicted probability
#     return {'Status': 'Model training completed', 'training_metrics(loss)': str(train_output),
#             'predicted_class': round(float(preds.flatten())), 'predicted_prob': float(preds.flatten())}

@app.post('/benchmark')
async def benchmark(incoming_data: ProjectID):
    new_data = incoming_data.dict()
    project_id = new_data['ProjectID']
    logger.debug('projectid_check')
    logger.debug(project_id)
    results = await Model_predict.benchmark(project_id=project_id)
    logger.debug('inhula4')
    file_path = "./benchmark_out.png"
    try:
        return FileResponse(file_path, media_type="image/png", filename="Benchmark_out.jpg")
    except:
        return {"error": "File not found!"}


@app.post('/benchmark_ensemble')
async def benchmark_ensemble(incoming_data: ProjectID):
    new_data = incoming_data.dict()
    project_id = new_data['ProjectID']
    logger.debug('projectid_check')
    logger.debug(project_id)
    results = await Model_predict.benchmark_ensemble(project_id=project_id)
    logger.debug('inhula4')
    file_path = "./benchmark_out_ensemble.png"
    try:
        return FileResponse(file_path, media_type="image/png", filename="Benchmark_out_ensemble.jpg")
    except:
        return {"error": "File not found!"}

@app.post('/model_summary')
async def model_summary(incoming_data: ProjectID):
    new_data = incoming_data.dict()
    project_id = new_data['ProjectID']
    logger.debug('projectid_check')
    logger.debug(project_id)
    results = await Model_predict.summary(project_id=project_id)
    logger.debug('inhula4')
    file_path = "./model_summary.txt"
    try:
        return FileResponse(file_path, media_type="text/plain", filename="model_summary.txt")
    except:
        return {"error": "File not found!"}


# Provide a stored model stage in satellite.
@app.post('/get_stage')
async def get_stage(incoming_data: Project):
    new_data = incoming_data.dict()
    project_id = new_data['ProjectID']
    stage_id = new_data['StageID']
    stage_level = new_data["StageLevel"]
    version = new_data['Version']
    # key = new_data["Key"]
    if stage_level == 'c_stage':
        stage_path = '/mnt/VolumeLocal/model_zoo_local/' + str(project_id) + '_' + str(stage_id) + '_' + str(
                    version) + '.h5'
    elif stage_level == 's_stage':
        stage_path = '/mnt/VolumeLocal/model_zoo_local/SecondStage_' + str(project_id) + '.h5'
    # if key in accepted_keys:
    if True:
        try:
            return FileResponse(stage_path)
        except:
            return {'message': 'Project_ID or the version does not exist'}
    else:
        return {'message': 'Unauthorized request.'}

@app.get('/get_azimuth')
async def get_azimuth():
    global rtp
    t = rtp.time_logs()
    return t


@app.post('/set_stage')
async def set_stage(file: UploadFile, details: Project = Depends()):
    logger.debug('recieveing something ' + str(time.time()))
    new_data = details.dict()
    project_id = new_data['ProjectID']
    stage_id = new_data['StageID']
    stage_level = new_data["StageLevel"]
    version = new_data['Version']
    print(version)
    print(project_id)
    if stage_level == 'c_stage':
        destination = '/mnt/VolumeLocal/model_zoo_local/' + str(project_id) + '_' + str(stage_id) + '_' + str(
            version) + '.h5'
    elif stage_level == 's_stage':
        logger.debug('recieveing ss_stage ' + str(time.time()))
        destination = '/mnt/VolumeLocal/model_zoo_local/SecondStage_' + str(project_id) + '.h5'
    elif stage_level == 'p_stage':
        logger.debug('recieveing p_stage ' + str(time.time()))
        destination = '/mnt/VolumeLocal/model_zoo_local/Local_stages_proximal_' + str(project_id) + '.h5'
    elif stage_level == 'd_stage':
        logger.debug('recieveing d_stage ' + str(time.time()))
        destination = '/mnt/VolumeLocal/model_zoo_local/Local_stages_distal_' + str(project_id) + '.h5'

    print(accepted_keys)
    logger.debug('inside set_stage')
    # if new_data['Key'] in accepted_keys:
    if True:
        try:
            with open(destination, 'wb+') as buffer:
                await asyncio.to_thread(shutil.copyfileobj, file.file, buffer)
            return {"Result": "OK"}
        except Exception as e:
            print(e)
            logger.debug('error found here')
            return {"Result": "Stage not updated."}
        # finally:
        #     file.file.close()
        #     return {"Result": "OK"}
    else:
        return {"Key Authentication Error: Transaction invalid."}


@app.post('/set_aux_stage')
async def set_aux_stage(file: UploadFile, details: Project = Depends()):
    logger.debug('recieveing aux ' + str(time.time()))
    new_data = details.dict()
    project_id = new_data['ProjectID']
    stage_id = new_data['StageID']
    stage_level = new_data["StageLevel"]
    version = new_data['Version']
    print(version)
    print(project_id)
    if stage_level == 'c_stage':
        destination = '/mnt/VolumeLocal/model_zoo_local/' + str(project_id) + '_' + str(stage_id) + '_' + str(
            version) + '.h5'
    elif stage_level == 's_stage':
        logger.debug('recieveing ss_stage ' + str(time.time()))
        destination = '/mnt/VolumeLocal/model_zoo_local/SecondStage_' + str(project_id) + '_' + str(stage_id) + '.h5'
    # elif stage_level == 'p_stage':
    #     logger.debug('recieveing p_stage ' + str(time.time()))
    #     destination = '/mnt/VolumeLocal/model_zoo_local/Local_stages_proximal_' + str(project_id) + '.h5'
    # elif stage_level == 'd_stage':
    #     logger.debug('recieveing d_stage ' + str(time.time()))
    #    destination = '/mnt/VolumeLocal/model_zoo_local/Local_stages_distal_' + str(project_id) + '.h5'

    print(accepted_keys)
    logger.debug('inside set_stage')
    # if new_data['Key'] in accepted_keys:
    if True:
        try:
            with open(destination, 'wb+') as buffer:
                await asyncio.to_thread(shutil.copyfileobj, file.file, buffer)
            return {"Result": "OK"}
        except Exception as e:
            print(e)
            logger.debug('error found here')
            return {"Result": "Stage not updated."}
        # finally:
        #     file.file.close()
        #     return {"Result": "OK"}
    else:
        return {"Key Authentication Error: Transaction invalid."}

# TODO: This needs work taking only second stage at the moment.
@app.post('/get_entropy')
async def get_entropy(incoming_data: ProjectID):
        # new_data = incoming_data.dict()
        # project_id = new_data['ProjectID']
        # active_stage = '/mnt/VolumeLocal/model_zoo_local/SecondStage_' + str(project_id) + '.h5'
        # model = tf.keras.models.load_model(active_stage)
        # WeightMatrix = model.get_weights()
        # r = WeightMatrix * (la.logm(WeightMatrix) / la.logm(np.matrix([[2]])))
        # # s = -np.matrix.trace(r)
        # global rtp
        # t = rtp.entropy_logs()
        # return t
        s = 1.0  # TODO: Remove this after debugging.
        return s


@app.get('/get_rtj')
async def get_rtj():
        return os.environ["RTJ"]


@app.post('/set_rtj')
async def set_rtj(incoming_data: RTJ):
        new_data = incoming_data.dict()
        status = new_data['RTJ']
        logger.debug('Testing_setrtj')
        logger.debug(status)
        os.environ["RTJ"] = status


@app.post('/satellite_status')
async def set_satellite_stats(incoming_data: Status):
    new_data = incoming_data.dict()
    global StagesInUse
    StagesInUse = new_data["Stages"]
    global OrbitID
    OrbitID = new_data["OrbitID"]
    global OrbitEntropy
    OrbitEntropy = new_data["OrbitEntropy"]
    global OrbitHeight
    OrbitHeight = new_data["OrbitHeight"]


@app.post('/get_shannon_entropy')
async def get_shannon_entropy(incoming_data: ProjectID):
    global rtp
    t = rtp.entropy_logs()
    return t
    # return {'Entropy': str(entropy)}


@app.post('/set_extras')
async def set_extras(file: UploadFile, details: ProjectExtras = Depends()):
    new_data = details.dict()
    project_id = new_data['ProjectID']
    field = new_data['Field']
    logger.debug('extra setting')
    logger.debug(field)
    path = '/mnt/Data/'
    stage_path = 'aa'
    if field == 'std':
        stage_path = 'sd.npy'
        logger.debug('sd recieved')
    elif field == 'mea':
        stage_path = 'mea.npy'
        logger.debug('mea recieved')
    # if new_data['Key'] in accepted_keys:
    if True:
        try:
            with open(path+stage_path, 'wb+') as buffer:
                await asyncio.to_thread(shutil.copyfileobj, file.file, buffer)
            return {"Result": "OK"}
        except Exception as e:
            print(e)
            logger.debug('error found here')
            return {"Result": "Stage not updated."}
        # finally:
        #     file.file.close()
        #     return {"Result": "OK"}
    else:
        return {"Key Authentication Error: Transaction invalid."}

# # Benchmark endpoint
# @app.post('/benchmark')
# @logger.catch()
# async def benchmark(incoming_data: Benchmark):
#     benchmark_data = incoming_data.dict()
#     logger.info('Performance benchmarking has been been requested.')
#     logger.info(str(benchmark_data['Data'] + '-' + str(benchmark_data['Version']) + '-' + str(benchmark_data['Batch'])))
#
#     preds, train_output, model_state = get_predictions(benchmark_data)
#     logger.debug('Predictions successfully generated for the user')
#     logger.debug(preds)
#     logger.debug(train_output)
#     logger.debug(model_state)
#
#     # Return the predicted class and the predicted probability
#     return {'Status': 'Model training completed', 'training_metrics(loss)': str(train_output),
#             'predicted_class': round(float(preds.flatten())), 'predicted_prob': float(preds.flatten())}


# async def train_model(project_id, stages):
#     current_path = os.getcwd()
#     os.chdir('\mnt\VolumeLocal\codes')
#     import train_model
#     train_model.train(project_id, stages)
#
#     os.chdir(current_path)

# if __name__ == "__main__":
#    ip_inp = "0.0.0.0"
# Run app with uvicorn with port and host specified. Host needed for docker port mapping
#    uvicorn.run(app, host=ip_inp, port=8002)
