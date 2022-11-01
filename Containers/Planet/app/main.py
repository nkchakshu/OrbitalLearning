# Import Needed Libraries
# import uvicorn
# import joblib
# import pandas as pd
import shutil
import socket
from fastapi import FastAPI, UploadFile, Depends
from pydantic import BaseModel
from loguru import logger
from fastapi.responses import FileResponse
import os
import asyncio
import Model_train_planet

# import tensorflow as tf
# import tensorflow_federated as tff


hostname = socket.gethostname()
ip_address = socket.gethostbyname(hostname)
access_key = os.environ['ACCESS_KEY']
# Initiate app instance
app = FastAPI(title='Orbital Learning', version='1.0',
              description='Orbital Learning powered by combination of Split Federated Learning, Swarm learning and '
                          'Reinforcement Learning')


# Benchmark Data to be used
class Benchmark(BaseModel):
    Data: float
    Version: float
    Batch: float
    # Preprocessing details may be required. Placeholder- for such details to be added.


class Project(BaseModel):
    ProjectID: str
    Version: float
    SatelliteID: str
    StageID: str
    StageLevel: str
    # StartDate: str
    # LastModified: str
    # LastUpdater: str
    # Key: str


class ProjectLaunch(BaseModel):
    ProjectID: str
    Version: float
    StageLevel: str


class ProjectExtras(BaseModel):
    ProjectID: str
    Field: str


class Path(BaseModel):
    path = str


# Initiate logging
log_format = "{time} | {level} | {message} | {file} | {line} | {function} | {exception}"
logger.add(sink='app/log_files/logs.log', format=log_format, level='DEBUG', compression='zip')


@app.on_event('startup')
async def app_startup():
    os.environ['PLANET_READY'] = "False"
    asyncio.create_task(Model_train_planet.model_train(True))


# Api root or home endpoint
@app.get('/')
@app.get('/home')
async def read_home():
    logger.debug('User checked the root page')
    return {'message': 'Planet Active. Refer to documentation to access different sections.'}


@app.get('/ready_state')
async def ready_state():
    ik = False
    while True:
        ik = os.environ['PLANET_READY']
        if ik == "True":
            break
    return str(ik)


# Provide a stored model stage in Planet.
@app.post('/get_stage')
async def get_stage(incoming_data: Project):
    new_data = incoming_data.dict()
    project_id = new_data['ProjectID']
    stage_id = new_data['StageID']
    stage_level = new_data["StageLevel"]
    version = new_data['Version']
    satellite_id = new_data['SatelliteID']
    if stage_level == 'c_stage':
        stage_path = '/mnt/VolumeLocal/model_zoo_local/' + str(project_id) + '_' + str(stage_id) + '_' + str(
            version) + '_' + str(satellite_id) + '.h5'
    elif stage_level == 's_stage':
        stage_path = '/mnt/VolumeLocal/model_zoo_local/SecondStage_' + str(project_id) + '_' + str(satellite_id) + '.h5'
    # if key in accepted_keys:
    if True:
        try:
            return FileResponse(stage_path)
        except:
            return {'message': 'Project_ID or the version does not exist'}
    else:
        return {'message': 'Unauthorized request.'}


# Update stored stage of a model in Planet.
@app.post('/update_stage')
async def update_stage(file: UploadFile, details: Project = Depends()):
    new_data = details.dict()
    project_id = new_data['ProjectID']
    stage_id = new_data['StageID']
    stage_level = new_data["StageLevel"]
    satellite_id = new_data['SatelliteID']
    version = new_data['Version']
    print(version)
    print(project_id)
    destination = '/mnt/VolumeLocal/model_zoo_local/'
    if stage_level == 'c_stage':
        destination = '/mnt/VolumeLocal/model_zoo_local/' + str(project_id) + '_' + str(stage_id) + '_' + str(
            version) + '_' + str(satellite_id) + '.h5'
    elif stage_level == 's_stage':
        destination = '/mnt/VolumeLocal/model_zoo_local/SecondStage_' + str(project_id) + '_' + str(satellite_id) + \
                      '.h5'
    #    print(accepted_keys)
    logger.debug('inside set_stage planet')
    # if new_data['Key'] in accepted_keys:
    if True:
        try:
            with open(destination, 'wb+') as buffer:
                await asyncio.to_thread(shutil.copyfileobj, file.file, buffer)
            return {"Result": "OK"}
        except Exception as e:
            logger.debug(e)
            logger.debug('error found here')
            return {"Result": "Stage not updated."}
        # finally:
        #     file.file.close()
        #     return {"Result": "OK"}
    # else:
    #     return {"Key Authentication Error": "Transaction invalid"}



@app.post('/launch_model')
async def launch_model(incoming_data: ProjectLaunch):
    new_data = incoming_data.dict()
    project_id = new_data['ProjectID']
    stage_level = new_data['StageLevel']
    version = new_data['Version']
    path = '/mnt/VolumeLocal/model_zoo_local/'
    if stage_level == 'p_stage':
        stage_path = 'Local_stages_proximal_' + str(project_id) + '.h5'
    elif stage_level == 'd_stage':
        stage_path = 'Local_stages_distal_' + str(project_id) + '.h5'
    elif stage_level == 's_stage':
        stage_path = 'SecondStage_' + str(project_id) + '.h5'
    return FileResponse(path + stage_path)


@app.post('/launch_extras')
async def launch_extras(incoming_data: ProjectExtras):
    new_data = incoming_data.dict()
    project_id = new_data['ProjectID']
    field = new_data['Field']
    path = '/mnt/Data/'
    stage_path = 'aa'
    if field == 'std':
        stage_path = 'sd.npy'
    elif field == 'mea':
        stage_path = 'mea.npy'
    return FileResponse(path + stage_path)