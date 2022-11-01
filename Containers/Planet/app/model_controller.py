import tensorflow as tf
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from loguru import logger
from Training_file import train_client, client_data, model_fn
import tensorflow as tf
import tensorflow_federated as tff


@logger.catch
def get_predictions(data):

    new_data = {k: [v] for k, v in data.items()}

    new_data_df = pd.DataFrame.from_dict(new_data)
    logger.debug(new_data_df)

    new_data_dfx = new_data_df[['variance_of_wavelet', 'skewness_of_wavelet',
                               'curtosis_of_wavelet', 'entropy_of_wavelet']]

    # Load saved standardising object
    scaler = joblib.load('scaler.joblib')
    logger.debug('Saved standardising object successfully loaded')

    # retrain model if required
    output_train = 'no training required'
    if int(new_data_df['retrain_model_'])==1:
        logger.debug('excess1_test_ml_api')
        model_state, output_train = train_client()
    else:
        model_state=0

    # Load saved keras model
    model = tf.keras.models.load_model('planet_check.h5')
    logger.debug('Saved ANN model loaded successfully')

    # Scale new data using the loaded object
    X = scaler.transform(new_data_dfx.values)
    logger.debug('Incoming data successfully standardised with saved object')

    # Make new predictions from the newly scaled data and return this prediction
    preds = model.predict(X)

    return preds, output_train, model_state

