import tensorflow as tf
import tensorflow_federated as tff
from loguru import logger

# Load simulation data.

def client_data(n):
  source, _ = tff.simulation.datasets.emnist.load_data()
  return source.create_tf_dataset_for_client(source.client_ids[n]).map(
      lambda e: (tf.reshape(e['pixels'], [-1]), e['label'])
  ).repeat(10).batch(20)

# Pick a subset of client devices to participate in training.


train_data = [client_data(n) for n in range(3)]
# Wrap a Keras model for use with TFF.
def model_fn():

  model = tf.keras.models.Sequential([
      tf.keras.layers.Dense(10, tf.nn.softmax, input_shape=(784,),
                            kernel_initializer='zeros')
  ])
  return tff.learning.from_keras_model(
      model,
      input_spec=train_data[0].element_spec,
      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])


# Simulate a few rounds of training with the selected client devices.
def train_client():
    tff.backends.native.set_local_python_execution_context()
    #modelx = model_fn()
    logger.debug('train_client_1')
    trainer = tff.learning.build_federated_averaging_process(model_fn,
            client_optimizer_fn=lambda: tf.keras.optimizers.SGD(0.1))
    #print(trainer.__dict__)
    logger.debug('train_client_2')
    state = trainer.initialize()
    logger.debug('train_client_3')
    for _ in range(100):
        logger.debug('run')
        state, metrics = trainer.next(state, train_data)

    #lop = tff.learning.models.functional_model_from_keras(model_fn, )
    #print(state.__dict__)
    #tff.learning.assign_weights_to_keras_model(modelx, state.model)
    #tff.learning.models.save(modelx, '.')
    return state.model, metrics['train']['loss']

