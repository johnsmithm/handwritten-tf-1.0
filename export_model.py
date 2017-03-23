# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities to export a model for batch prediction."""

import tensorflow as tf
import tensorflow.contrib.slim as slim

from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils as saved_model_utils


from losses import CTCDecoder

class ModelExporter(object):

  def __init__(self, slices_features, model, reader):
    self.slices_features = slices_features
    self.model = model
    self.reader = reader

    with tf.Graph().as_default() as graph:
      self.inputs, self.outputs = self.build_inputs_and_outputs()
      self.graph = graph
      self.saver = tf.train.Saver(tf.trainable_variables(), sharded=True)

  def export_model(self, model_dir, global_step_val, last_checkpoint):
    """Exports the model so that it can used for batch predictions."""

    with self.graph.as_default():
      with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        self.saver.restore(session, last_checkpoint)

        signature = signature_def_utils.build_signature_def(
            inputs=self.inputs,
            outputs=self.outputs,
            method_name=signature_constants.PREDICT_METHOD_NAME)

        signature_map = {signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: 
                         signature}

        model_builder = saved_model_builder.SavedModelBuilder(model_dir)
        model_builder.add_meta_graph_and_variables(session,
            tags=[tag_constants.SERVING],
            signature_def_map=signature_map,
            clear_devices=True)
        model_builder.save()

  def build_inputs_and_outputs(self):

    if not self.slices_features:

      serialized_examples = tf.placeholder(tf.string, shape=(None,))

      fn = lambda x: self.build_prediction_graph(x)
      decodedPrediction = (
          tf.map_fn(fn, serialized_examples, 
                    dtype=(tf.int32)))

    else:

      serialized_examples = tf.placeholder(tf.string, shape=(None,))

      decodedPrediction = (
          self.build_prediction_graph(serialized_examples))

    inputs = {"example_bytes": 
              saved_model_utils.build_tensor_info(serialized_examples)}

    outputs = {"predictions": saved_model_utils.build_tensor_info(decodedPrediction)}

    return inputs, outputs

  def build_prediction_graph(self, serialized_examples):    

    imageInput, seq_len , target =   self.reader.prepare_serialized_examples(serialized_examples)
    
    #feature_dim = len(model_input_raw.get_shape()) - 1
    #model_input = tf.nn.l2_normalize(model_input_raw, feature_dim)
    seq_len = tf.cast(seq_len, tf.int32)    
    target = tf.cast(target, tf.int32)
    batch_size = tf.shape(imageInput)[0]
    seq_len = tf.reshape(seq_len, [batch_size])

    with tf.name_scope("model"):
      result = self.model.create_model(
          imageInput,
          seq_len=seq_len,
          vocab_size=self.reader.num_classes,
          target=target,
          
          is_training=False)

      for variable in slim.get_model_variables():
        tf.summary.histogram(variable.op.name, variable)

      predictions = result["predictions"]
      #decoder = CTCDecoder('beam_search')
      #decodedPrediction, ler = decoder.decode(predictions, target, seq_len,1)
        
    return  predictions
