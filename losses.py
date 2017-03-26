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

"""Provides definitions for non-regularized training or test losses."""

import tensorflow as tf
try:
    from tensorflow.python.ops import ctc_ops
except ImportError:
    from tensorflow.contrib.ctc import ctc_ops

class BaseLoss(object):
  """Inherit from this class when implementing new losses."""

  def calculate_loss(self, unused_predictions, unused_labels, **unused_params):
    """Calculates the average loss of the examples in a mini-batch.

     Args:
      unused_predictions: a 2-d tensor storing the prediction scores, in which
        each row represents a sample in the mini-batch and each column
        represents a class.
      unused_labels: a 2-d tensor storing the labels, which has the same shape
        as the unused_predictions. The labels must be in the range of 0 and 1.
      unused_params: loss specific parameters.

    Returns:
      A scalar loss tensor.
    """
    raise NotImplementedError()

class CTCLoss(BaseLoss):
  """Calculate the CTC loss between the predictions and labels.

  The function calculates the loss in the following way: first we feed the
  predictions to the softmax activation function and then we calculate
  the minus linear dot product between the logged softmax activations and the
  normalized ground truth label.

  It is an extension to the one-hot label. It allows for more than one positive
  labels for each sample.
  """

  def calculate_loss(self, logits, targets, seq_len, **unused_params):
    """Implements ctc loss
    
    @param outputs: [batch,num_classes]
    @param targets: sparce tensor 
    @param seq_len: the length of the inputs sequences [batch]
    
    @returns: loss
    """
    

    with tf.name_scope('CTC-loss'):
            loss = tf.nn.ctc_loss(targets, logits,  seq_len)
            cost = tf.reduce_mean(loss)
            
    return cost

class CTCDecoder(object):
    
    def __init__(self, ctc_decoder='beam_search'):
        self.ctc_decoder = ctc_decoder
        
    def decode(self, predictions, seq_len,k):
        #print(target.get_shape().as_list(),'target')
        if self.ctc_decoder == 'greedy':
                decoded, log_prob = ctc_ops.ctc_greedy_decoder(predictions, seq_len)
        elif self.ctc_decoder == 'beam_search':
                decoded, log_prob = ctc_ops.ctc_beam_search_decoder(predictions, seq_len,top_paths=k)
        else:
                raise Exception("model type not supported: {}".format(self.ctc_decoder))

        return decoded
    
    def lebelRateError(self, decoded, target):
        return tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32),
                                              target))
    def useVocabulary(self, target):
        l = []
        for i in self.decoded:
            l.append(tf.edit_distance(tf.cast(i, tf.int32),
                                              target))
            #print(l[0],'is {}'.format(i))
        return tf.stack(l,0)