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

"""Contains model definitions."""
import math

import models
import tensorflow as tf
import utils

from tensorflow import flags
import tensorflow.contrib.slim as slim

FLAGS = flags.FLAGS
flags.DEFINE_integer(
    "hidden", 20,
    "The number neurons in the recurrent cell")
flags.DEFINE_integer(
    "layers", 2,
    "The number of layers.")
flags.DEFINE_integer(
    "input_chanels", 1,
    "The number of images as input for an image.")
flags.DEFINE_string(
    "rnn_cell", 'LSTM',
    "The type of recurrent network.")

class LogisticModel(models.BaseModel):
  """Logistic model with L2 regularization."""

  def create_model(self, model_input, vocab_size, l2_penalty=1e-8, **unused_params):
    """Creates a logistic model.

    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""
    output = slim.fully_connected(
        model_input, vocab_size, activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    return {"predictions": output}


class LSTMCTCModel(models.BaseModel):
  """Logistic model with L2 regularization."""

  def weight_variable(self,shape,name="v"):
        return tf.get_variable(name+"_weight",shape,
                              initializer=tf.truncated_normal_initializer(mean=0., stddev=.075, seed=None, dtype=tf.float32))

  def bias_variable(self,shape,name="v"):
   
   return tf.get_variable(name+"_bias",shape,
                         initializer=tf.truncated_normal_initializer(mean=0., stddev=.001, seed=None, dtype=tf.float32))

  def conv2d(self,x, W):
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

  def max_pool_2x2(self,x,h=2,w=2):
      return tf.nn.max_pool(x, ksize=[1, h, w, 1],
                        strides=[1, h, w, 1], padding='SAME')

  def convLayer(self,data,chanels_out,size_window=5,keep_prob=0.8,maxPool=None,scopeN="l1"):
    """Implement convolutional layer
    @param data: [batch,h,w,chanels]
    @param chanels_out: number of out chanels
    @param size_windows: the windows size
    @param keep_prob: the dropout amount
    @param maxPool: if true the max pool is applyed
    @param scopeN: the scope name
    
    returns convolutional output [batch,h,w,chanels_out]
    """
    with tf.name_scope("conv-"+scopeN):
        shape = data.get_shape().as_list()
        with tf.variable_scope("convVars-"+scopeN) as scope:
            W_conv1 = self.weight_variable([size_window, size_window, shape[3], chanels_out], scopeN)
            b_conv1 = self.bias_variable([chanels_out], scopeN)
        h_conv1 = tf.nn.relu(self.conv2d(data, W_conv1) + b_conv1)
        
        h_conv1 = slim.dropout(h_conv1, is_training=self.train_b, scope='dropout3'+scopeN)
        
        if maxPool is not None:
            h_conv1 = self.max_pool_2x2(h_conv1,maxPool[0],maxPool[1])
    return h_conv1

  def CNN(self, inputs):
        is_training = self.train_b
        batch_norm_params = {'is_training': is_training, 'decay': 0.9, 'updates_collections': None}
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params):
            x = inputs#tf.reshape(inputs, [-1, self.height, self.width, 1])

            # For slim.conv2d, default argument values are like
            # normalizer_fn = None, normalizer_params = None, <== slim.arg_scope changes these arguments
            # padding='SAME', activation_fn=nn.relu,
            # weights_initializer = initializers.xavier_initializer(),
            # biases_initializer = init_ops.zeros_initializer,
            net = slim.conv2d(x, 32, [5, 5], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.conv2d(net, 64, [5, 5], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.conv2d(net, 124, [2, 2], scope='conv3')
            net = slim.max_pool2d(net, [2, 1], scope='pool3')
            net = slim.conv2d(net, 256, [2, 2], scope='conv4')
            net = slim.max_pool2d(net, [2, 1], scope='pool4')
            #net = slim.flatten(net, scope='flatten3')

            # For slim.fully_connected, default argument values are like
            # activation_fn = nn.relu,
            # normalizer_fn = None, normalizer_params = None, <== slim.arg_scope changes these arguments
            # weights_initializer = initializers.xavier_initializer(),
            # biases_initializer = init_ops.zeros_initializer,
            #net = slim.fully_connected(net, 1024, scope='fc3')
            #net = slim.dropout(net, is_training=is_training, scope='dropout3')  # 0.5 by default
            #outputs = slim.fully_connected(net, self.num_classes, activation_fn=None, normalizer_fn=None, scope='fco')
        return net  

  def create_model(self, model_input, seq_len, vocab_size, target=None, is_training=True,keep_prob=1., **unused_params):
    """Creates a logistic model.

    Args:
      model_input: 'batch' x 'time_steps' x'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""
    imageInputs  = tf.cast(model_input, tf.float32)
    seq_lens = tf.cast(seq_len, tf.int32)      
    #targets = tf.cast(target, tf.int32)      
    seq_lens = tf.reshape(seq_lens,[FLAGS.batch_size])  
    self.keep_prob = keep_prob
    self.train_b = is_training
        
    imageInputs = tf.reshape(imageInputs , [FLAGS.batch_size*FLAGS.slices,FLAGS.height, FLAGS.width,FLAGS.input_chanels])
    #tf.summary.image("images", 255*(tf.reshape(imageInputs , [-1,FLAGS.height, FLAGS.width,1])+0.5))
    with tf.name_scope('convLayers'):
        if True:
            conv4 = self.CNN(imageInputs)
        else:
            conv1 = self.convLayer(imageInputs, 32 ,              scopeN="l1",keep_prob=self.keep_prob,maxPool=[2,2])
            conv2 = self.convLayer(conv1,       64 ,              scopeN="l2",keep_prob=self.keep_prob,maxPool=[2,2])
            conv3 = self.convLayer(conv2,      128 ,size_window=3,scopeN="l3",keep_prob=self.keep_prob,maxPool=[2,1])
            conv4 = self.convLayer(conv3,      256 ,size_window=2,scopeN="l4",keep_prob=self.keep_prob,maxPool=[2,1])
            
    with tf.name_scope('preprocess'):
            hh,ww,chanels = conv4.get_shape().as_list()[1:4]
            #assert ww == self.width,'image does not have to become smaller in width'
            #assert chanels == 256
            
            h_pool2_flat = tf.reshape(conv4, [FLAGS.batch_size, FLAGS.slices ,hh*ww*chanels])
            #h_pool2_flat = tf.transpose(h_pool2_flat, [1, 0, 2])
            #h_pool2_flat = tf.reshape(h_pool2_flat, [ww, self.batch_size ,hh*chanels])
            
            # Permuting batch_size and n_steps
            #x = tf.transpose(h_pool2_flat, [2, 0, 1])
            # Reshape to (n_steps*batch_size, n_input)
            #x = tf.reshape(h_pool2_flat, [-1, hh*chanels*ww])
            # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
            x = h_pool2_flat#tf.split(x, FLAGS.slices, 0)
            
    myInitializer = tf.truncated_normal_initializer(mean=0., stddev=.075, seed=None, dtype=tf.float32)
    
    if FLAGS.rnn_cell == "LSTM":
                cell = tf.contrib.rnn.LSTMCell(FLAGS.hidden, state_is_tuple=True,initializer=myInitializer)
    elif FLAGS.rnn_cell == "BasicLSTM":
                cell = tf.contrib.rnn.BasicLSTMCell(FLAGS.hidden,forget_bias=1.0,state_is_tuple=True)
    elif FLAGS.rnn_cell == "GRU":
                cell = tf.contrib.rnn.GRUCell(FLAGS.hidden)
    elif FLAGS.rnn_cell == "GRIDLSTM":#does not works
                cell = tf.contrib.rnn.GridLSTMCell(FLAGS.hidden,
                                                   use_peepholes=True,state_is_tuple=True,
                                                   forget_bias=1.0, feature_size = 5,frequency_skip=5,
                                                   num_frequency_blocks=[102])
                cell1 = tf.contrib.rnn.GridLSTMCell(FLAGS.hidden,
                                                   use_peepholes=True,state_is_tuple=True,
                                                   forget_bias=1.0, feature_size = 5,frequency_skip=5,
                                                   num_frequency_blocks=[816])
                cells = [cell,cell1]
    else:
                raise Exception("model type not supported: {}".format(FLAGS.rnn_cell))
    keep_prob1 = tf.cond(tf.convert_to_tensor(self.train_b, dtype='bool',name='is_training'),
                         lambda:tf.constant(keep_prob,name='g1'),
                         lambda:tf.constant(1.0,name='dd'))
    cell = tf.contrib.rnn.DropoutWrapper(cell,input_keep_prob=keep_prob1)
    
    stackf = tf.contrib.rnn.MultiRNNCell([cell] * (FLAGS.layers) if FLAGS.rnn_cell[:4] != "GRID" else cells,
                                            state_is_tuple=(FLAGS.rnn_cell[-4:] == "LSTM"))
    stackb = tf.contrib.rnn.MultiRNNCell([cell] * (FLAGS.layers) if FLAGS.rnn_cell[:4] != "GRID" else cells,
                                                state_is_tuple=(FLAGS.rnn_cell[-4:] == "LSTM"))
    
    self.reset_state_stackf = stackf.zero_state(FLAGS.batch_size, dtype=tf.float32)
            
    self.reset_state_stackb = stackb.zero_state(FLAGS.batch_size, dtype=tf.float32)
    
    if True:
        outputs, (self.state_fw, self.state_bw)  = tf.nn.bidirectional_dynamic_rnn(stackf, stackb, x,
                                                                                 sequence_length=seq_lens,
                                                                                 dtype=tf.float32,
                                                                                 initial_state_fw=self.reset_state_stackf,
                                                                                 initial_state_bw=self.reset_state_stackb)
    else:
        outputs, self.state_fw, self.state_bw  = tf.contrib.rnn.stack_bidirectional_rnn(stackf, stackb, x,
                                                                                 sequence_length=seq_lens,
                                                                                 dtype=tf.float32,
                                                                                 initial_state_fw=self.reset_state_stackf,
                                                                                 initial_state_bw=self.reset_state_stackb)
    #print(outputs[0].get_shape().as_list(),'outputs')
    if True:
        y_predict = tf.reshape(tf.concat(outputs, 2), [-1, 2*FLAGS.hidden])
    else:
        y_predict = tf.reshape(outputs, [-1, 2*FLAGS.hidden])
    #print(y_predict.get_shape().as_list(),'predict')
    
    with tf.name_scope('Train'):
        with tf.variable_scope("ctc_loss-1") as scope:
            W = tf.get_variable('w',[FLAGS.hidden*2,vocab_size],initializer=myInitializer)
            # Zero initialization
            b = tf.get_variable('b', shape=[vocab_size],initializer=myInitializer)

        tf.summary.histogram('histogram-b-ctc', b)
        tf.summary.histogram('histogram-w-ctc', W)

        # Doing the affine projection
        logits = tf.matmul(y_predict, W) +  b 

        logits = slim.dropout(logits, is_training=self.train_b, scope='dropout3')
        
        # Reshaping back to the original shape
        logits = tf.reshape(logits, [FLAGS.batch_size, FLAGS.slices,  vocab_size]) 
        pre = tf.transpose(logits, [1, 0, 2])
        #print(pre.get_shape().as_list(),'last')
        #pre.name = 'preee'
    return {"predictions": pre}