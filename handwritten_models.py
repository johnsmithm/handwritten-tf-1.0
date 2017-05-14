# -*- coding: utf-8 -*-
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
import numpy as np
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
flags.DEFINE_integer(
    "stride", -1,
    "The number of images as input for an image.")
flags.DEFINE_integer(
    "Bwidth", 96,
    "The number of images as input for an image.")
flags.DEFINE_integer(
    "num_words", 10,
    "The number of images as input for an image.")
flags.DEFINE_integer(
    "word_size", 20,
    "The number of images as input for an image.")
flags.DEFINE_integer(
    "num_heads", 2,
    "The number of images as input for an image.")
#num_words=FLAGS.num_words, word_size=FLAGS.word_size, num_heads
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

class DNC:
    def __init__(self, input_size, output_size, seq_len, num_words=256, word_size=64, num_heads=4):
        #define data
        #input data - [[1 0] [0 1] [0 0] [0 0]]
        self.input_size = input_size
        #output data [[0 0] [0 0] [1 0] [0 1]]
        self.output_size = output_size #Y
        
        #define read + write vector size
        #10
        self.num_words = num_words #N
        #4 characters
        self.word_size = word_size #W
        
        #define number of read+write heads
        #we could have multiple, but just 1 for simplicity
        self.num_heads = num_heads #R

        #size of output vector from controller that defines interactions with memory matrix
        self.interface_size = num_heads*word_size + 3*word_size + 5*num_heads + 3

        #the actual size of the neural network input after flatenning and
        # concatenating the input vector with the previously read vctors from memory
        self.nn_input_size = num_heads * word_size + input_size
        
        #size of output
        self.nn_output_size = output_size + self.interface_size
        
        #gaussian normal distribution for both outputs
        self.nn_out = tf.truncated_normal([1, self.output_size], stddev=0.1)
        self.interface_vec = tf.truncated_normal([1, self.interface_size], stddev=0.1)

        #Create memory matrix
        self.mem_mat = tf.zeros([num_words, word_size]) #N*W
        
        #other variables
        #The usage vector records which locations have been used so far, 
        self.usage_vec = tf.fill([num_words, 1], 1e-6) #N*1
        #a temporal link matrix records the order in which locations were written;
        self.link_mat = tf.zeros([num_words,num_words]) #N*N
        #represents degrees to which last location was written to
        self.precedence_weight = tf.zeros([num_words, 1]) #N*1

        #Read and write head variables
        self.read_weights = tf.fill([num_words, num_heads], 1e-6) #N*R
        self.write_weights = tf.fill([num_words, 1], 1e-6) #N*1
        self.read_vecs = tf.fill([num_heads, word_size], 1e-6) #R*W

        ###NETWORK VARIABLES
        #gateways into the computation graph for input output pairs
        #self.i_data = xData#tf.placeholder(tf.float32, [seq_len, self.input_size], name='input_node')
        #self.o_data = tf.placeholder(tf.int32, [seq_len], name='output_node')
        
        #2 layer feedforwarded network
        
        myInitializer = tf.truncated_normal_initializer(mean=0., stddev=.075, seed=None, dtype=tf.float32)
        self.W1 = tf.get_variable('layer1_weights',[self.nn_input_size, 32],initializer=myInitializer)
        #tf.Variable(tf.truncated_normal([self.nn_input_size, 32], stddev=0.1),
        #                      name='layer1_weights', dtype=tf.float32)
        self.b1 = tf.get_variable('layer1_bias',[32],initializer=myInitializer)
        #tf.Variable(tf.zeros([32]), name='layer1_bias', dtype=tf.float32)
        self.W2 = tf.get_variable('layer2_weights',[32, self.nn_output_size],initializer=myInitializer)
        #tf.Variable(tf.truncated_normal([32, self.nn_output_size], stddev=0.1),
        #                      name='layer2_weights', dtype=tf.float32)
        self.b2 = tf.get_variable('layer2_bias',[self.nn_output_size],initializer=myInitializer)
        #tf.Variable(tf.zeros([self.nn_output_size]), name='layer2_bias', dtype=tf.float32)

        ###DNC OUTPUT WEIGHTS
        self.nn_out_weights = tf.get_variable('net_output_weights',[self.nn_output_size, self.output_size],initializer=myInitializer)
        #tf.Variable(tf.truncated_normal([self.nn_output_size, self.output_size], stddev=0.1), name='net_output_weights')
        self.interface_weights = tf.get_variable('interface_weights',[self.nn_output_size,
                                                                      self.interface_size],
                                                 initializer=myInitializer)
        #tf.Variable(tf.truncated_normal([self.nn_output_size,
        #                                                          self.interface_size], 
        #                                                         stddev=0.1), name='interface_weights')
        
        self.read_vecs_out_weight = tf.get_variable('read_vector_weights',[self.num_heads*self.word_size,self.output_size],
                                                    initializer=myInitializer)
        #tf.Variable(tf.truncated_normal([self.num_heads*self.word_size,self.output_size], stddev=0.1),
        #                                        name='read_vector_weights')

    #3 attention mechanisms for read/writes to memory 
    
    #1
    #a key vector emitted by the controller is compared to the 
    #content of each location in memory according to a similarity measure 
    #The similarity scores determine a weighting that can be used by the read heads 
    #for associative recall1 or by the write head to modify an existing vector in memory.
    def content_lookup(self, key, str):
        #The l2 norm of a vector is the square root of the sum of the 
        #absolute values squared
        norm_mem = tf.nn.l2_normalize(self.mem_mat, 1) #N*W
        norm_key = tf.nn.l2_normalize(key, 0) #1*W for write or R*W for read
        #get similarity measure between both vectors, transpose before multiplicaiton
        ##(N*W,W*1)->N*1 for write
        #(N*W,W*R)->N*R for read
        sim = tf.matmul(norm_mem, norm_key, transpose_b=True) 
        #str is 1*1 or 1*R
        #returns similarity measure
        return tf.nn.softmax(sim*str, 0) #N*1 or N*R

    #2
    #retreives the writing allocation weighting based on the usage free list
    #The ‘usage’ of each location is represented as a number between 0 and 1, 
    #and a weighting that picks out unused locations is delivered to the write head. 
    
    # independent of the size and contents of the memory, meaning that 
    #DNCs can be trained to solve a task using one size of memory and later 
    #upgraded to a larger memory without retraining
    def allocation_weighting(self):
        #sorted usage - the usage vector sorted ascndingly
        #the original indices of the sorted usage vector
        sorted_usage_vec, free_list = tf.nn.top_k(-1 * self.usage_vec, k=self.num_words)
        sorted_usage_vec *= -1
        cumprod = tf.cumprod(sorted_usage_vec, axis=0, exclusive=True)
        unorder = (1-sorted_usage_vec)*cumprod

        alloc_weights = tf.zeros([self.num_words])
        I = tf.constant(np.identity(self.num_words, dtype=np.float32))
        
        #for each usage vec
        for pos, idx in enumerate(tf.unstack(free_list[0])):
            #flatten
            m = tf.squeeze(tf.slice(I, [idx, 0], [1, -1]))
            #add to weight matrix
            alloc_weights += m*unorder[0, pos]
        #the allocation weighting for each row in memory
        return tf.reshape(alloc_weights, [self.num_words, 1])

    #at every time step the controller receives input vector from dataset and emits output vector. 
    #it also recieves a set of read vectors from the memory matrix at the previous time step via 
    #the read heads. then it emits an interface vector that defines its interactions with the memory
    #at the current time step
    def step_m(self, x):
        
        #reshape input
        input = tf.concat([x, tf.reshape(self.read_vecs, [1, self.num_heads*self.word_size])],1)
        
        #forward propagation
        l1_out = tf.matmul(input, self.W1) + self.b1
        l1_act = tf.nn.relu(l1_out)
        l2_out = tf.matmul(l1_act, self.W2) + self.b2
        l2_act = tf.nn.relu(l2_out)
        
        #output vector
        self.nn_out = tf.matmul(l2_act, self.nn_out_weights) #(1*eta+Y, eta+Y*Y)->(1*Y)
        #interaction vector - how to interact with memory
        self.interface_vec = tf.matmul(l2_act, self.interface_weights) #(1*eta+Y, eta+Y*eta)->(1*eta)
        
        
        partition = tf.constant([[0]*(self.num_heads*self.word_size) + [1]*(self.num_heads)\
                                 + [2]*(self.word_size) + [3] + \
                    [4]*(self.word_size) + [5]*(self.word_size) + \
                    [6]*(self.num_heads) + [7] + [8] + [9]*(self.num_heads*3)], dtype=tf.int32)

        #convert interface vector into a set of read write vectors
        #using tf.dynamic_partitions(Partitions interface_vec into 10 tensors using indices from partition)
        (read_keys, read_str, write_key, write_str,
         erase_vec, write_vec, free_gates, alloc_gate, write_gate, read_modes) = \
            tf.dynamic_partition(self.interface_vec, partition, 10)
        
        #read vectors
        read_keys = tf.reshape(read_keys,[self.num_heads, self.word_size]) #R*W
        read_str = 1 + tf.nn.softplus(tf.expand_dims(read_str, 0)) #1*R
        
        #write vectors
        write_key = tf.expand_dims(write_key, 0) #1*W
        #help init our write weights
        write_str = 1 + tf.nn.softplus(tf.expand_dims(write_str, 0)) #1*1
        erase_vec = tf.nn.sigmoid(tf.expand_dims(erase_vec, 0)) #1*W
        write_vec = tf.expand_dims(write_vec, 0) #1*W
        
        #the degree to which locations at read heads will be freed
        free_gates = tf.nn.sigmoid(tf.expand_dims(free_gates, 0)) #1*R
        #the fraction of writing that is being allocated in a new location
        alloc_gate = tf.nn.sigmoid(alloc_gate) #1
        #the amount of information to be written to memory
        write_gate = tf.nn.sigmoid(write_gate) #1
        #the softmax distribution between the three read modes (backward, forward, lookup)
        #The read heads can use gates called read modes to switch between content lookup 
        #using a read key and reading out locations either forwards or backwards 
        #in the order they were written.
        read_modes = tf.nn.softmax(tf.reshape(read_modes, [3, self.num_heads])) #3*R
        
        #used to calculate usage vector, what's available to write to?
        retention_vec = tf.reduce_prod(1-free_gates*self.read_weights, reduction_indices=1)
        #used to dynamically allocate memory
        self.usage_vec = (self.usage_vec + self.write_weights - 
                          self.usage_vec * self.write_weights) * retention_vec

        ##retreives the writing allocation weighting 
        alloc_weights = self.allocation_weighting() #N*1
        #where to write to??
        write_lookup_weights = self.content_lookup(write_key, write_str) #N*1
        #define our write weights now that we know how much space to allocate for them and where to write to
        self.write_weights = write_gate*(alloc_gate*alloc_weights + (1-alloc_gate)*write_lookup_weights)

        #write erase, then write to memory!
        self.mem_mat = self.mem_mat*(1-tf.matmul(self.write_weights, erase_vec)) + \
                       tf.matmul(self.write_weights, write_vec)

        #As well as writing, the controller can read from multiple locations in memory. 
        #Memory can be searched based on the content of each location, or the associative 
        #temporal links can be followed forward and backward to recall information written 
        #in sequence or in reverse. (3rd attention mechanism)
        
        #updates and returns the temporal link matrix for the latest write
        #given the precedence vector and the link matrix from previous step
        nnweight_vec = tf.matmul(self.write_weights, tf.ones([1,self.num_words])) #N*N
        self.link_mat = (1 - nnweight_vec - tf.transpose(nnweight_vec))*self.link_mat + \
                        tf.matmul(self.write_weights, self.precedence_weight, transpose_b=True)
        self.link_mat *= tf.ones([self.num_words, self.num_words]) - tf.constant(np.identity(self.num_words, dtype=np.float32))

        
        self.precedence_weight = (1-tf.reduce_sum(self.write_weights, reduction_indices=0)) * \
                                 self.precedence_weight + self.write_weights
        #3 modes - forward, backward, content lookup
        forw_w = read_modes[2]*tf.matmul(self.link_mat, self.read_weights) #(N*N,N*R)->N*R
        look_w = read_modes[1]*self.content_lookup(read_keys, read_str) #N*R
        back_w = read_modes[0]*tf.matmul(self.link_mat, self.read_weights, transpose_a=True) #N*R

        #use them to intiialize read weights
        self.read_weights = back_w + look_w + forw_w #N*R
        #create read vectors by applying read weights to memory matrix
        self.read_vecs = tf.transpose(tf.matmul(self.mem_mat, self.read_weights, transpose_a=True)) #(W*N,N*R)^T->R*W

        #multiply them together
        read_vec_mut = tf.matmul(tf.reshape(self.read_vecs, [1, self.num_heads * self.word_size]),
                                 self.read_vecs_out_weight)  # (1*RW, RW*Y)-> (1*Y)
        
        #return output + read vecs product
        return self.nn_out+read_vec_mut

    #output list of numbers (one hot encoded) by running the step function
    def run(self,i_data):
        big_out = []
        for t, seq in enumerate(tf.unstack(i_data, axis=0)):
            seq = tf.expand_dims(seq, 0)
            y = self.step_m(seq)
            big_out.append(y)
        return tf.stack(big_out, axis=0)

class DNCModel(models.BaseModel):
  """Logistic model with L2 regularization."""
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
            #net = slim.conv2d(net, 256, [2, 2], scope='conv4')
            #net = slim.max_pool2d(net, [2, 1], scope='pool4')
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
    imageInputs1  = tf.cast(model_input, tf.float32)
    seq_lens = tf.cast(seq_len, tf.int32)      
    #targets = tf.cast(target, tf.int32)      
    seq_lens1 = tf.reshape(seq_lens,[FLAGS.batch_size])  
    self.keep_prob = keep_prob
    self.train_b = is_training
    if FLAGS.stride == -1:    
        imageInputs = tf.reshape(imageInputs1 , [FLAGS.batch_size*FLAGS.slices,FLAGS.height, FLAGS.width,FLAGS.input_chanels])
    else:
        imageInputs2 = tf.reshape(imageInputs1 , [FLAGS.batch_size,FLAGS.height, FLAGS.Bwidth,FLAGS.input_chanels])
        imageInputs, seq_lens = self.get_slices(imageInputs2, seq_lens1)
        imageInputs = tf.reshape(imageInputs , [FLAGS.batch_size*FLAGS.slices,FLAGS.height, FLAGS.width,FLAGS.input_chanels])
        seq_lens = tf.cast(seq_lens, tf.int32)
        if FLAGS.input_chanels == 1:
            tf.summary.image("image", (tf.reshape(imageInputs2 , [-1,FLAGS.height, FLAGS.Bwidth,1])),1)
            tf.summary.image("image-slices", imageInputs, FLAGS.slices)
    with tf.name_scope('convLayers'):
        if True:
            conv4 = self.CNN(imageInputs)
            
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
            
    dnc = DNC(input_size=hh*ww*chanels, output_size=FLAGS.hidden,
                      seq_len=FLAGS.slices, num_words=FLAGS.num_words, word_size=FLAGS.word_size, num_heads=FLAGS.num_heads)
    
    big_out = []
    for t, seq in enumerate(tf.unstack(x, axis=0)):
            
            y = dnc.run(seq)            
            big_out.append(y)
    outputs =  tf.stack(big_out, axis=0)
    if True:
        y_predict = tf.reshape(outputs, [-1, FLAGS.hidden])
    
    myInitializer = tf.truncated_normal_initializer(mean=0., stddev=.075, seed=None, dtype=tf.float32)
    with tf.name_scope('Train'):
        with tf.variable_scope("ctc_loss-1") as scope:
            W = tf.get_variable('w',[FLAGS.hidden,vocab_size],initializer=myInitializer)
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
            #net = slim.conv2d(net, 256, [2, 2], scope='conv4')
            #net = slim.max_pool2d(net, [2, 1], scope='pool4')
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
    
  def get_slices(self,inputs, seq_len):
    #inputs bat x hei x width x chanels
    
    #add padding based on stride, width
    shape = tf.shape(inputs)
    print(inputs.get_shape().as_list())
    if shape[2] - FLAGS.width % FLAGS.stride != 0:
        print((inputs.get_shape().as_list()[2] - FLAGS.width) % FLAGS.stride)
        inputs = tf.concat([inputs,tf.zeros([shape[0],shape[1],
                                             (FLAGS.stride -(shape[2] - FLAGS.width) % FLAGS.stride),shape[3]],
                                            dtype=tf.float32)],2)
    print(inputs.get_shape().as_list())
    #return inputs , -1
    h = []
    inputsT = tf.transpose(inputs,[2,0,1,3])#w b h c
    for i in range(FLAGS.slices):
        g = tf.gather(inputsT, tf.range(FLAGS.width)+tf.constant(i*FLAGS.stride))
        h.append(g)# w b h c
    k= tf.stack(h,0)#slices, w, b , h , c
    
    return tf.transpose(k,[2,0,3,1,4]), tf.maximum(tf.minimum(\
        tf.floor_div(tf.maximum(seq_len-FLAGS.width+2*FLAGS.stride,FLAGS.stride),FLAGS.stride),FLAGS.slices),1)


  def create_model(self, model_input, seq_len, vocab_size, target=None, is_training=True,keep_prob=1., **unused_params):
    """Creates a logistic model.

    Args:
      model_input: 'batch' x 'time_steps' x'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""
    imageInputs1  = tf.cast(model_input, tf.float32)
    seq_lens = tf.cast(seq_len, tf.int32)      
    #targets = tf.cast(target, tf.int32)      
    seq_lens1 = tf.reshape(seq_lens,[FLAGS.batch_size])  
    self.keep_prob = keep_prob
    self.train_b = is_training
    if FLAGS.stride == -1:    
        imageInputs = tf.reshape(imageInputs1 , [FLAGS.batch_size*FLAGS.slices,FLAGS.height, FLAGS.width,FLAGS.input_chanels])
    else:
        imageInputs2 = tf.reshape(imageInputs1 , [FLAGS.batch_size,FLAGS.height, FLAGS.Bwidth,FLAGS.input_chanels])
        imageInputs, seq_lens = self.get_slices(imageInputs2, seq_lens1)
        imageInputs = tf.reshape(imageInputs , [FLAGS.batch_size*FLAGS.slices,FLAGS.height, FLAGS.width,FLAGS.input_chanels])
        seq_lens = tf.cast(seq_lens, tf.int32)
        if FLAGS.input_chanels == 1:
            tf.summary.image("image", (tf.reshape(imageInputs2 , [-1,FLAGS.height, FLAGS.Bwidth,1])),1)
            tf.summary.image("image-slices", imageInputs, FLAGS.slices)
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
                cellf = tf.contrib.rnn.LSTMCell(FLAGS.hidden, state_is_tuple=True,initializer=myInitializer)
                cellb = tf.contrib.rnn.LSTMCell(FLAGS.hidden, state_is_tuple=True,initializer=myInitializer)
    elif FLAGS.rnn_cell == "BasicLSTM":
                cellf = tf.contrib.rnn.BasicLSTMCell(FLAGS.hidden,forget_bias=1.0,state_is_tuple=True)
                cellb = tf.contrib.rnn.BasicLSTMCell(FLAGS.hidden,forget_bias=1.0,state_is_tuple=True)
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
    #cellfd = tf.contrib.rnn.DropoutWrapper(cellf,input_keep_prob=keep_prob1)
    #cellbd = tf.contrib.rnn.DropoutWrapper(cellb,input_keep_prob=keep_prob1)
    
    stackf = tf.contrib.rnn.MultiRNNCell([cellf for _ in range(FLAGS.layers)] if FLAGS.rnn_cell[:4] != "GRID" else cells,
                                            state_is_tuple=(FLAGS.rnn_cell[-4:] == "LSTM"))
    stackb = tf.contrib.rnn.MultiRNNCell([cellb for _ in range(FLAGS.layers)] if FLAGS.rnn_cell[:4] != "GRID" else cells,
                                                state_is_tuple=(FLAGS.rnn_cell[-4:] == "LSTM"))
    
    self.reset_state_stackf = stackf.zero_state(FLAGS.batch_size, dtype=tf.float32)
            
    self.reset_state_stackb = stackb.zero_state(FLAGS.batch_size, dtype=tf.float32)
    
    tf.add_to_collection("reset_state_stackb", self.reset_state_stackb)
    tf.add_to_collection("reset_state_stackf", self.reset_state_stackf)
    
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
    
    tf.add_to_collection("final_state_stackb", self.state_bw)
    tf.add_to_collection("final_state_stackf", self.state_fw)
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

class RNNCTCModel(models.BaseModel):
  """Logistic model with L2 regularization."""

  def create_model(self, model_input, seq_len, vocab_size, target=None, is_training=True,keep_prob=1., **unused_params):
    """Creates a logistic model.

    Args:
      model_input: 'batch' x 'time_steps' x'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""
    imageInputs1  = tf.cast(model_input, tf.float32)
    seq_lens = tf.cast(seq_len, tf.int32)      
    #targets = tf.cast(target, tf.int32)      
    seq_lens1 = tf.reshape(seq_lens,[FLAGS.batch_size])  
    self.keep_prob = keep_prob
    self.train_b = is_training
    if FLAGS.width == 1:    
        x = tf.reshape(imageInputs1 , [FLAGS.batch_size, FLAGS.slices, FLAGS.height])  
        
    
    with tf.name_scope('Train'):        
        myInitializer = tf.truncated_normal_initializer(mean=0., stddev=.075, seed=None, dtype=tf.float32)

        if FLAGS.rnn_cell == "LSTM":
                    cellf = tf.contrib.rnn.LSTMCell(FLAGS.hidden, state_is_tuple=True,initializer=myInitializer)
                    cellb = tf.contrib.rnn.LSTMCell(FLAGS.hidden, state_is_tuple=True,initializer=myInitializer)
        elif FLAGS.rnn_cell == "BasicLSTM":
                    cellf = tf.contrib.rnn.BasicLSTMCell(FLAGS.hidden,forget_bias=1.0,state_is_tuple=True)
                    cellb = tf.contrib.rnn.BasicLSTMCell(FLAGS.hidden,forget_bias=1.0,state_is_tuple=True)
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
        #cellf = tf.contrib.rnn.DropoutWrapper(cellf,input_keep_prob=keep_prob1)
        #cellb = tf.contrib.rnn.DropoutWrapper(cellb,input_keep_prob=keep_prob1)

        stackf = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(FLAGS.hidden, state_is_tuple=True,initializer=myInitializer),input_keep_prob=keep_prob1) for _ in range(FLAGS.layers)],#* (FLAGS.layers) if FLAGS.rnn_cell[:4] != "GRID" else cells,
                                                state_is_tuple=(FLAGS.rnn_cell[-4:] == "LSTM"))
        stackb = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(FLAGS.hidden, state_is_tuple=True,initializer=myInitializer),input_keep_prob=keep_prob1) for _ in range(FLAGS.layers)],#* (FLAGS.layers) if FLAGS.rnn_cell[:4] != "GRID" else cells,
                                                    state_is_tuple=(FLAGS.rnn_cell[-4:] == "LSTM"))

        self.reset_state_stackf = stackf.zero_state(FLAGS.batch_size, dtype=tf.float32)

        self.reset_state_stackb = stackb.zero_state(FLAGS.batch_size, dtype=tf.float32)
        
        tf.add_to_collection("reset_state_stackb", self.reset_state_stackb)
        tf.add_to_collection("reset_state_stackf", self.reset_state_stackf)

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
        
        tf.add_to_collection("final_state_stackb", self.state_bw[1])
        tf.add_to_collection("final_state_stackf", self.state_bw[0])
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