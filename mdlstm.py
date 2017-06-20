import tensorflow as tf

import tensorflow.contrib.slim as slim

def ln(tensor, scope = None, epsilon = 1e-5):
    """ Layer normalizes a 2D tensor along its second axis """
    assert(len(tensor.get_shape()) == 2)
    m, v = tf.nn.moments(tensor, [1], keep_dims=True)
    if not isinstance(scope, str):
        scope = ''
    with tf.variable_scope(scope + 'layer_norm'):
        scale = tf.get_variable('scale',
                                shape=[tensor.get_shape()[1]],
                                initializer=tf.constant_initializer(1))
        shift = tf.get_variable('shift',
                                shape=[tensor.get_shape()[1]],
                                initializer=tf.constant_initializer(0))
    LN_initial = (tensor - m) / tf.sqrt(v + epsilon)

    return LN_initial * scale + shift

from tensorflow.contrib.rnn.python.ops.rnn_cell import _linear

class MultiDimentionalLSTMCell(tf.contrib.rnn.RNNCell):
    """
    Adapted from TF's BasicLSTMCell to use Layer Normalization.
    Note that state_is_tuple is always True.
    """

    def __init__(self, num_units, forget_bias=0.0, activation=tf.nn.tanh):
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._activation = activation

    @property
    def state_size(self):
        return tf.contrib.rnn.LSTMStateTuple(self._num_units, self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """Long short-term memory cell (LSTM).
        @param: imputs (batch,n)
        @param state: the states and hidden unit of the two cells
        """
        with tf.variable_scope(scope or type(self).__name__):
            c1,c2,h1,h2 = state

            # change bias argument to False since LN will add bias via shift
            concat = _linear([inputs, h1, h2], 5 * self._num_units, False)

            i, j, f1, f2, o = tf.split(concat, 5, 1)

            # add layer normalization to each gate
            i =  ln(i, scope = 'i/')
            j =  ln(j, scope = 'j/')
            f1 = ln(f1, scope = 'f1/')
            f2 = ln(f2, scope = 'f2/')
            o =  ln(o, scope = 'o/')

            new_c = (c1 * tf.nn.sigmoid(f1 + self._forget_bias) + 
                     c2 * tf.nn.sigmoid(f2 + self._forget_bias) + tf.nn.sigmoid(i) *
                   self._activation(j))

            # add layer_normalization in calculation of new hidden state
            new_h = self._activation(ln(new_c, scope = 'new_h/')) * tf.nn.sigmoid(o)
            new_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)

            return new_h, new_state

        
def multiDimentionalRNN_whileLoop(rnn_size,input_data,sh,dims=None,scopeN="layer1"):
        """Implements naive multidimentional recurent neural networks
        
        @param rnn_size: the hidden units
        @param input_data: the data to process of shape [batch,h,w,chanels]
        @param sh: [heigth,width] of the windows 
        @param dims: dimentions to reverse the input data,eg.
            dims=[False,True,True,False] => true means reverse dimention
        @param scopeN : the scope
        
        returns [batch,h/sh[0],w/sh[1],chanels*sh[0]*sh[1]] the output of the lstm
        """
        with tf.variable_scope("MultiDimentionalLSTMCell-"+scopeN):
            cell = MultiDimentionalLSTMCell(rnn_size)
        
            shape = input_data.get_shape().as_list()

            if shape[1]%sh[0] != 0:
                offset = tf.zeros([shape[0], sh[0]-(shape[1]%sh[0]), shape[2], shape[3]])
                input_data = tf.concat([input_data,offset],1)
                shape = input_data.get_shape().as_list()
            if shape[2]%sh[1] != 0:
                offset = tf.zeros([shape[0], shape[1], sh[1]-(shape[2]%sh[1]), shape[3]])
                input_data = tf.concat([input_data,offset],2)
                shape = input_data.get_shape().as_list()

            h,w = int(shape[1]/sh[0]),int(shape[2]/sh[1])
            features = sh[1]*sh[0]*shape[3]
            batch_size = shape[0]

            #x =  tf.reshape(input_data, [batch_size,h,w, features])
            lines = tf.split(input_data,h,axis=1)#have a list of h blocks of sh[0] lines
            x1 = []
            for line in lines:#shape[0], sh[0], shape[2], shape[3] - bs, sh[0], total width, chanels
              line = tf.transpose(line,[0,2,3,1])
              line = tf.reshape(line,[batch_size,w,features])
              x1.append(line)
            x = tf.stack(x1,axis=1)
            if dims is not None:
                #assert dims[0] == False and dims[3] == False
                x = tf.reverse(x, dims)
            x = tf.transpose(x, [1,2,0,3])
            x =  tf.reshape(x, [-1, features])
            x = tf.split(x, h*w, 0)     

            sequence_length = tf.ones(shape=(batch_size,), dtype=tf.int32)*shape[0]
            inputs_ta = tf.TensorArray(dtype=tf.float32, size=h*w,name='input_ta')
            inputs_ta = inputs_ta.unstack(x)
            states_ta = tf.TensorArray(dtype=tf.float32, size=h*w+1,name='state_ta',
                                       clear_after_read=False)
            outputs_ta = tf.TensorArray(dtype=tf.float32, size=h*w,name='output_ta')

            states_ta = states_ta.write(h*w, 
                                        tf.contrib.rnn.LSTMStateTuple(
                                            tf.zeros([batch_size,rnn_size], tf.float32),
                                                         tf.zeros([batch_size,rnn_size],
                                                                  tf.float32)))
            def getindex1(t,w):
                return tf.cond(tf.less_equal(tf.constant(w),t),
                               lambda:t-tf.constant(w),
                               lambda:tf.constant(h*w))
            def getindex2(t,w):
                return tf.cond(tf.less(tf.constant(0),tf.mod(t,tf.constant(w))),
                               lambda:t-tf.constant(1),
                               lambda:tf.constant(h*w))

            time = tf.constant(0)

            def body(time, outputs_ta, states_ta):
                constant_val = tf.constant(0)
                stateUp = tf.cond(tf.less_equal(tf.constant(w),time),
                                  lambda: states_ta.read(getindex1(time,w)),
                                  lambda: states_ta.read(h*w))
                stateLast = tf.cond(tf.less(constant_val,tf.mod(time,tf.constant(w))),
                                    lambda: states_ta.read(getindex2(time,w)),
                                    lambda: states_ta.read(h*w)) 

                currentState = stateUp[0],stateLast[0],stateUp[1],stateLast[1]
                out , state = cell(inputs_ta.read(time),currentState)  
                outputs_ta = outputs_ta.write(time,out)
                states_ta = states_ta.write(time,state)
                return time + 1, outputs_ta, states_ta

            def condition(time,outputs_ta,states_ta):
                return tf.less(time ,  tf.constant(h*w)) 

            result , outputs_ta, states_ta = tf.while_loop(condition, body, [time,outputs_ta,states_ta]
                                                           ,parallel_iterations=1)


            outputs = outputs_ta.stack()
            states  = states_ta.stack()

            y =  tf.reshape(outputs, [h,w,batch_size,rnn_size])
            y = tf.transpose(y, [2,0,1,3])
            if dims is not None:
                y = tf.reverse(y, dims)

            return y#,states

    
def tanAndSum(rnn_size,input_data,scope,sh):
        outs = []
        for i in range(2):
            for j in range(2):
                dims = []
                if i!=0:
                    dims.append(1)
                if j!=0:
                    dims.append(2)                 
                outputs  = multiDimentionalRNN_whileLoop(rnn_size,input_data,sh,
                                                       dims,scope+"-multi-l{0}".format(i*2+j))
                outs.append(outputs)
        #return outs
        outs = tf.stack(outs, axis=0)
        mean = tf.reduce_mean(outs, 0)
        return tf.nn.tanh(mean)
    
def tanAndSumConv(rnn_size,input_data,scope,sh,is_training,wid,outChanels,dropout):
        outs = []
        batch_norm_params = {'is_training': is_training, 'decay': 0.9, 'updates_collections': None}
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params):
            for i in range(2):
                for j in range(2):
                    dims = []
                    if i!=0:
                        dims.append(1)
                    if j!=0:
                        dims.append(2)                 
                    outputs  = multiDimentionalRNN_whileLoop(rnn_size,input_data,sh,
                                                           dims,scope+"-multi-l{0}".format(i*2+j))
                    outputs = slim.conv2d(outputs, outChanels, [wid[0], wid[1]], scope=scope+'-conv-{}'.format(i*2+j))
                    if dropout!=1.:
                        outputs = slim.dropout(outputs,dropout, is_training=is_training, scope=scope+'-dropout-{}'.format(i*2+j))
                    outs.append(outputs)
        #return outs
        outs = tf.stack(outs, axis=0)
        mean = tf.reduce_mean(outs, 0)
        return tf.nn.tanh(mean)
    
if False:
    graph = tf.Graph()
    with graph.as_default():

        input_data =  tf.placeholder(tf.float32, [2,4,6,1])
        nr =  tf.placeholder(tf.float32)
        #input_data = tf.ones([20,36,90,1],dtype=tf.float32)
        sh = [2,2]
        out = tanAndSum(20,input_data,'l1',[2,2])
        #out = tanAndSum(25,out1,'l2',[1,1])
        #out = multiDimentionalRNN_whileLoop(20,input_data,sh,dims=None,scopeN="layer1")
        '''
        cell = tf.contrib.rnn.BasicLSTMCell(20)    
        x = tf.transpose(input_data, [2, 0, 1,3])
        # Reshaping to (n_steps*batch_size, n_input)
        x = tf.reshape(x, [-1, 4])
        # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        x = tf.split(0, 6, x)    
        out, st = tf.nn.rnn(cell, x, dtype=tf.float32)
        '''

        # 'outputs' is a list of output at every timestep, we stack them in a Tensor
        # and change back dimension to [batch_size, n_step, n_input]
        #outputs = tf.stack(out)
        #outputs = tf.transpose(out, [1, 0, 2,3])
        outputs = tf.reshape(out, [-1, 20])
        weights = {
            'out': tf.Variable(tf.random_normal([20, 2]))
        }
        biases = {
            'out': tf.Variable(tf.random_normal([2]))
        }

        tt = tf.matmul(outputs, weights['out']) + biases['out']

        s = tf.reduce_mean(tt)
        cost = (s-nr)*(s-nr)
        #gr = tf.gradients(cost,)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
        #out,st = multiDimentionalRNN_whileLoop(2,input_data,sh,dims=[False,True,True,False],scopeN="layer1")
        #cell = MultiDimentionalLSTMCell(10)
        #out = cell.zero_state(2, tf.float32).c

    from random import randint
    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        for i in range(10000):
            dd = np.zeros([2,4,6,1]).astype('float')
            nn = randint(1,9)
            for k in range(2):
                for j in range(nn):
                    w = randint(0,5)
                    h = randint(0,3)
                    nr1 = 0
                    while dd[k,h,w,0] !=0.0 and nr1<10000:
                        nr1 = nr1 + 1
                        w = randint(0,5)
                        h = randint(0,3)
                    assert nr1 < 9098
                    dd[k,h,w,0] = 1.0
            nn = float(nn)     

            c,_ = session.run([cost,optimizer],{input_data:dd,nr:nn})
            if i%1000==0:
                #print (nn, dd[0])
                print (i,c)

#print('no errors!')
