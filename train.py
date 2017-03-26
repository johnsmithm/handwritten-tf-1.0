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
"""Binary for training Tensorflow models on the YouTube-8M dataset."""

import json
import os
import time

import eval_util
import export_model
import losses
import handwritten_models
import readers
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow import app
from tensorflow import flags
from tensorflow import gfile
from tensorflow import logging
import utils

FLAGS = flags.FLAGS

if __name__ == "__main__":
  # Dataset flags.
  flags.DEFINE_string("train_dir", "/tmp/yt8m_model/",
                      "The directory to save the model files in.")
  flags.DEFINE_string(
      "train_data_pattern", "",
      "File glob for the training dataset. If the files refer to Frame Level "
      "features (i.e. tensorflow.SequenceExample), then set --reader_type "
      "format. The (Sequence)Examples are expected to have 'rgb' byte array "
      "sequence feature as well as a 'labels' int64 context feature.")
  flags.DEFINE_string(
      "test_data_pattern", "",
      "File glob for the training dataset. If the files refer to Frame Level "
      "features (i.e. tensorflow.SequenceExample), then set --reader_type "
      "format. The (Sequence)Examples are expected to have 'rgb' byte array "
      "sequence feature as well as a 'labels' int64 context feature.")
  flags.DEFINE_string("feature_names", "mean_rgb", "Name of the feature "
                      "to use for training.")
  flags.DEFINE_string("feature_sizes", "1024", "Length of the feature vectors.")

  # Model flags.
  flags.DEFINE_bool(
      "slice_features", True,
      "If set, then the input should have 4 dimentions ")
  flags.DEFINE_string(
      "model", "LSTMCTCModel",
      "Which architecture to use for the model. Models are defined "
      "in models.py.")
  flags.DEFINE_string(
      "vocab_path", "vocabulary.txt",
      "Which vocabulary to use in order to help the prediction "
      "in models.py.")
  flags.DEFINE_bool(
      "start_new_model", False,
      "If set, this will not resume from a checkpoint and will instead create a"
      " new model instance.")
  flags.DEFINE_integer("width", 12,
                       "image width")
  flags.DEFINE_integer("height", 36,
                       "image height")
  flags.DEFINE_integer("slices", 15,
                       "slices number")
  
  flags.DEFINE_integer("vocabulary_size", 29,
                       "character's number")
    
  flags.DEFINE_integer("beam_size", 2,
                       "guess number")

  flags.DEFINE_integer("display_step", 3,
                       "image width")
    
  flags.DEFINE_integer("display_step_lme", 6,
                       "image width")

  # Training flags.
  flags.DEFINE_integer("batch_size", 1,
                       "How many examples to process per batch for training.")
  flags.DEFINE_string("label_loss", "CTCLoss",
                      "Which loss function to use for training the model.")
  flags.DEFINE_float(
      "regularization_penalty", 1,
      "How much weight to give to the regularization loss (the label loss has "
      "a weight of 1).")
    
  flags.DEFINE_float("drop_out", 0.6,
                     "Which learning rate to start with.")

  flags.DEFINE_float("base_learning_rate", 0.001,
                     "Which learning rate to start with.")
  flags.DEFINE_float("learning_rate_decay", 0.95,
                     "Learning rate decay factor to be applied every "
                     "learning_rate_decay_examples.")
  flags.DEFINE_float("learning_rate_decay_examples", 4000000,
                     "Multiply current learning rate by learning_rate_decay "
                     "every learning_rate_decay_examples.")
  flags.DEFINE_integer("num_epochs", 50,
                       "How many passes to make over the dataset before "
                       "halting training.")
  flags.DEFINE_integer("max_steps", 10,
                       "The maximum number of iterations of the training loop.")
  flags.DEFINE_integer("export_model_steps", 1000,
                       "The period, in number of steps, with which the model "
                       "is exported for batch prediction.")

  # Other flags.
  flags.DEFINE_integer("num_readers", 2,
                       "How many threads to use for reading input files.")
  flags.DEFINE_string("optimizer", "AdamOptimizer",
                      "What optimizer class to use.")
  flags.DEFINE_float("clip_gradient_norm", 1.0, "Norm to clip gradients to.")
  flags.DEFINE_bool(
      "log_device_placement", False,
      "Whether to write the device on which every op will run into the "
      "logs on startup.")

def validate_class_name(flag_value, category, modules, expected_superclass):
  """Checks that the given string matches a class of the expected type.

  Args:
    flag_value: A string naming the class to instantiate.
    category: A string used further describe the class in error messages
              (e.g. 'model', 'reader', 'loss').
    modules: A list of modules to search for the given class.
    expected_superclass: A class that the given class should inherit from.

  Raises:
    FlagsError: If the given class could not be found or if the first class
    found with that name doesn't inherit from the expected superclass.

  Returns:
    True if a class was found that matches the given constraints.
  """
  candidates = [getattr(module, flag_value, None) for module in modules]
  for candidate in candidates:
    if not candidate:
      continue
    if not issubclass(candidate, expected_superclass):
      raise flags.FlagsError("%s '%s' doesn't inherit from %s." %
                             (category, flag_value,
                              expected_superclass.__name__))
    return True
  raise flags.FlagsError("Unable to find %s '%s'." % (category, flag_value))

def get_input_data_tensors(reader,
                           data_pattern,
                           batch_size=1000,
                           num_epochs=None,
                           num_readers=1,
                          nameT='train'):
  """Creates the section of the graph which reads the training data.

  Args:
    reader: A class which parses the training data.
    data_pattern: A 'glob' style path to the data files.
    batch_size: How many examples to process at a time.
    num_epochs: How many passes to make over the training data. Set to 'None'
                to run indefinitely.
    num_readers: How many I/O threads to use.

  Returns:
    A tuple containing the features tensor, labels tensor, and optionally a
    tensor containing the number of frames per video. The exact dimensions
    depend on the reader being used.

  Raises:
    IOError: If no files matching the given pattern were found.
  """
  logging.info("Using batch size of " + str(batch_size) + " for {}ing.".format(nameT))
  with tf.name_scope(nameT+"_input"):
    print(data_pattern)
    files = gfile.Glob(data_pattern)
    if not files:
      raise IOError("Unable to find {}ing files. data_pattern='".format(nameT) +
                    data_pattern + "'.")
    logging.info("Number of "+nameT+"ing files: %s.", str(len(files)))
    if nameT == "test":
        filename_queue = tf.train.string_input_producer(
        files, shuffle=True)
    else:
        filename_queue = tf.train.string_input_producer(
        files, num_epochs=num_epochs, shuffle=True)
    training_data = [
        reader.prepare_reader(filename_queue,batch_size) for _ in range(num_readers)
    ]

    return tf.train.shuffle_batch_join(
        training_data,
        batch_size=batch_size,
        capacity=FLAGS.batch_size * 5,
        min_after_dequeue=FLAGS.batch_size,
        allow_smaller_final_batch=True,
        enqueue_many=True)


def find_class_by_name(name, modules):
  """Searches the provided modules for the named class and returns it."""
  modules = [getattr(module, name, None) for module in modules]
  return next(a for a in modules if a)


def build_graph(reader,
                model,
                train_data_pattern,
                label_loss_fn=losses.CTCLoss(),
                batch_size=100,
                base_learning_rate=0.01,
                learning_rate_decay_examples=1000000,
                learning_rate_decay=0.95,
                optimizer_class=tf.train.AdamOptimizer,
                clip_gradient_norm=1.0,
                regularization_penalty=1,
                num_readers=1,
                num_epochs=None,
                decoder=losses.CTCDecoder('beam_search')):
  """Creates the Tensorflow graph.

  This will only be called once in the life of
  a training model, because after the graph is created the model will be
  restored from a meta graph file rather than being recreated.

  Args:
    reader: The data file reader. It should inherit from BaseReader.
    model: The core model (e.g. logistic or neural net). It should inherit
           from BaseModel.
    train_data_pattern: glob path to the training data files.
    label_loss_fn: What kind of loss to apply to the model. It should inherit
                from BaseLoss.
    batch_size: How many examples to process at a time.
    base_learning_rate: What learning rate to initialize the optimizer with.
    optimizer_class: Which optimization algorithm to use.
    clip_gradient_norm: Magnitude of the gradient to clip to.
    regularization_penalty: How much weight to give the regularization loss
                            compared to the label loss.
    num_readers: How many threads to use for I/O operations.
    num_epochs: How many passes to make over the data. 'None' means an
                unlimited number of passes.
  """
  
  global_step = tf.Variable(0, trainable=False, name="global_step")
  
  learning_rate = tf.train.exponential_decay(
      base_learning_rate,
      global_step * batch_size,
      learning_rate_decay_examples,
      learning_rate_decay,
      staircase=True)
  tf.summary.scalar('learning_rate', learning_rate)

  optimizer = optimizer_class(learning_rate)
  imageInput_tr, seq_len_tr , target_tr = (
      get_input_data_tensors(
          reader,
          train_data_pattern,
          batch_size=batch_size,
          num_readers=num_readers,
          num_epochs=num_epochs))
  imageInput_te, seq_len_te , target_te = (
      get_input_data_tensors(
          reader,
          FLAGS.test_data_pattern,
          batch_size=batch_size,
          num_readers=num_readers,
          num_epochs=num_epochs,nameT='test'))
    
    
  train_batch = tf.placeholder_with_default(True,[]) 
  
  imageInput, seq_len , target = tf.cond(
                train_batch,
                lambda:[imageInput_tr, seq_len_tr , target_tr],
                lambda:[imageInput_te, seq_len_te , target_te]
            )  
    
  seq_len = tf.cast(seq_len, tf.int32)      
  target = tf.cast(target, tf.int32)
  imageInput1 = tf.reshape(imageInput , [FLAGS.batch_size*FLAGS.slices,FLAGS.height, FLAGS.width,FLAGS.input_chanels])  
  seq_len1 = tf.reshape(seq_len, [FLAGS.batch_size])
  tf.summary.histogram("model/input_raw", imageInput)
  
    
    

  with tf.name_scope("model"):
    result = model.create_model(
        imageInput1,
        seq_len=seq_len1,
        vocab_size=reader.num_classes,
        is_training=train_batch,
        keep_prob=FLAGS.drop_out)

    for variable in slim.get_model_variables():
      tf.summary.histogram(variable.op.name, variable)

    predictions = result["predictions"]
    if "loss" in result.keys():
      label_loss = result["loss"]
    else:
      label_loss = label_loss_fn.calculate_loss(predictions, target, seq_len1)
    tf.summary.scalar("label_loss", label_loss)

    if "regularization_loss" in result.keys():
      reg_loss = result["regularization_loss"]
    else:
      reg_loss = tf.constant(0.0)
    
    reg_losses = tf.losses.get_regularization_losses()
    if reg_losses:
      reg_loss += tf.add_n(reg_losses)
    
    if regularization_penalty != 0:
      tf.summary.scalar("reg_loss", reg_loss)

    # Adds update_ops (e.g., moving average updates in batch normalization) as
    # a dependency to the train_op.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    if "update_ops" in result.keys():
      update_ops += result["update_ops"]
    if update_ops:
      with tf.control_dependencies(update_ops):
        barrier = tf.no_op(name="gradient_barrier")
        with tf.control_dependencies([barrier]):
          label_loss = tf.identity(label_loss)
        
    if decoder is not None:
        with tf.name_scope('Prediction'):
            decodedPrediction = decoder.decode(predictions, seq_len1,FLAGS.beam_size)
            ler = decoder.lebelRateError(decodedPrediction,target)
            #voDe = decoder.useVocabulary(target)

    # Incorporate the L2 weight penalties etc.
    final_loss = regularization_penalty * reg_loss + label_loss
    train_op = slim.learning.create_train_op(
        final_loss,
        optimizer,
        global_step=global_step,
        clip_gradient_norm=clip_gradient_norm)

    tf.add_to_collection("global_step", global_step)
    tf.add_to_collection("loss", label_loss)
    tf.add_to_collection("predictions", predictions)
    tf.add_to_collection("input_batch", imageInput)
    tf.add_to_collection("seq_len", seq_len)
    tf.add_to_collection("train_batch", train_batch)
    tf.add_to_collection("ler", ler)
    decoders = []
    for i in range(FLAGS.beam_size):
        #decoders.append(tf.sparse_tensor_to_dense(decodedPrediction[i]))
    
        tf.add_to_collection("decodedPrediction{}".format(i),tf.sparse_tensor_to_dense(decodedPrediction[i]))
        #tf.stack(decoders,0))
    tf.add_to_collection("labels", tf.sparse_tensor_to_dense(target))
    tf.add_to_collection("train_op", train_op)


class Trainer(object):
  """A Trainer to train a Tensorflow graph."""

  def __init__(self, cluster, task, train_dir, model, reader, model_exporter, 
               log_device_placement=True, max_steps=None, 
               export_model_steps=1000):
    """"Creates a Trainer.

    Args:
      cluster: A tf.train.ClusterSpec if the execution is distributed.
        None otherwise.
      task: A TaskSpec describing the job type and the task index.
    """

    self.cluster = cluster
    self.task = task
    self.is_master = (task.type == "master" and task.index == 0)
    self.train_dir = train_dir
    self.config = tf.ConfigProto(log_device_placement=log_device_placement)
    self.model = model
    self.reader = reader
    self.model_exporter = model_exporter
    self.max_steps = max_steps
    self.max_steps_reached = False
    self.export_model_steps = export_model_steps
    self.last_model_export_step = 0

    if self.is_master and self.task.index > 0:
      raise StandardError("%s: Only one replica of master expected",
                          task_as_string(self.task))

  def run(self, start_new_model=False):
    """Performs training on the currently defined Tensorflow graph.

    Returns:
      A tuple of the training Hit@1 and the training PERR.
    """
    if self.is_master and start_new_model:
      self.remove_training_directory(self.train_dir)

    target, device_fn = self.start_server_if_distributed()

    meta_filename = self.get_meta_filename(start_new_model, self.train_dir)

    with tf.Graph().as_default() as graph:

      if meta_filename:
        saver = self.recover_model(meta_filename)

      with tf.device(device_fn):

        if not meta_filename:
          saver = self.build_model(self.model, self.reader)

        global_step = tf.get_collection("global_step")[0]
        loss = tf.get_collection("loss")[0]
        predictions = tf.get_collection("predictions")[0]
        labels = tf.get_collection("labels")[0]
        train_batch = tf.get_collection("train_batch")[0]
        train_op = tf.get_collection("train_op")[0]
        decodedPrediction = []
        for i in range(FLAGS.beam_size):
            decodedPrediction.append(tf.get_collection("decodedPrediction{}".format(i))[0])
        ler = tf.get_collection("ler")[0]
        init_op = tf.global_variables_initializer()

    sv = tf.train.Supervisor(
        graph,
        logdir=self.train_dir,
        init_op=init_op,
        is_chief=self.is_master,
        global_step=global_step,
        save_model_secs=15 * 60,
        save_summaries_secs=120,
        saver=saver)
    
    vocabulary = eval_util.read_vocab(FLAGS.vocab_path)
    vocabulary = sorted(vocabulary, key=lambda word: len(word))
    
    logging.info("%s: Starting managed session.", task_as_string(self.task))
    with sv.managed_session(target, config=self.config) as sess:

      try:
        logging.info("%s: Entering training loop.", task_as_string(self.task))
        while (not sv.should_stop()) and (not self.max_steps_reached):

          batch_start_time = time.time()
          _ ,global_step_val = sess.run([train_op ,global_step])
          seconds_per_batch = time.time() - batch_start_time
        
          #print(decodedPr,'decoder pr');print(labels_val,'val label')#;print(decV1,'edit dis')
          #todo: add test/evaluation here--add placeholder

          if self.max_steps and self.max_steps <= global_step_val:
            self.max_steps_reached = True

          if self.is_master and global_step_val%FLAGS.display_step==0:
            global_step_val, loss_val, predictions_val, labels_val, labelRateError, decodedPr = sess.run(
              [ global_step, loss, predictions, labels, ler, decodedPrediction])
            
            global_step_val_te, loss_val_te, predictions_val_te, labels_val_te, labelRateError_te, decodedPr_te = sess.run(
              [ global_step, loss, predictions, labels, ler, decodedPrediction],{train_batch:False})
            
            examples_per_second = len(labels_val) / seconds_per_batch
            
            
            if global_step_val % FLAGS.display_step_lme == 0:
                lme, newGuess = eval_util.calculate_models_error_withLanguageModel(decodedPr, 
                                                                                   labels_val,
                                                                                   vocabulary, 
                                                                                   FLAGS.beam_size)
                lme_te, newGuess_te = eval_util.calculate_models_error_withLanguageModel(decodedPr_te,
                                                                                         labels_val_te,
                                                                                         vocabulary, 
                                                                                         FLAGS.beam_size)
            else:
                lme,  lme_te = -1., -1.
            if False:
                eval_util.show_prediction(decodedPr, labels_val)

            logging.info(
                "%s: training step " + str(global_step_val) 
                + "| LME: " +  ("%.2f" % lme) + "| LME-te: " +  ("%.2f" % lme_te) 
                + " ler: " +   ("%.2f" % labelRateError) + " ler-te: " +   ("%.2f" % labelRateError_te) 
                + " Loss: " +  ("%.2f" % loss_val) + " Loss-te: " + str(loss_val_te),
                task_as_string(self.task))

            sv.summary_writer.add_summary(
                utils.MakeSummary("model/labelRateError_train", labelRateError),
                global_step_val)
            sv.summary_writer.add_summary(
                utils.MakeSummary("model/labelRateError_test", labelRateError_te),
                global_step_val)
            sv.summary_writer.add_summary(
                utils.MakeSummary("model/lme_train", lme), global_step_val)
            sv.summary_writer.add_summary(
                utils.MakeSummary("model/lme_test", lme_te), global_step_val)
            sv.summary_writer.add_summary(
                utils.MakeSummary("model/loss_train", loss_val), global_step_val)
            sv.summary_writer.add_summary(
                utils.MakeSummary("model/loss_test", loss_val_te), global_step_val)
            sv.summary_writer.add_summary(
                utils.MakeSummary("global_step/Examples/Second",
                                  examples_per_second), global_step_val)
            sv.summary_writer.flush()

            # Exporting the model every x steps
            time_to_export = ((self.last_model_export_step == 0) or 
                (global_step_val - self.last_model_export_step 
                 >= self.export_model_steps))

            if self.is_master and time_to_export:
              eval_util.show_prediction(decodedPr, labels_val)
              self.export_model(global_step_val, sv.saver, sv.save_path, sess)
              self.last_model_export_step = global_step_val

        # Exporting the final model
        if self.is_master:
          eval_util.show_prediction(decodedPr, labels_val)
          self.export_model(global_step_val, sv.saver, sv.save_path, sess)

      except tf.errors.OutOfRangeError:
        logging.info("%s: Done training -- epoch limit reached.",
                     task_as_string(self.task))

    logging.info("%s: Exited training loop.", task_as_string(self.task))
    sv.Stop()

  def export_model(self, global_step_val, saver, save_path, session):

    # If the model has already been exported at this step, return.
    if global_step_val == self.last_model_export_step:
      return

    last_checkpoint = saver.save(session, save_path, global_step_val)

    model_dir = "{0}/export/step_{1}".format(self.train_dir, global_step_val)
    logging.info("%s: Exporting the model at step %s to %s.",
                 task_as_string(self.task), global_step_val, model_dir)

    self.model_exporter.export_model(
        model_dir=model_dir, 
        global_step_val=global_step_val,
        last_checkpoint=last_checkpoint)


  def start_server_if_distributed(self):
    """Starts a server if the execution is distributed."""

    if self.cluster:
      logging.info("%s: Starting trainer within cluster %s.",
                   task_as_string(self.task), self.cluster.as_dict())
      server = start_server(self.cluster, self.task)
      target = server.target
      device_fn = tf.train.replica_device_setter(
          ps_device="/job:ps",
          worker_device="/job:%s/task:%d" % (self.task.type, self.task.index),
          cluster=self.cluster)
    else:
      target = ""
      device_fn = ""
    return (target, device_fn)

  def remove_training_directory(self, train_dir):
    """Removes the training directory."""
    try:
      logging.info(
          "%s: Removing existing train directory.",
          task_as_string(self.task))
      gfile.DeleteRecursively(train_dir)
    except:
      logging.error(
          "%s: Failed to delete directory " + train_dir +
          " when starting a new model. Please delete it manually and" +
          " try again.", task_as_string(self.task))

  def get_meta_filename(self, start_new_model, train_dir):
    if start_new_model:
      logging.info("%s: Flag 'start_new_model' is set. Building a new model.",
                   task_as_string(self.task))
      return None
    
    latest_checkpoint = tf.train.latest_checkpoint(train_dir)
    if not latest_checkpoint: 
      logging.info("%s: No checkpoint file found. Building a new model.",
                   task_as_string(self.task))
      return None
    
    meta_filename = latest_checkpoint + ".meta"
    if not gfile.Exists(meta_filename):
      logging.info("%s: No meta graph file found. Building a new model.",
                     task_as_string(self.task))
      return None
    else:
      return meta_filename

  def recover_model(self, meta_filename):
    logging.info("%s: Restoring from meta graph file %s",
                 task_as_string(self.task), meta_filename)
    return tf.train.import_meta_graph(meta_filename)

  def build_model(self, model, reader):
    """Find the model and build the graph."""

    label_loss_fn = find_class_by_name(FLAGS.label_loss, [losses])()
    optimizer_class = find_class_by_name(FLAGS.optimizer, [tf.train])
  
    build_graph(reader=reader,
                 model=model,
                 optimizer_class=optimizer_class,
                 clip_gradient_norm=FLAGS.clip_gradient_norm,
                 train_data_pattern=FLAGS.train_data_pattern,
                 label_loss_fn=label_loss_fn,
                 base_learning_rate=FLAGS.base_learning_rate,
                 learning_rate_decay=FLAGS.learning_rate_decay,
                 learning_rate_decay_examples=FLAGS.learning_rate_decay_examples,
                 regularization_penalty=FLAGS.regularization_penalty,
                 num_readers=FLAGS.num_readers,
                 batch_size=FLAGS.batch_size,
                 num_epochs=FLAGS.num_epochs)
  
    return tf.train.Saver(max_to_keep=0, keep_checkpoint_every_n_hours=0.25)


def get_reader():
  # Convert feature_names and feature_sizes to lists of values.
  feature_names, feature_sizes = utils.GetListOfFeatureNamesAndSizes(
      FLAGS.feature_names, FLAGS.feature_sizes)

  if FLAGS.slice_features:
    reader = readers.AIMAggregatedFeatureReader(
        feature_names=feature_names, feature_sizes=feature_sizes,
               height=FLAGS.height,
               width=FLAGS.width,
               slices=FLAGS.slices,
                num_classes = FLAGS.vocabulary_size,
                input_chanels=FLAGS.input_chanels)
  else:
    raise NotImplementedError()
    
  return reader


class ParameterServer(object):
  """A parameter server to serve variables in a distributed execution."""

  def __init__(self, cluster, task):
    """Creates a ParameterServer.

    Args:
      cluster: A tf.train.ClusterSpec if the execution is distributed.
        None otherwise.
      task: A TaskSpec describing the job type and the task index.
    """

    self.cluster = cluster
    self.task = task

  def run(self):
    """Starts the parameter server."""

    logging.info("%s: Starting parameter server within cluster %s.",
                 task_as_string(self.task), self.cluster.as_dict())
    server = start_server(self.cluster, self.task)
    server.join()


def start_server(cluster, task):
  """Creates a Server.

  Args:
    cluster: A tf.train.ClusterSpec if the execution is distributed.
      None otherwise.
    task: A TaskSpec describing the job type and the task index.
  """

  if not task.type:
    raise ValueError("%s: The task type must be specified." %
                     task_as_string(task))
  if task.index is None:
    raise ValueError("%s: The task index must be specified." %
                     task_as_string(task))

  # Create and start a server.
  return tf.train.Server(
      tf.train.ClusterSpec(cluster),
      protocol="grpc",
      job_name=task.type,
      task_index=task.index)

def task_as_string(task):
  return "/job:%s/task:%s" % (task.type, task.index)

def main(unused_argv):
  # Load the environment.
  env = json.loads(os.environ.get("TF_CONFIG", "{}"))

  # Load the cluster data from the environment.
  cluster_data = env.get("cluster", None)
  cluster = tf.train.ClusterSpec(cluster_data) if cluster_data else None

  # Load the task data from the environment.
  task_data = env.get("task", None) or {"type": "master", "index": 0}
  task = type("TaskSpec", (object,), task_data)

  # Logging the version.
  logging.set_verbosity(tf.logging.INFO)
  logging.info("%s: Tensorflow version: %s.",
               task_as_string(task), tf.__version__)

  # Dispatch to a master, a worker, or a parameter server.
  if not cluster or task.type == "master" or task.type == "worker":
    
    model = find_class_by_name(FLAGS.model, 
        [handwritten_models])()
    
    reader = get_reader()
    
    model_exporter = export_model.ModelExporter(
        slices_features=FLAGS.slice_features,
        model=model,
        reader=reader)

    Trainer(cluster, task, FLAGS.train_dir, model, reader, model_exporter, 
            FLAGS.log_device_placement, FLAGS.max_steps, 
            FLAGS.export_model_steps).run(start_new_model=FLAGS.start_new_model)

  elif task.type == "ps":

    ParameterServer(cluster, task).run()

  else:

    raise ValueError("%s: Invalid task_type: %s." %
                     (task_as_string(task), task.type))

if __name__ == "__main__":
  app.run()
