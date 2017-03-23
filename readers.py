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

"""Provides readers configured for different datasets."""

import tensorflow as tf
import utils

from tensorflow import logging


class BaseReader(object):
  """Inherit from this class when implementing new readers."""

  def prepare_reader(self, unused_filename_queue):
    """Create a thread for generating prediction and label tensors."""
    raise NotImplementedError()


class AIMAggregatedFeatureReader(BaseReader):
  """Reads TFRecords of pre-aggregated Examples.

  The TFRecords must contain Examples with a sparse int64 'labels' feature and
  a fixed length float32 feature, obtained from the features in 'feature_name'.
  The float features are assumed to be an average of dequantized values.
  """

  def __init__(self,
               num_classes=29,
               feature_sizes=[36],
               feature_names=["imageInput"],
               height=36,
               width=100,
               slices=12,
              input_chanels=1):
    """Construct a YT8MAggregatedFeatureReader.

    Args:
      num_classes: a positive integer for the number of classes.
      feature_sizes: positive integer(s) for the feature dimensions as a list.
      feature_names: the feature name(s) in the tensorflow record as a list.
    """

    assert len(feature_names) == len(feature_sizes), \
    "length of feature_names (={}) != length of feature_sizes (={})".format( \
    len(feature_names), len(feature_sizes))

    self.num_classes = num_classes
    self.feature_sizes = feature_sizes
    self.feature_names = feature_names
    self.height, self.width = height, width
    self.slices = slices
    self.input_chanels = input_chanels

  def prepare_reader(self, filename_queue, batch_size=50):
    """Creates a single reader thread for pre-aggregated YouTube 8M Examples.

    Args:
      filename_queue: A tensorflow queue of filename locations.

    Returns:
      A tuple of video indexes, features, labels, and padding data.
    """
    reader = tf.TFRecordReader()
    _, serialized_examples = reader.read_up_to(filename_queue, batch_size)

    tf.add_to_collection("serialized_examples", serialized_examples)
    return self.prepare_serialized_examples(serialized_examples)

  def prepare_serialized_examples(self, serialized_examples):
    # set the mapping from the fields to data types in the proto
    num_features = len(self.feature_names)
    assert num_features > 0, "self.feature_names is empty!"
    assert len(self.feature_names) == len(self.feature_sizes), \
    "length of feature_names (={}) != length of feature_sizes (={})".format( \
    len(self.feature_names), len(self.feature_sizes))

    '''features = tf.parse_single_example(
            serialized_examples,
            features={
                # We know the length of both fields. If not the
                # tf.VarLenFeature could be used
                'seq_len': tf.FixedLenFeature([1], tf.int64),
                'target': tf.VarLenFeature(tf.int64),     
                'imageInput': tf.FixedLenFeature([self.height*self.slices*self.width], tf.float32)
            })
        # now return the converted data
    imageInput = features['imageInput']
    seq_len     = features['seq_len']
    target     = features['target']'''
    
    feature_map = {'seq_len': tf.FixedLenFeature([1], tf.int64),
                'target': tf.VarLenFeature(tf.int64),     
                'imageInput': tf.FixedLenFeature([self.height*self.slices*self.width*self.input_chanels], tf.float32)}
    

    features = tf.parse_example(serialized_examples, features=feature_map)

    imageInput = features['imageInput']
    seq_len     = features['seq_len']
    target     = features['target']

    return imageInput, seq_len, target


