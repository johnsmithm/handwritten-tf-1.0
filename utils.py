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

"""Contains a collection of util functions for training and evaluating.
"""

import numpy
import tensorflow as tf
from tensorflow import logging




def MakeSummary(name, value):
  """Creates a tf.Summary proto with the given name and value."""
  summary = tf.Summary()
  val = summary.value.add()
  val.tag = str(name)
  val.simple_value = float(value)
  return summary





def GetListOfFeatureNamesAndSizes(feature_names, feature_sizes):
  """Extract the list of feature names and the dimensionality of each feature
     from string of comma separated values.

  Args:
    feature_names: string containing comma separated list of feature names
    feature_sizes: string containing comma separated list of feature sizes

  Returns:
    List of the feature names and list of the dimensionality of each feature.
    Elements in the first/second list are strings/integers.
  """
  list_of_feature_names = [
      feature_names.strip() for feature_names in feature_names.split(',')]
  list_of_feature_sizes = [
      int(feature_sizes) for feature_sizes in feature_sizes.split(',')]
  if len(list_of_feature_names) != len(list_of_feature_sizes):
    logging.error("length of the feature names (=" +
                  str(len(list_of_feature_names)) + ") != length of feature "
                  "sizes (=" + str(len(list_of_feature_sizes)) + ")")

  return list_of_feature_names, list_of_feature_sizes

