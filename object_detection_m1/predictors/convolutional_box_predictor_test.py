# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Tests for object_detection.predictors.convolutional_box_predictor."""
import numpy as np
import tensorflow as tf

from google.protobuf import text_format
from object_detection.builders import box_predictor_builder
from object_detection.builders import hyperparams_builder
from object_detection.predictors import convolutional_box_predictor as box_predictor
from object_detection.protos import hyperparams_pb2
from object_detection.utils import test_case


class ConvolutionalBoxPredictorTest(test_case.TestCase):

  def _build_arg_scope_with_conv_hyperparams(self):
    conv_hyperparams = hyperparams_pb2.Hyperparams()
    conv_hyperparams_text_proto = """
      activation: RELU_6
      regularizer {
        l2_regularizer {
        }
      }
      initializer {
        truncated_normal_initializer {
        }
      }
    """
    text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams)
    return hyperparams_builder.build(conv_hyperparams, is_training=True)

  def test_get_boxes_for_five_aspect_ratios_per_location(self):
    def graph_fn(image_features):
      conv_box_predictor = (
          box_predictor_builder.build_convolutional_box_predictor(
              is_training=False,
              num_classes=0,
              conv_hyperparams_fn=self._build_arg_scope_with_conv_hyperparams(),
              min_depth=0,
              max_depth=32,
              num_layers_before_predictor=1,
              use_dropout=True,
              dropout_keep_prob=0.8,
              kernel_size=1,
              box_code_size=4))
      box_predictions = conv_box_predictor.predict(
          [image_features], num_predictions_per_location=[5],
          scope='BoxPredictor')
      box_encodings = tf.concat(
          box_predictions[box_predictor.BOX_ENCODINGS], axis=1)
      objectness_predictions = tf.concat(
          box_predictions[box_predictor.CLASS_PREDICTIONS_WITH_BACKGROUND],
          axis=1)
      return (box_encodings, objectness_predictions)
    image_features = np.random.rand(4, 8, 8, 64).astype(np.float32)
    (box_encodings, objectness_predictions) = self.execute(graph_fn,
                                                           [image_features])
    self.assertAllEqual(box_encodings.shape, [4, 320, 1, 4])
    self.assertAllEqual(objectness_predictions.shape, [4, 320, 1])

if __name__ == '__main__':
  tf.test.main()
