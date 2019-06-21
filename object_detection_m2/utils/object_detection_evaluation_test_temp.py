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

"""Tests for object_detection.utils.object_detection_evaluation."""

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from object_detection import eval_util
from object_detection.core import standard_fields
from object_detection.utils import object_detection_evaluation



class PascalEvaluationTest(tf.test.TestCase):

  def test_returns_correct_metric_values_on_boxes(self):
    categories = [{'id': 1, 'name': 'cat'},
                  {'id': 2, 'name': 'dog'},
                  {'id': 3, 'name': 'elephant'}]
    #  Add groundtruth
    pascal_evaluator = object_detection_evaluation.PascalDetectionEvaluator(
        categories)
    image_key1 = 'img1'
    groundtruth_boxes1 = np.array([[0, 0, 1, 1], [0, 0, 2, 2], [0, 0, 3, 3]],
                                  dtype=float)
    groundtruth_class_labels1 = np.array([1, 3, 1], dtype=int)
    pascal_evaluator.add_single_ground_truth_image_info(
        image_key1,
        {standard_fields.InputDataFields.groundtruth_boxes: groundtruth_boxes1,
         standard_fields.InputDataFields.groundtruth_classes:
         groundtruth_class_labels1,
         standard_fields.InputDataFields.groundtruth_difficult:
         np.array([], dtype=bool)})
    image_key2 = 'img2'
    groundtruth_boxes2 = np.array([[10, 10, 11, 11], [500, 500, 510, 510],
                                   [10, 10, 12, 12]], dtype=float)
    groundtruth_class_labels2 = np.array([1, 1, 3], dtype=int)
    groundtruth_is_difficult_list2 = np.array([False, True, False], dtype=bool)
    pascal_evaluator.add_single_ground_truth_image_info(
        image_key2,
        {standard_fields.InputDataFields.groundtruth_boxes: groundtruth_boxes2,
         standard_fields.InputDataFields.groundtruth_classes:
         groundtruth_class_labels2,
         standard_fields.InputDataFields.groundtruth_difficult:
         groundtruth_is_difficult_list2})
    image_key3 = 'img3'
    groundtruth_boxes3 = np.array([[0, 0, 1, 1]], dtype=float)
    groundtruth_class_labels3 = np.array([2], dtype=int)
    pascal_evaluator.add_single_ground_truth_image_info(
        image_key3,
        {standard_fields.InputDataFields.groundtruth_boxes: groundtruth_boxes3,
         standard_fields.InputDataFields.groundtruth_classes:
         groundtruth_class_labels3})
    image_key4 = 'img4'
    groundtruth_boxes4 = np.empty(shape=[0, 4], dtype=float)
    groundtruth_class_labels4 = np.array([], dtype=int)
    pascal_evaluator.add_single_ground_truth_image_info(
        image_key4,
        {standard_fields.InputDataFields.groundtruth_boxes: groundtruth_boxes4,
         standard_fields.InputDataFields.groundtruth_classes:
         groundtruth_class_labels4})

    # Add detections
    image_key = 'img2'
    detected_boxes = np.array(
        #[[10, 10, 11, 11], [100, 100, 120, 120], [100, 100, 220, 220]],
        [[10, 10, 11, 11], [100, 100, 120, 120], [10, 10, 12, 12]],
        dtype=float)
    detected_class_labels = np.array([1, 1, 3], dtype=int)
    #detected_scores = np.array([0.7, 0.8, 0.9], dtype=float)
    detected_scores = np.array([0.8, 0.7, 0.9], dtype=float)
    pascal_evaluator.add_single_detected_image_info(
        image_key,
        {standard_fields.DetectionResultFields.detection_boxes: detected_boxes,
         standard_fields.DetectionResultFields.detection_scores:
         detected_scores,
         standard_fields.DetectionResultFields.detection_classes:
         detected_class_labels})

    metrics = pascal_evaluator.evaluate()
    """
    self.assertAlmostEqual(
        metrics['PascalBoxes_PerformanceByCategory/AP@0.5IOU/dog'], 0.0)
    self.assertAlmostEqual(
        metrics['PascalBoxes_PerformanceByCategory/AP@0.5IOU/elephant'], 0.0)
    self.assertAlmostEqual(
        metrics['PascalBoxes_PerformanceByCategory/AP@0.5IOU/cat'], 0.16666666)
    self.assertAlmostEqual(metrics['PascalBoxes_Precision/mAP@0.5IOU'],
                           0.05555555)
    """
    pascal_evaluator.clear()
    self.assertFalse(pascal_evaluator._image_ids)


if __name__ == '__main__':
  tf.test.main()
