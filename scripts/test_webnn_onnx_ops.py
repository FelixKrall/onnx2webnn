# SPDX-License-Identifier: Apache-2.0
"""Regressions for the Python consumers of the generated operator catalog."""

import unittest

from webnn_onnx_ops import is_webnn_supported_op


class OperatorSupportTests(unittest.TestCase):
    def test_pattern_only_ops_are_not_standalone_generator_successes(self):
        self.assertFalse(is_webnn_supported_op("ai.onnx", "SequenceAt"))
        self.assertFalse(is_webnn_supported_op("ai.onnx", "SplitToSequence"))
        self.assertTrue(is_webnn_supported_op("ai.onnx", "Add"))


if __name__ == "__main__":
    unittest.main()
