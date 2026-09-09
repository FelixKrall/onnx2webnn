#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import types
import unittest

# These focused tests do not use ONNX. Permit them to run before the optional
# generator dependencies have been installed.
sys.modules.setdefault("onnx", types.ModuleType("onnx"))

import generate_manifest as manifest


class ValidationMetadataTests(unittest.TestCase):
    def test_retained_cases_keep_branch_specific_validation(self):
        file = "org--repo/onnx/decoder_model.onnx"
        existing = [
            {
                "file": file,
                "override_dims": {"past_sequence_length": 0},
                "validation": {"tier": "smoke"},
            },
            {
                "file": file,
                "override_dims": {"past_sequence_length": 16},
                "validation": {"tier": "blocked", "reason": "known failure"},
            },
        ]
        baselines = manifest.Baselines(existing)

        self.assertEqual(
            baselines.manual_fields_for(file, {}, False)["validation"],
            {"tier": "smoke"},
        )
        self.assertEqual(
            baselines.manual_fields_for(file, {}, True)["validation"],
            {"tier": "blocked", "reason": "known failure"},
        )

    def test_new_and_previously_untriaged_cases_are_explicitly_untriaged(self):
        retained = "org--repo/onnx/model.onnx"
        baselines = manifest.Baselines([{"file": retained, "validation": None}])

        self.assertEqual(
            baselines.manual_fields_for(retained, {}, False)["validation"],
            {"tier": "untriaged"},
        )
        self.assertEqual(
            baselines.manual_fields_for("org--repo/onnx/new.onnx", {}, False)["validation"],
            {"tier": "untriaged"},
        )


if __name__ == "__main__":
    unittest.main()
