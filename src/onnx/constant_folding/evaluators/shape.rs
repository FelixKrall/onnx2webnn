/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Tarek Ziadé <tarek@ziade.org>
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Shape operation evaluator
// Extracts the shape of a tensor as a 1D int64 tensor

use crate::onnx::constant_folding::{
    ConstantEvaluator as EvaluatorTrait, ConstantFoldingContext, ConstantTensor, TensorData,
};
use crate::onnx::convert::OnnxError;
use crate::protos::onnx::{NodeProto, TensorProto_DataType};

pub struct ShapeEvaluator;

impl EvaluatorTrait for ShapeEvaluator {
    fn op_type(&self) -> &str {
        "Shape"
    }

    fn can_evaluate(&self, node: &NodeProto, ctx: &ConstantFoldingContext) -> bool {
        if node.op_type.as_str() != "Shape" {
            return false;
        }

        // Shape operation requires that we know the input's shape
        // The input doesn't need to be a constant, but we need its shape metadata
        if let Some(input_name) = node.input.first() {
            // Check if we have this as a constant (which includes shape info)
            if ctx.is_constant(input_name.as_str()) {
                return true;
            }
        }

        false
    }

    fn evaluate(
        &self,
        node: &NodeProto,
        ctx: &ConstantFoldingContext,
    ) -> Result<Vec<ConstantTensor>, OnnxError> {
        let input_name = node
            .input
            .first()
            .ok_or_else(|| OnnxError::MissingAttribute {
                attr: "input".to_string(),
                op: "Shape".to_string(),
            })?;

        let input_tensor = ctx.get_constant(input_name.as_str()).ok_or_else(|| {
            OnnxError::ShapeInference(format!(
                "Input tensor '{}' not found in constants",
                input_name
            ))
        })?;

        let (start, end) =
            crate::onnx::shape_inference::shape_attr_range(node, input_tensor.shape.len());
        let shape_values = input_tensor.shape[start..end].to_vec();

        // Create output tensor (1D int64 array)
        let output = ConstantTensor {
            data: TensorData::Int64(shape_values.clone()),
            shape: vec![shape_values.len() as i64],
            data_type: TensorProto_DataType::Int64.into(),
        };

        Ok(vec![output])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protos::onnx::{AttributeProto, TensorProto};
    use std::collections::HashMap;

    #[test]
    fn test_shape_evaluator() {
        // Create a test tensor with shape [2, 3, 4]
        let tensor = TensorProto {
            name: "test_input".to_string(),
            data_type: TensorProto_DataType::Float.into(),
            dims: vec![2, 3, 4],
            raw_data: vec![0u8; 4 * 2 * 3 * 4], // 24 floats
            ..Default::default()
        };

        // We need to leak the tensor to get a 'static reference for the test
        // In production code, the model owns the tensors
        let leaked_tensor: &'static TensorProto = Box::leak(Box::new(tensor));

        let mut init_map = HashMap::new();
        init_map.insert("test_input".to_string(), leaked_tensor);

        let ctx = ConstantFoldingContext::new(&init_map).unwrap();
        let evaluator = ShapeEvaluator;

        // Create a Shape node
        let mut node = NodeProto {
            op_type: "Shape".to_string(),
            input: vec!["test_input".to_string()],
            output: vec!["test_output".to_string()],
            ..Default::default()
        };

        // Check can_evaluate
        assert!(evaluator.can_evaluate(&node, &ctx));

        // Evaluate
        let result = evaluator.evaluate(&node, &ctx).unwrap();
        assert_eq!(result.len(), 1);

        let output = &result[0];
        assert_eq!(output.shape, vec![3]); // Output is 1D with 3 elements
        assert_eq!(output.data_type, TensorProto_DataType::Int64 as i32);

        if let TensorData::Int64(ref values) = output.data {
            assert_eq!(values, &vec![2, 3, 4]);
        } else {
            panic!("Expected Int64 data");
        }

        for (start, end, expected) in [
            (Some(1), Some(2), vec![3]),
            (Some(1), None, vec![3, 4]),
            (None, Some(-1), vec![2, 3]),
            (Some(-2), Some(-1), vec![3]),
            (Some(-100), Some(100), vec![2, 3, 4]),
            (Some(3), None, vec![]),
            (Some(100), None, vec![]),
            (Some(2), Some(1), vec![]),
            (None, Some(-100), vec![]),
            (Some(i64::MIN), Some(i64::MAX), vec![2, 3, 4]),
        ] {
            node.attribute = [("start", start), ("end", end)]
                .into_iter()
                .filter_map(|(name, value)| {
                    value.map(|i| AttributeProto {
                        name: name.into(),
                        i,
                        ..Default::default()
                    })
                })
                .collect();
            let result = evaluator.evaluate(&node, &ctx).unwrap();
            assert_eq!(result[0].shape, vec![expected.len() as i64]);
            assert_eq!(result[0].data_type, TensorProto_DataType::Int64 as i32);
            let TensorData::Int64(values) = &result[0].data else {
                panic!("Expected Int64 data");
            };
            assert_eq!(values, &expected, "start={start:?}, end={end:?}");
        }

        let scalar = TensorProto {
            name: "test_input".into(),
            data_type: TensorProto_DataType::Float as i32,
            raw_data: vec![0; 4],
            ..Default::default()
        };
        let scalar_map = HashMap::from([("test_input".into(), &scalar)]);
        let scalar_ctx = ConstantFoldingContext::new(&scalar_map).unwrap();
        let result = evaluator.evaluate(&node, &scalar_ctx).unwrap();
        assert_eq!(result[0].shape, vec![0]);
        assert!(matches!(&result[0].data, TensorData::Int64(values) if values.is_empty()));
    }
}
