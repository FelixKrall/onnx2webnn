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

// Gather operation evaluator
// Gathers elements from a data tensor at indices

use crate::onnx::constant_folding::{
    ConstantEvaluator as EvaluatorTrait, ConstantFoldingContext, ConstantTensor, TensorData,
};
use crate::onnx::convert::OnnxError;
use crate::protos::onnx::NodeProto;

pub struct GatherEvaluator;

impl GatherEvaluator {
    /// Gather complete axis-0 slices, preserving the indices tensor's rank.
    fn gather_data(
        data: &TensorData,
        data_shape: &[i64],
        indices: &[i64],
        indices_shape: &[i64],
        axis: i64,
    ) -> Result<(TensorData, Vec<i64>), OnnxError> {
        let normalized_axis = if axis < 0 {
            data_shape.len() as i64 + axis
        } else {
            axis
        };
        if normalized_axis < 0 || normalized_axis >= data_shape.len() as i64 {
            return Err(OnnxError::InvalidShape(format!(
                "Gather axis {axis} out of bounds for shape {data_shape:?}"
            )));
        }
        if normalized_axis != 0 {
            return Err(OnnxError::unsupported_op(
                "Gather",
                "constant folding supports only axis=0",
            ));
        }
        let element_count = |shape: &[i64]| -> Result<usize, OnnxError> {
            shape.iter().try_fold(1usize, |size, &dim| {
                usize::try_from(dim)
                    .ok()
                    .and_then(|dim| size.checked_mul(dim))
                    .ok_or_else(|| {
                        OnnxError::InvalidShape(format!("invalid Gather shape {shape:?}"))
                    })
            })
        };
        if element_count(data_shape)? != data.len()
            || element_count(indices_shape)? != indices.len()
        {
            return Err(OnnxError::InvalidShape(
                "Gather tensor shape does not match its data length".into(),
            ));
        }
        let slice_size = element_count(&data_shape[1..])?;
        let axis_size = data_shape[0];
        let starts = indices
            .iter()
            .map(|&index| {
                let normalized = if index < 0 { index + axis_size } else { index };
                if normalized < 0 || normalized >= axis_size {
                    return Err(OnnxError::InvalidShape(format!(
                        "Gather index {index} out of bounds for axis size {axis_size}"
                    )));
                }
                Ok(normalized as usize * slice_size)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let output_data = match data {
            TensorData::Int64(values) => {
                TensorData::Int64(Self::gather_slices(values, &starts, slice_size))
            }
            TensorData::Int32(values) => {
                TensorData::Int32(Self::gather_slices(values, &starts, slice_size))
            }
            TensorData::Float32(values) => {
                TensorData::Float32(Self::gather_slices(values, &starts, slice_size))
            }
            _ => {
                return Err(OnnxError::unsupported_op(
                    "Gather",
                    format!("data type {:?} not supported", data.data_type()),
                ))
            }
        };
        let mut output_shape = indices_shape.to_vec();
        output_shape.extend_from_slice(&data_shape[1..]);
        Ok((output_data, output_shape))
    }

    fn gather_slices<T: Copy>(values: &[T], starts: &[usize], slice_size: usize) -> Vec<T> {
        let mut output = Vec::new();
        for &start in starts {
            output.extend_from_slice(&values[start..start + slice_size]);
        }
        output
    }

    /// Get indices as int64 vector
    fn get_indices(tensor: &ConstantTensor) -> Result<Vec<i64>, OnnxError> {
        match &tensor.data {
            TensorData::Int64(v) => Ok(v.clone()),
            TensorData::Int32(v) => Ok(v.iter().map(|&x| x as i64).collect()),
            _ => Err(OnnxError::InvalidShape(format!(
                "Gather indices must be integer type, got {:?}",
                tensor.data_type
            ))),
        }
    }

    /// Check if data tensor looks like an embedding table
    /// Heuristic: Large tensors (>10K elements) are likely embedding tables
    /// that shouldn't be folded even if indices appear constant
    fn is_embedding_pattern(data_name: &str, ctx: &ConstantFoldingContext) -> bool {
        if let Some(tensor) = ctx.get_constant(data_name) {
            let num_elements: i64 = tensor.shape.iter().product();
            // Threshold: 10K elements
            // - Typical embedding tables: 50K vocab x 512 dims = 25M+ elements
            // - Shape tensors: <10 elements
            // - Small lookup tables: <1K elements
            if num_elements > 10000 {
                return true;
            }
        }
        false
    }
}

impl EvaluatorTrait for GatherEvaluator {
    fn op_type(&self) -> &str {
        "Gather"
    }

    fn can_evaluate(&self, node: &NodeProto, ctx: &ConstantFoldingContext) -> bool {
        if node.op_type.as_str() != "Gather" {
            return false;
        }

        // Need both data and indices to be constants
        let inputs = node.input.as_slice();
        if inputs.len() < 2 {
            return false;
        }

        // Check if this looks like an embedding lookup pattern
        // Don't fold these as indices may vary at runtime despite static shapes
        if Self::is_embedding_pattern(inputs[0].as_str(), ctx) {
            return false;
        }

        ctx.is_constant(inputs[0].as_str()) && ctx.is_constant(inputs[1].as_str())
    }

    fn evaluate(
        &self,
        node: &NodeProto,
        ctx: &ConstantFoldingContext,
    ) -> Result<Vec<ConstantTensor>, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.len() < 2 {
            return Err(OnnxError::MissingAttribute {
                attr: "inputs".to_string(),
                op: "Gather".to_string(),
            });
        }

        let data_tensor = ctx.get_constant(inputs[0].as_str()).ok_or_else(|| {
            OnnxError::ShapeInference(format!("Data tensor '{}' not found", inputs[0]))
        })?;

        let indices_tensor = ctx.get_constant(inputs[1].as_str()).ok_or_else(|| {
            OnnxError::ShapeInference(format!("Indices tensor '{}' not found", inputs[1]))
        })?;

        // Get axis attribute (default 0)
        let axis = node
            .attribute
            .iter()
            .find(|a| a.name.as_str() == "axis")
            .map(|a| a.i)
            .unwrap_or(0);

        // Convert indices to int64
        let indices = Self::get_indices(indices_tensor)?;

        // Perform gather
        let (output_data, output_shape) = Self::gather_data(
            &data_tensor.data,
            &data_tensor.shape,
            &indices,
            &indices_tensor.shape,
            axis,
        )?;

        let output = ConstantTensor {
            data_type: output_data.data_type().into(),
            data: output_data,
            shape: output_shape,
        };

        Ok(vec![output])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protos::onnx::{NodeProto, TensorProto, TensorProto_DataType};
    use std::collections::HashMap;

    #[test]
    fn test_gather_shape_dimensions() {
        // Test gathering specific dimensions from a shape tensor
        // This is the common pattern: Shape -> Gather

        // Create context with a shape tensor [2, 128, 384]
        let shape_tensor = TensorProto {
            name: "shape".to_string(),
            data_type: TensorProto_DataType::Int64.into(),
            dims: vec![3],
            int64_data: vec![2, 128, 384],
            ..Default::default()
        };

        let shape_tensor_static: &'static TensorProto = Box::leak(Box::new(shape_tensor));

        // Create indices tensor [0, 1]  (gather first two dimensions)
        let indices_tensor = TensorProto {
            name: "indices".to_string(),
            data_type: TensorProto_DataType::Int64.into(),
            dims: vec![2],
            int64_data: vec![0, 1],
            ..Default::default()
        };

        let indices_tensor_static: &'static TensorProto = Box::leak(Box::new(indices_tensor));

        let mut init_map = HashMap::new();
        init_map.insert("shape".to_string(), shape_tensor_static);
        init_map.insert("indices".to_string(), indices_tensor_static);

        let ctx = ConstantFoldingContext::new(&init_map).unwrap();
        let evaluator = GatherEvaluator;

        // Create Gather node
        let node = NodeProto {
            op_type: "Gather".to_string(),
            input: vec!["shape".to_string(), "indices".to_string()],
            output: vec!["gathered".to_string()],
            ..Default::default()
        };

        // Evaluate
        assert!(evaluator.can_evaluate(&node, &ctx));
        let result = evaluator.evaluate(&node, &ctx).unwrap();

        assert_eq!(result.len(), 1);
        let output = &result[0];

        if let TensorData::Int64(ref values) = output.data {
            assert_eq!(values, &vec![2, 128]);
        } else {
            panic!("Expected Int64 data");
        }

        assert_eq!(output.shape, vec![2]);
    }

    #[test]
    fn test_gather_scalar_index() {
        // Test gathering a single element (scalar result)

        let data_tensor = TensorProto {
            name: "data".to_string(),
            data_type: TensorProto_DataType::Int64.into(),
            dims: vec![4],
            int64_data: vec![10, 20, 30, 40],
            ..Default::default()
        };

        let data_tensor_static: &'static TensorProto = Box::leak(Box::new(data_tensor));

        let index_tensor = TensorProto {
            name: "index".to_string(),
            data_type: TensorProto_DataType::Int64.into(),
            dims: vec![],        // Scalar
            int64_data: vec![2], // Get index 2 (value 30)
            ..Default::default()
        };

        let index_tensor_static: &'static TensorProto = Box::leak(Box::new(index_tensor));

        let mut init_map = HashMap::new();
        init_map.insert("data".to_string(), data_tensor_static);
        init_map.insert("index".to_string(), index_tensor_static);

        let ctx = ConstantFoldingContext::new(&init_map).unwrap();
        let evaluator = GatherEvaluator;

        let node = NodeProto {
            op_type: "Gather".to_string(),
            input: vec!["data".to_string(), "index".to_string()],
            output: vec!["result".to_string()],
            ..Default::default()
        };

        let result = evaluator.evaluate(&node, &ctx).unwrap();
        let output = &result[0];

        if let TensorData::Int64(ref values) = output.data {
            assert_eq!(values, &vec![30]);
        } else {
            panic!("Expected Int64 data");
        }

        assert_eq!(output.shape, Vec::<i64>::new()); // Scalar output
    }
    #[test]
    fn gather_preserves_slices_and_indices_rank_for_supported_types() {
        for data in [
            TensorData::Int64(vec![10, 11, 20, 21]),
            TensorData::Int32(vec![10, 11, 20, 21]),
            TensorData::Float32(vec![10., 11., 20., 21.]),
        ] {
            for (indices, indices_shape, shape, expected) in [
                (vec![1], vec![1], vec![1, 2], vec![20., 21.]),
                (vec![-1], vec![], vec![2], vec![20., 21.]),
                (
                    vec![1, 0],
                    vec![1, 2],
                    vec![1, 2, 2],
                    vec![20., 21., 10., 11.],
                ),
                (
                    vec![0, -2, 0],
                    vec![3],
                    vec![3, 2],
                    vec![10., 11., 10., 11., 10., 11.],
                ),
                (vec![], vec![0], vec![0, 2], vec![]),
            ] {
                let (values, output_shape) =
                    GatherEvaluator::gather_data(&data, &[2, 2], &indices, &indices_shape, -2)
                        .unwrap();
                assert_eq!(output_shape, shape);
                let actual = match values {
                    TensorData::Int64(v) => v.into_iter().map(|v| v as f32).collect::<Vec<_>>(),
                    TensorData::Int32(v) => v.into_iter().map(|v| v as f32).collect::<Vec<_>>(),
                    TensorData::Float32(v) => v,
                    _ => unreachable!(),
                };
                assert_eq!(actual, expected);
            }
            for index in [2, -3] {
                assert!(GatherEvaluator::gather_data(&data, &[2, 2], &[index], &[], 0).is_err());
            }
        }
        let empty = TensorData::Int64(vec![]);
        let (values, shape) = GatherEvaluator::gather_data(&empty, &[2, 0], &[1], &[1], 0).unwrap();
        assert!(values.is_empty());
        assert_eq!(shape, [1, 0]);
        assert!(GatherEvaluator::gather_data(&empty, &[0, 2], &[0], &[1], 0).is_err());
        assert!(GatherEvaluator::gather_data(&empty, &[0, 2], &[], &[1], 0).is_err());
        assert!(GatherEvaluator::gather_data(&empty, &[], &[], &[0], 0).is_err());
    }
}
