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

// Reshape-related operation evaluators
// Unsqueeze, Squeeze, Cast operations

use crate::onnx::constant_folding::{
    ConstantEvaluator as EvaluatorTrait, ConstantFoldingContext, ConstantTensor, TensorData,
};
use crate::onnx::convert::OnnxError;
use crate::protos::onnx::{NodeProto, TensorProto_DataType};

/// Read the axes input for newer opsets, or the legacy attribute.
fn constant_axes(
    node: &NodeProto,
    ctx: &ConstantFoldingContext,
) -> Result<Option<Vec<i64>>, OnnxError> {
    if let Some(name) = node.input.get(1).filter(|name| !name.is_empty()) {
        let tensor = ctx.get_constant(name).ok_or_else(|| {
            OnnxError::ShapeInference(format!("Axes tensor '{name}' is not constant"))
        })?;
        if tensor.shape.len() != 1 {
            return Err(OnnxError::InvalidShape(
                "axes input must be a 1D int64 tensor".into(),
            ));
        }
        return match &tensor.data {
            TensorData::Int64(values) => Ok(Some(values.clone())),
            _ => Err(OnnxError::InvalidShape(
                "axes input must be a 1D int64 tensor".into(),
            )),
        };
    }
    Ok(node
        .attribute
        .iter()
        .find(|attr| attr.name == "axes")
        .map(|attr| attr.ints.clone()))
}

/// Normalize all axes against one rank before changing the shape.
fn normalized_axes(axes: &[i64], rank: usize) -> Result<Vec<usize>, OnnxError> {
    let mut normalized = axes
        .iter()
        .map(|&axis| {
            let axis = if axis < 0 { axis + rank as i64 } else { axis };
            if axis < 0 || axis >= rank as i64 {
                return Err(OnnxError::InvalidShape(format!(
                    "axis {axis} out of bounds for rank {rank}"
                )));
            }
            Ok(axis as usize)
        })
        .collect::<Result<Vec<_>, _>>()?;
    normalized.sort_unstable();
    if normalized.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(OnnxError::InvalidShape("duplicate axes".into()));
    }
    Ok(normalized)
}

/// Unsqueeze operation: add dimensions of size 1
pub struct UnsqueezeEvaluator;

impl EvaluatorTrait for UnsqueezeEvaluator {
    fn op_type(&self) -> &str {
        "Unsqueeze"
    }

    fn can_evaluate(&self, node: &NodeProto, ctx: &ConstantFoldingContext) -> bool {
        if node.op_type.as_str() != "Unsqueeze" {
            return false;
        }

        !node.input.is_empty()
            && node
                .input
                .iter()
                .filter(|name| !name.is_empty())
                .all(|name| ctx.is_constant(name))
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
                op: "Unsqueeze".to_string(),
            })?;

        let input_tensor = ctx.get_constant(input_name.as_str()).ok_or_else(|| {
            OnnxError::ShapeInference(format!("Input tensor '{}' not found", input_name))
        })?;

        let axes = constant_axes(node, ctx)?.ok_or_else(|| OnnxError::MissingAttribute {
            attr: "axes".into(),
            op: "Unsqueeze".into(),
        })?;
        let output_rank = input_tensor.shape.len() + axes.len();
        let axes = normalized_axes(&axes, output_rank)?;
        let mut output_shape = input_tensor.shape.clone();
        for axis in axes {
            output_shape.insert(axis, 1);
        }

        // Data remains the same, only shape changes
        let output = ConstantTensor {
            data: input_tensor.data.clone(),
            shape: output_shape,
            data_type: input_tensor.data_type,
        };

        Ok(vec![output])
    }
}

/// Squeeze operation: remove dimensions of size 1
pub struct SqueezeEvaluator;

impl EvaluatorTrait for SqueezeEvaluator {
    fn op_type(&self) -> &str {
        "Squeeze"
    }

    fn can_evaluate(&self, node: &NodeProto, ctx: &ConstantFoldingContext) -> bool {
        if node.op_type.as_str() != "Squeeze" {
            return false;
        }

        !node.input.is_empty()
            && node
                .input
                .iter()
                .filter(|name| !name.is_empty())
                .all(|name| ctx.is_constant(name))
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
                op: "Squeeze".to_string(),
            })?;

        let input_tensor = ctx.get_constant(input_name.as_str()).ok_or_else(|| {
            OnnxError::ShapeInference(format!("Input tensor '{}' not found", input_name))
        })?;

        let axes = constant_axes(node, ctx)?.unwrap_or_default();
        let output_shape = if axes.is_empty() {
            input_tensor
                .shape
                .iter()
                .copied()
                .filter(|&dim| dim != 1)
                .collect()
        } else {
            let axes = normalized_axes(&axes, input_tensor.shape.len())?;
            if axes.iter().any(|&axis| input_tensor.shape[axis] != 1) {
                return Err(OnnxError::InvalidShape(
                    "Squeeze axes must select dimensions of size 1".into(),
                ));
            }
            input_tensor
                .shape
                .iter()
                .enumerate()
                .filter_map(|(axis, &dim)| axes.binary_search(&axis).is_err().then_some(dim))
                .collect()
        };

        // Data remains the same, only shape changes
        let output = ConstantTensor {
            data: input_tensor.data.clone(),
            shape: output_shape,
            data_type: input_tensor.data_type,
        };

        Ok(vec![output])
    }
}

/// Cast operation: convert data type
pub struct CastEvaluator;

impl CastEvaluator {
    /// Cast tensor data to a new type
    fn cast_data(
        data: &TensorData,
        target_type: TensorProto_DataType,
    ) -> Result<TensorData, OnnxError> {
        match (data, target_type) {
            // Int64 -> other types
            (TensorData::Int64(v), TensorProto_DataType::Int64) => Ok(TensorData::Int64(v.clone())),
            (TensorData::Int64(v), TensorProto_DataType::Int32) => {
                Ok(TensorData::Int32(v.iter().map(|&x| x as i32).collect()))
            }
            (TensorData::Int64(v), TensorProto_DataType::Float) => {
                Ok(TensorData::Float32(v.iter().map(|&x| x as f32).collect()))
            }
            (TensorData::Int64(v), TensorProto_DataType::Double) => {
                Ok(TensorData::Float64(v.iter().map(|&x| x as f64).collect()))
            }

            // Int32 -> other types
            (TensorData::Int32(v), TensorProto_DataType::Int32) => Ok(TensorData::Int32(v.clone())),
            (TensorData::Int32(v), TensorProto_DataType::Int64) => {
                Ok(TensorData::Int64(v.iter().map(|&x| x as i64).collect()))
            }
            (TensorData::Int32(v), TensorProto_DataType::Float) => {
                Ok(TensorData::Float32(v.iter().map(|&x| x as f32).collect()))
            }
            (TensorData::Int32(v), TensorProto_DataType::Double) => {
                Ok(TensorData::Float64(v.iter().map(|&x| x as f64).collect()))
            }

            // Float32 -> other types
            (TensorData::Float32(v), TensorProto_DataType::Float) => {
                Ok(TensorData::Float32(v.clone()))
            }
            (TensorData::Float32(v), TensorProto_DataType::Double) => {
                Ok(TensorData::Float64(v.iter().map(|&x| x as f64).collect()))
            }
            (TensorData::Float32(v), TensorProto_DataType::Int64) => {
                Ok(TensorData::Int64(v.iter().map(|&x| x as i64).collect()))
            }
            (TensorData::Float32(v), TensorProto_DataType::Int32) => {
                Ok(TensorData::Int32(v.iter().map(|&x| x as i32).collect()))
            }

            // Float64 -> other types
            (TensorData::Float64(v), TensorProto_DataType::Double) => {
                Ok(TensorData::Float64(v.clone()))
            }
            (TensorData::Float64(v), TensorProto_DataType::Float) => {
                Ok(TensorData::Float32(v.iter().map(|&x| x as f32).collect()))
            }
            (TensorData::Float64(v), TensorProto_DataType::Int64) => {
                Ok(TensorData::Int64(v.iter().map(|&x| x as i64).collect()))
            }
            (TensorData::Float64(v), TensorProto_DataType::Int32) => {
                Ok(TensorData::Int32(v.iter().map(|&x| x as i32).collect()))
            }

            _ => Err(OnnxError::unsupported_op(
                "Cast",
                format!(
                    "cast from {:?} to {:?} not supported",
                    data.data_type(),
                    target_type
                ),
            )),
        }
    }
}

impl EvaluatorTrait for CastEvaluator {
    fn op_type(&self) -> &str {
        "Cast"
    }

    fn can_evaluate(&self, node: &NodeProto, ctx: &ConstantFoldingContext) -> bool {
        if node.op_type.as_str() != "Cast" {
            return false;
        }

        // Input must be constant
        if let Some(input) = node.input.first() {
            return ctx.is_constant(input.as_str());
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
                op: "Cast".to_string(),
            })?;

        let input_tensor = ctx.get_constant(input_name.as_str()).ok_or_else(|| {
            OnnxError::ShapeInference(format!("Input tensor '{}' not found", input_name))
        })?;

        // Get 'to' attribute (target type)
        let target_type_i32 = node
            .attribute
            .iter()
            .find(|a| a.name.as_str() == "to" && a.i != 0)
            .map(|a| a.i as i32)
            .ok_or_else(|| OnnxError::MissingAttribute {
                attr: "to".to_string(),
                op: "Cast".to_string(),
            })?;

        // Convert i32 to TensorProto_DataType enum
        let target_type = match target_type_i32 {
            1 => TensorProto_DataType::Float,
            2 => TensorProto_DataType::Uint8,
            3 => TensorProto_DataType::Int8,
            5 => TensorProto_DataType::Int16,
            6 => TensorProto_DataType::Int32,
            7 => TensorProto_DataType::Int64,
            11 => TensorProto_DataType::Double,
            _ => {
                return Err(OnnxError::unsupported_op(
                    "Cast",
                    format!("unsupported target type: {}", target_type_i32),
                ));
            }
        };

        // Perform cast
        let output_data = Self::cast_data(&input_tensor.data, target_type)?;

        let output = ConstantTensor {
            data: output_data,
            shape: input_tensor.shape.clone(),
            data_type: target_type.into(),
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
    fn test_unsqueeze() {
        let tensor = TensorProto {
            name: "input".to_string(),
            data_type: TensorProto_DataType::Int64.into(),
            dims: vec![3],
            int64_data: vec![2, 128, 384],
            ..Default::default()
        };

        let tensor_static: &'static TensorProto = Box::leak(Box::new(tensor));

        let mut init_map = HashMap::new();
        init_map.insert("input".to_string(), tensor_static);

        let ctx = ConstantFoldingContext::new(&init_map).unwrap();
        let evaluator = UnsqueezeEvaluator;

        // Unsqueeze at axes [0]
        let mut node = NodeProto {
            op_type: "Unsqueeze".to_string(),
            input: vec!["input".to_string()],
            output: vec!["output".to_string()],
            ..Default::default()
        };

        let axes_attr = AttributeProto {
            name: "axes".to_string(),
            ints: vec![0],
            ..Default::default()
        };
        node.attribute.push(axes_attr);

        let result = evaluator.evaluate(&node, &ctx).unwrap();
        let output = &result[0];

        assert_eq!(output.shape, vec![1, 3]); // Added dimension at front

        if let TensorData::Int64(ref values) = output.data {
            assert_eq!(values, &vec![2, 128, 384]); // Data unchanged
        } else {
            panic!("Expected Int64 data");
        }
    }

    #[test]
    fn test_squeeze() {
        let tensor = TensorProto {
            name: "input".to_string(),
            data_type: TensorProto_DataType::Int64.into(),
            dims: vec![1, 3, 1],
            int64_data: vec![2, 128, 384],
            ..Default::default()
        };

        let tensor_static: &'static TensorProto = Box::leak(Box::new(tensor));

        let mut init_map = HashMap::new();
        init_map.insert("input".to_string(), tensor_static);

        let ctx = ConstantFoldingContext::new(&init_map).unwrap();
        let evaluator = SqueezeEvaluator;

        // Squeeze all dimensions of size 1
        let node = NodeProto {
            op_type: "Squeeze".to_string(),
            input: vec!["input".to_string()],
            output: vec!["output".to_string()],
            ..Default::default()
        };

        let result = evaluator.evaluate(&node, &ctx).unwrap();
        let output = &result[0];

        assert_eq!(output.shape, vec![3]); // Removed dimensions of size 1

        if let TensorData::Int64(ref values) = output.data {
            assert_eq!(values, &vec![2, 128, 384]); // Data unchanged
        } else {
            panic!("Expected Int64 data");
        }
    }

    #[test]
    fn test_cast_int64_to_int32() {
        let tensor = TensorProto {
            name: "input".to_string(),
            data_type: TensorProto_DataType::Int64.into(),
            dims: vec![3],
            int64_data: vec![2, 128, 384],
            ..Default::default()
        };

        let tensor_static: &'static TensorProto = Box::leak(Box::new(tensor));

        let mut init_map = HashMap::new();
        init_map.insert("input".to_string(), tensor_static);

        let ctx = ConstantFoldingContext::new(&init_map).unwrap();
        let evaluator = CastEvaluator;

        // Cast to INT32
        let mut node = NodeProto {
            op_type: "Cast".to_string(),
            input: vec!["input".to_string()],
            output: vec!["output".to_string()],
            ..Default::default()
        };

        let to_attr = AttributeProto {
            name: "to".to_string(),
            i: 6, // INT32 = 6 in ONNX
            ..Default::default()
        };
        node.attribute.push(to_attr);

        let result = evaluator.evaluate(&node, &ctx).unwrap();
        let output = &result[0];

        assert_eq!(output.data_type, TensorProto_DataType::Int32 as i32);

        if let TensorData::Int32(ref values) = output.data {
            assert_eq!(values, &vec![2, 128, 384]);
        } else {
            panic!("Expected Int32 data");
        }
    }
    fn reshape_case(
        op: &str,
        shape: &[i64],
        axes: Option<&[i64]>,
        input_axes: bool,
    ) -> Result<ConstantTensor, OnnxError> {
        let initializers = HashMap::new();
        let mut ctx = ConstantFoldingContext::new(&initializers)?;
        ctx.add_constant(
            "data".into(),
            ConstantTensor {
                data: TensorData::Int64(vec![10, 20]),
                shape: shape.to_vec(),
                data_type: TensorProto_DataType::Int64 as i32,
            },
        );
        let mut node = NodeProto {
            op_type: op.into(),
            input: vec!["data".into()],
            output: vec!["y".into()],
            ..Default::default()
        };
        if let Some(axes) = axes {
            if input_axes {
                node.input.push("axes".into());
                ctx.add_constant(
                    "axes".into(),
                    ConstantTensor {
                        data: TensorData::Int64(axes.to_vec()),
                        shape: vec![axes.len() as i64],
                        data_type: TensorProto_DataType::Int64 as i32,
                    },
                );
            } else {
                node.attribute.push(AttributeProto {
                    name: "axes".into(),
                    ints: axes.to_vec(),
                    ..Default::default()
                });
            }
        }
        let outputs = if op == "Squeeze" {
            SqueezeEvaluator.evaluate(&node, &ctx)?
        } else {
            UnsqueezeEvaluator.evaluate(&node, &ctx)?
        };
        Ok(outputs.into_iter().next().unwrap())
    }

    #[test]
    fn squeeze_axes_select_only_requested_unit_dimensions() {
        for input_axes in [false, true] {
            for (axes, expected) in [
                (Some(vec![0]), vec![2, 1]),
                (Some(vec![-1]), vec![1, 2]),
                (Some(vec![-3, 2]), vec![2]),
                (None, vec![2]),
            ] {
                let output =
                    reshape_case("Squeeze", &[1, 2, 1], axes.as_deref(), input_axes).unwrap();
                assert_eq!(output.shape, expected);
                assert!(matches!(&output.data, TensorData::Int64(values) if values == &[10, 20]));
            }
            for axes in [vec![1], vec![3], vec![-4], vec![0, 0], vec![0, -3]] {
                assert!(reshape_case("Squeeze", &[1, 2, 1], Some(&axes), input_axes).is_err());
            }
        }
    }

    #[test]
    fn unsqueeze_axes_are_order_independent_and_relative_to_final_rank() {
        for input_axes in [false, true] {
            for (axes, expected) in [
                (vec![-2, -1], vec![2, 1, 1]),
                (vec![2, 1], vec![2, 1, 1]),
                (vec![-1, 1], vec![2, 1, 1]),
                (vec![2, 0], vec![1, 2, 1]),
            ] {
                let output = reshape_case("Unsqueeze", &[2], Some(&axes), input_axes).unwrap();
                assert_eq!(output.shape, expected);
                assert!(matches!(&output.data, TensorData::Int64(values) if values == &[10, 20]));
            }
            for axes in [vec![-3], vec![2], vec![0, 0], vec![0, -3]] {
                assert!(reshape_case("Unsqueeze", &[2], Some(&axes), input_axes).is_err());
            }
        }
    }

    #[test]
    fn runtime_axes_prevent_reshape_folding() {
        let initializers = HashMap::new();
        let mut ctx = ConstantFoldingContext::new(&initializers).unwrap();
        ctx.add_constant(
            "data".into(),
            ConstantTensor {
                data: TensorData::Int64(vec![10, 20]),
                shape: vec![1, 2, 1],
                data_type: TensorProto_DataType::Int64 as i32,
            },
        );
        for op in ["Squeeze", "Unsqueeze"] {
            let node = NodeProto {
                op_type: op.into(),
                input: vec!["data".into(), "runtime_axes".into()],
                ..Default::default()
            };
            let evaluator: &dyn EvaluatorTrait = if op == "Squeeze" {
                &SqueezeEvaluator
            } else {
                &UnsqueezeEvaluator
            };
            assert!(!evaluator.can_evaluate(&node, &ctx));
            assert!(evaluator.evaluate(&node, &ctx).is_err());
        }
    }
}
