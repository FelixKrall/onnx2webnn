/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Tarek Ziadé <tarek@ziade.org>
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// MatMul, Gemm, and MatMulNBits (com.microsoft) operator handlers

use crate::onnx::builder::{map_op_error, operand_index, tensor_proto_to_bytes, OnnxBuilder};
use crate::onnx::builder_helpers::{
    i64_slice_to_mldim, output_label, record_node_output, reshape_with_shape,
};
use crate::onnx::convert::OnnxError;
use crate::onnx::ops::{ConversionContext, ConversionResult, OpHandler};
use crate::protos::onnx::{NodeProto, TensorProto_DataType};
use rustnn::mlcontext::MLOperand;
use rustnn::operator_options::{MLGemmOptions, MLTransposeOptions};
use rustnn::DataType;

pub struct MatMulHandler;

impl OpHandler for MatMulHandler {
    fn supports(&self, op_type: &str) -> bool {
        matches!(op_type, "MatMul" | "Gemm" | "MatMulNBits")
    }

    fn convert(
        &self,
        node: &NodeProto,
        context: &ConversionContext,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let op_type = node.op_type.as_str();
        let node_name = if !node.name.is_empty() {
            node.name.clone()
        } else {
            "unnamed".to_string()
        };

        match op_type {
            "MatMul" => self.convert_matmul(node, &node_name, b),
            "Gemm" => self.convert_gemm(node, &node_name, context, b),
            "MatMulNBits" => self.convert_matmul_nbits(node, &node_name, context, b),
            _ => Err(OnnxError::unsupported_op(op_type.to_string(), node_name)),
        }
    }
}

impl MatMulHandler {
    fn convert_matmul(
        &self,
        node: &NodeProto,
        node_name: &str,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.len() != 2 {
            return Err(OnnxError::InvalidShape(format!(
                "MatMul expects 2 inputs, got {}",
                inputs.len()
            )));
        }

        let output_name = output_label(node, node_name);
        let a = b.resolve_operand(&inputs[0])?;
        let b_in = b.resolve_operand(&inputs[1])?;
        let opts = OnnxBuilder::labeled_options(&output_name);
        let out = b
            .builder
            .matmul_with_options(a, b_in, opts)
            .map_err(map_op_error)?;

        if let Some(onnx_out) = node.output.first() {
            record_node_output(b, onnx_out, &output_name, out);
        } else {
            b.record_operand(&[&output_name], out);
        }
        Ok(ConversionResult::default())
    }

    fn convert_gemm(
        &self,
        node: &NodeProto,
        node_name: &str,
        _context: &ConversionContext,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.len() < 2 {
            return Err(OnnxError::InvalidShape(format!(
                "Gemm expects at least 2 inputs, got {}",
                inputs.len()
            )));
        }

        let mut alpha = 1.0f64;
        let mut beta = 1.0f64;
        let mut trans_a = false;
        let mut trans_b = false;
        for attr in node.attribute.as_slice() {
            match attr.name.as_str() {
                "alpha" if attr.f != 0.0 => alpha = attr.f as f64,
                "beta" if attr.f != 0.0 => beta = attr.f as f64,
                "transA" if attr.i != 0 => trans_a = true,
                "transB" if attr.i != 0 => trans_b = true,
                _ => {}
            }
        }

        let output_name = output_label(node, node_name);
        let a = b.resolve_operand(&inputs[0])?;
        let b_in = b.resolve_operand(&inputs[1])?;
        let c = inputs
            .get(2)
            .map(|name| b.resolve_operand(name))
            .transpose()?;

        let opts = MLGemmOptions {
            label: output_name.clone(),
            alpha,
            beta,
            a_transpose: trans_a,
            b_transpose: trans_b,
            c: c.map(operand_index),
        };
        let out = b
            .builder
            .gemm_with_options(a, b_in, opts)
            .map_err(map_op_error)?;

        if let Some(onnx_out) = node.output.first() {
            record_node_output(b, onnx_out, &output_name, out);
        } else {
            b.record_operand(&[&output_name], out);
        }
        Ok(ConversionResult::default())
    }

    /// Lower `com.microsoft.MatMulNBits` the same way ORT's WebNN EP does:
    /// `dequantizeLinear` → reshape `[N,K]` → transpose `[K,N]` → `matmul` (+ optional bias).
    ///
    /// Supported: bits=4, constant packed `B`, optional constant zero_points, optional bias.
    /// Rejected: bits≠4, `g_idx`, non-constant `B`/`zero_points`.
    fn convert_matmul_nbits(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.len() < 3 {
            return Err(OnnxError::InvalidShape(format!(
                "MatMulNBits expects at least 3 inputs (A, B, scales), got {}",
                inputs.len()
            )));
        }

        let mut k = 0i64;
        let mut n = 0i64;
        let mut bits = 4i64;
        let mut block_size = 32i64;
        for attr in &node.attribute {
            match attr.name.as_str() {
                "K" => k = attr.i,
                "N" => n = attr.i,
                "bits" => bits = attr.i,
                "block_size" => block_size = attr.i,
                _ => {}
            }
        }
        if bits != 4 {
            return Err(OnnxError::unsupported_op(
                format!("MatMulNBits(bits={bits})"),
                node_name.to_string(),
            ));
        }
        if k <= 0 || n <= 0 || block_size < 16 || !(block_size as u64).is_power_of_two() {
            return Err(OnnxError::InvalidShape(format!(
                "MatMulNBits requires positive K/N and power-of-two block_size≥16, \
                 got K={k} N={n} block_size={block_size}"
            )));
        }
        if inputs.get(4).is_some_and(|name| !name.is_empty()) {
            return Err(OnnxError::unsupported_op(
                "MatMulNBits(g_idx)",
                node_name.to_string(),
            ));
        }

        let b_name = inputs[1].as_str();
        let scales_name = inputs[2].as_str();
        let zero_points_name = inputs
            .get(3)
            .filter(|name| !name.is_empty())
            .map(String::as_str);
        let bias_name = inputs
            .get(5)
            .filter(|name| !name.is_empty())
            .map(String::as_str);

        let b_tensor = context.initializers.get(b_name).copied().ok_or_else(|| {
            OnnxError::unsupported_op("MatMulNBits(non-constant B)", node_name.to_string())
        })?;
        if b_tensor.data_type != TensorProto_DataType::Uint8 as i32 {
            return Err(OnnxError::InvalidShape(format!(
                "MatMulNBits B must be uint8 packed weights, got data_type={}",
                b_tensor.data_type
            )));
        }
        if b_tensor.dims.len() != 3 {
            return Err(OnnxError::InvalidShape(format!(
                "MatMulNBits B must have shape [N, n_blocks, blob_size], got {:?}",
                b_tensor.dims
            )));
        }
        let n_attr = n as u32;
        let k_attr = k as u32;
        let block_size_u = block_size as u32;
        let n_blocks = b_tensor.dims[1] as u32;
        let blob_size = b_tensor.dims[2] as u32;
        if b_tensor.dims[0] != n {
            return Err(OnnxError::InvalidShape(format!(
                "MatMulNBits B dim0 {} does not match N={n}",
                b_tensor.dims[0]
            )));
        }
        let expected_blocks = k_attr.div_ceil(block_size_u);
        if n_blocks != expected_blocks {
            return Err(OnnxError::InvalidShape(format!(
                "MatMulNBits n_blocks {n_blocks} != ceil(K/block_size)={expected_blocks}"
            )));
        }
        let expected_blob = (block_size_u * 4).div_ceil(8);
        if blob_size != expected_blob {
            return Err(OnnxError::InvalidShape(format!(
                "MatMulNBits blob_size {blob_size} != block_size*bits/8={expected_blob}"
            )));
        }

        let label = output_label(node, node_name);
        let packed = tensor_proto_to_bytes(b_tensor)?;
        // Reinterpret packed uint8 blobs as uint4 with doubled last dim (= block_size).
        let uint4_shape = [n_attr, n_blocks, blob_size * 2];
        let b_uint4_name = format!("{label}__B_uint4");
        b.register_constant_from_bytes(&b_uint4_name, DataType::Uint4, &uint4_shape, &packed)?;
        let b_uint4 = b.resolve_operand(&b_uint4_name)?;

        let scales = b.resolve_operand(scales_name)?;
        let scales = reshape_with_shape(
            b,
            scales,
            &format!("{label}__scales"),
            i64_slice_to_mldim(&[n, n_blocks as i64, 1])?,
        )?;

        let zero_point = register_matmul_nbits_zero_point(
            b,
            context,
            zero_points_name,
            n_attr,
            n_blocks,
            &format!("{label}__zero_point"),
        )?;

        let dequantized = b
            .builder
            .dequantize_linear_with_zeropoint(b_uint4, scales, zero_point)
            .map_err(map_op_error)?;
        let weights = reshape_with_shape(
            b,
            dequantized,
            &format!("{label}__weights_nk"),
            i64_slice_to_mldim(&[n, k])?,
        )?;
        let weights = b
            .builder
            .transpose_with_options(
                weights,
                MLTransposeOptions {
                    label: format!("{label}__weights_kn"),
                    permutation: vec![1, 0],
                },
            )
            .map_err(map_op_error)?;

        let a = b.resolve_operand(&inputs[0])?;
        let mut out = b
            .builder
            .matmul_with_options(a, weights, OnnxBuilder::labeled_options(&label))
            .map_err(map_op_error)?;
        if let Some(bias_name) = bias_name {
            let bias = b.resolve_operand(bias_name)?;
            out = b
                .builder
                .add_with_options(
                    out,
                    bias,
                    OnnxBuilder::labeled_options(&format!("{label}__bias")),
                )
                .map_err(map_op_error)?;
        }

        if let Some(onnx_out) = node.output.first().filter(|name| !name.is_empty()) {
            record_node_output(b, onnx_out, &label, out);
        }
        Ok(ConversionResult::default())
    }
}

fn register_matmul_nbits_zero_point(
    b: &mut OnnxBuilder<'_, '_, '_>,
    context: &ConversionContext,
    zero_points_name: Option<&str>,
    n: u32,
    n_blocks: u32,
    label: &str,
) -> Result<MLOperand, OnnxError> {
    let zp_shape = [n, n_blocks, 1];
    let element_count = (n as usize)
        .checked_mul(n_blocks as usize)
        .ok_or_else(|| OnnxError::InvalidShape("MatMulNBits zero_point size overflow".into()))?;
    let packed_len = element_count.div_ceil(2);

    let packed = if let Some(name) = zero_points_name {
        let tensor = context.initializers.get(name).copied().ok_or_else(|| {
            OnnxError::unsupported_op("MatMulNBits(non-constant zero_points)", label.to_string())
        })?;
        if tensor.data_type != TensorProto_DataType::Uint8 as i32 {
            return Err(OnnxError::InvalidShape(format!(
                "MatMulNBits zero_points must be packed uint8, got data_type={}",
                tensor.data_type
            )));
        }
        let bytes = tensor_proto_to_bytes(tensor)?;
        if bytes.len() != packed_len {
            return Err(OnnxError::InvalidShape(format!(
                "MatMulNBits zero_points packed length {} != expected {packed_len}",
                bytes.len()
            )));
        }
        bytes
    } else {
        // Default uint4 zero point is 8 → packed nibbles 0x88.
        vec![0x88u8; packed_len]
    };

    b.register_constant_from_bytes(label, DataType::Uint4, &zp_shape, &packed)?;
    b.resolve_operand(label)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protos::onnx::{AttributeProto, TensorProto, TensorProto_DataType};
    use rustnn::graph::pack_uint4;
    use std::collections::HashMap;

    fn create_test_node(op_type: &str, inputs: Vec<&str>, outputs: Vec<&str>) -> NodeProto {
        NodeProto {
            op_type: op_type.to_string(),
            name: format!("test_{}", op_type.to_lowercase()),
            input: inputs.iter().map(|s| s.to_string()).collect(),
            output: outputs.iter().map(|s| s.to_string()).collect(),
            ..Default::default()
        }
    }

    #[test]
    fn test_matmul_handler_supports() {
        let handler = MatMulHandler;
        assert!(handler.supports("MatMul"));
        assert!(handler.supports("Gemm"));
        assert!(handler.supports("MatMulNBits"));
    }

    #[test]
    fn test_convert_matmul() {
        let handler = MatMulHandler;
        let node = create_test_node("MatMul", vec!["a", "b"], vec!["c"]);
        crate::onnx::ops::convert_with_test_builder(&handler, &node).unwrap();
    }

    #[test]
    fn test_convert_gemm_simple() {
        let handler = MatMulHandler;
        let node = create_test_node("Gemm", vec!["a", "b"], vec!["c"]);
        crate::onnx::ops::convert_with_test_builder(&handler, &node).unwrap();
    }

    #[test]
    fn converts_matmul_nbits_4bit_without_zero_points() {
        let handler = MatMulHandler;
        let mut node = create_test_node("MatMulNBits", vec!["a", "b_q4", "scales"], vec!["y"]);
        node.domain = "com.microsoft".to_string();
        node.attribute = vec![
            AttributeProto {
                name: "K".to_string(),
                i: 32,
                ..Default::default()
            },
            AttributeProto {
                name: "N".to_string(),
                i: 16,
                ..Default::default()
            },
            AttributeProto {
                name: "bits".to_string(),
                i: 4,
                ..Default::default()
            },
            AttributeProto {
                name: "block_size".to_string(),
                i: 32,
                ..Default::default()
            },
        ];

        // B: [N=16, n_blocks=1, blob_size=16] packed uint8 (= 512 uint4 values).
        let values: Vec<u8> = (0..512).map(|v| (v % 16) as u8).collect();
        let packed = pack_uint4(&values);
        let b_tensor = TensorProto {
            name: "b_q4".to_string(),
            data_type: TensorProto_DataType::Uint8 as i32,
            dims: vec![16, 1, 16],
            raw_data: packed,
            ..Default::default()
        };
        let scale_bytes: Vec<u8> = (0..16).flat_map(|_| 0.5f32.to_le_bytes()).collect();
        let scales = TensorProto {
            name: "scales".to_string(),
            data_type: TensorProto_DataType::Float as i32,
            dims: vec![16, 1],
            raw_data: scale_bytes,
            ..Default::default()
        };

        let mut initializers = HashMap::new();
        initializers.insert("b_q4".to_string(), &b_tensor);
        initializers.insert("scales".to_string(), &scales);
        let value_shapes = HashMap::from([
            ("a".to_string(), vec![2, 32]),
            ("b_q4".to_string(), vec![16, 1, 16]),
            ("scales".to_string(), vec![16, 1]),
        ]);
        let value_types = HashMap::from([
            ("a".to_string(), DataType::Float32),
            ("b_q4".to_string(), DataType::Uint8),
            ("scales".to_string(), DataType::Float32),
        ]);
        let const_values = HashMap::new();
        let value_ids = HashMap::new();
        let context = ConversionContext {
            initializers: &initializers,
            value_shapes: &value_shapes,
            value_shape_dims: crate::onnx::ops::empty_value_shape_dims(),
            const_values: &const_values,
            value_ids: &value_ids,
            value_types: &value_types,
        };

        crate::onnx::ops::convert_handler_with_context(&handler, &node, &context).unwrap();
    }

    #[test]
    fn rejects_matmul_nbits_with_g_idx() {
        let handler = MatMulHandler;
        let mut node = create_test_node(
            "MatMulNBits",
            vec!["a", "b", "scales", "", "g_idx"],
            vec!["y"],
        );
        node.attribute = vec![
            AttributeProto {
                name: "K".to_string(),
                i: 32,
                ..Default::default()
            },
            AttributeProto {
                name: "N".to_string(),
                i: 16,
                ..Default::default()
            },
            AttributeProto {
                name: "bits".to_string(),
                i: 4,
                ..Default::default()
            },
            AttributeProto {
                name: "block_size".to_string(),
                i: 32,
                ..Default::default()
            },
        ];
        let err = crate::onnx::ops::convert_with_test_builder(&handler, &node).unwrap_err();
        assert!(matches!(err, OnnxError::UnsupportedOps(_)));
    }
}
