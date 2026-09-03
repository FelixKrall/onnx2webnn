//! Cache reload and deterministic native-ORT validation for exported WebNN graphs.
use crate::onnx::builder::OnnxBuilder;
use crate::onnx::convert::{OnnxError, ValidatedGraph};
use crate::protos::onnx::{
    tensor_shape_proto::dimension::Value as DimensionValue, type_proto::Value as TypeProtoValue,
    ModelProto, TensorProto_DataType, ValueInfoProto,
};
use half::f16;
use prost::Message;
use rustnn::graph::OperandDescriptor;
use rustnn::mlcontext::{
    MLContext, MLContextOptions, MLPowerPreference, MLTensor, MLTensorDescriptor,
};
use rustnn::operator_enums::MLOperandDataType;
use rustnn::{load_graph_from_path, run_onnx_with_inputs, OnnxInput, TensorData};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ValidationSummary {
    pub input_count: usize,
    pub output_count: usize,
}

/// Reload saved WebNN artifacts, dispatch deterministic fixed-shape inputs,
/// and compare results with native ONNX Runtime on the cached ONNX model.
pub fn validate_cached_model(
    onnx_path: impl AsRef<Path>,
    webnn_path: impl AsRef<Path>,
) -> Result<ValidationSummary, OnnxError> {
    validate_cached_model_with_overrides(onnx_path, webnn_path, &HashMap::new())
}

/// Validate cached artifacts using explicit bindings for symbolic ONNX input dimensions.
pub fn validate_cached_model_with_overrides(
    onnx_path: impl AsRef<Path>,
    webnn_path: impl AsRef<Path>,
    free_dim_overrides: &HashMap<String, u32>,
) -> Result<ValidationSummary, OnnxError> {
    let onnx_bytes = fs::read(onnx_path.as_ref())?;
    let model = ModelProto::decode(onnx_bytes.as_slice())
        .map_err(|e| OnnxError::ProtobufError(e.to_string()))?;
    let inputs = deterministic_inputs(&model, free_dim_overrides)?;
    let reference = run_onnx_with_inputs(&onnx_bytes, None, clone_inputs(&inputs))
        .map_err(|e| OnnxError::Validation(format!("native ORT run failed: {e}")))?;
    let graph_info = load_graph_from_path(webnn_path.as_ref())
        .map_err(|e| OnnxError::Validation(format!("failed to reload WebNN cache: {e}")))?;
    let mut context = MLContext::create(&MLContextOptions::new(MLPowerPreference::Default, false))
        .map_err(|e| OnnxError::Validation(format!("MLContext::create failed: {e}")))?;
    let graph = context
        .rustnn_build_graph(graph_info)
        .map_err(|e| OnnxError::Validation(format!("cached graph build failed: {e}")))?;
    let mut validated = ValidatedGraph { context, graph };
    let actual = dispatch_and_collect(&mut validated, &model, &inputs)?;
    compare_outputs(&model, &reference, &actual)?;
    Ok(ValidationSummary {
        input_count: inputs.len(),
        output_count: reference.len(),
    })
}

fn graph(model: &ModelProto) -> Result<&crate::protos::onnx::GraphProto, OnnxError> {
    model
        .graph
        .as_ref()
        .ok_or_else(|| OnnxError::ProtobufError("Missing graph in model".to_string()))
}

fn tensor_dims(
    vi: &ValueInfoProto,
    free_dim_overrides: &HashMap<String, u32>,
) -> Result<(i32, Vec<usize>), OnnxError> {
    let ty = vi
        .r#type
        .as_ref()
        .and_then(|ty| ty.value.as_ref())
        .ok_or_else(|| OnnxError::Validation(format!("missing tensor type for {}", vi.name)))?;
    let tensor = match ty {
        TypeProtoValue::TensorType(tensor) => tensor,
        _ => {
            return Err(OnnxError::Validation(format!(
                "non-tensor input {}",
                vi.name
            )))
        }
    };
    let shape = tensor
        .shape
        .as_ref()
        .ok_or_else(|| OnnxError::Validation(format!("missing shape for {}", vi.name)))?;
    let dims = shape
        .dim
        .iter()
        .map(|dim| match dim.value.as_ref() {
            Some(DimensionValue::DimValue(value)) if *value >= 0 => Ok(*value as usize),
            Some(DimensionValue::DimParam(name)) => free_dim_overrides
                .get(name)
                .map(|value| *value as usize)
                .ok_or_else(|| {
                    OnnxError::Validation(format!(
                        "dynamic input dimension {name} in {}; provide an override for validation",
                        vi.name
                    ))
                }),
            _ => Err(OnnxError::Validation(format!(
                "dynamic input dimension in {}; use fixed shapes for validation",
                vi.name
            ))),
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok((tensor.elem_type, dims))
}

fn feedable_inputs(model: &ModelProto) -> Result<Vec<&ValueInfoProto>, OnnxError> {
    let graph = graph(model)?;
    let initializer_names: HashSet<&str> =
        graph.initializer.iter().map(|t| t.name.as_str()).collect();
    Ok(graph
        .input
        .iter()
        .filter(|input| !initializer_names.contains(input.name.as_str()))
        .collect())
}

fn deterministic_inputs(
    model: &ModelProto,
    free_dim_overrides: &HashMap<String, u32>,
) -> Result<Vec<OnnxInput>, OnnxError> {
    feedable_inputs(model)?
        .into_iter()
        .map(|input| {
            let (elem_type, shape) = tensor_dims(input, free_dim_overrides)?;
            let count = shape.iter().product::<usize>().max(1);
            let data = match elem_type {
                x if x == TensorProto_DataType::Float as i32 => {
                    TensorData::Float32((0..count).map(|i| 0.125 * (i as f32 + 1.0)).collect())
                }
                x if x == TensorProto_DataType::Float16 as i32 => TensorData::Float16(
                    (0..count)
                        .map(|i| f16::from_f32(0.125 * (i as f32 + 1.0)).to_bits())
                        .collect(),
                ),
                x if x == TensorProto_DataType::Int32 as i32 => {
                    TensorData::Int32((0..count).map(|i| (i % 7) as i32).collect())
                }
                x if x == TensorProto_DataType::Int64 as i32 => {
                    TensorData::Int64((0..count).map(|i| (i % 7) as i64).collect())
                }
                x if x == TensorProto_DataType::Bool as i32 => {
                    TensorData::Uint8((0..count).map(|i| u8::from(i % 2 == 0)).collect())
                }
                x if x == TensorProto_DataType::Uint8 as i32 => {
                    TensorData::Uint8((0..count).map(|i| (i % 255) as u8).collect())
                }
                other => {
                    return Err(OnnxError::Validation(format!(
                        "unsupported deterministic input dtype {other} for {}",
                        input.name
                    )))
                }
            };
            Ok(OnnxInput {
                name: input.name.clone(),
                shape,
                data,
            })
        })
        .collect()
}

fn clone_inputs(inputs: &[OnnxInput]) -> Vec<OnnxInput> {
    inputs
        .iter()
        .map(|input| OnnxInput {
            name: input.name.clone(),
            shape: input.shape.clone(),
            data: match &input.data {
                TensorData::Float32(v) => TensorData::Float32(v.clone()),
                TensorData::Float16(v) => TensorData::Float16(v.clone()),
                TensorData::Int8(v) => TensorData::Int8(v.clone()),
                TensorData::Uint8(v) => TensorData::Uint8(v.clone()),
                TensorData::Int32(v) => TensorData::Int32(v.clone()),
                TensorData::Uint32(v) => TensorData::Uint32(v.clone()),
                TensorData::Int64(v) => TensorData::Int64(v.clone()),
                TensorData::Uint64(v) => TensorData::Uint64(v.clone()),
            },
        })
        .collect()
}

fn tensor_descriptor(desc: &OperandDescriptor) -> MLTensorDescriptor {
    let data_type = MLOperandDataType::try_from(desc.data_type).expect("WebNN operand type");
    let mut tensor = MLTensorDescriptor::new(
        data_type,
        desc.static_or_max_shape()
            .into_iter()
            .map(u64::from)
            .collect(),
    );
    tensor.set_readable(true);
    tensor.set_writable(true);
    tensor
}

fn write_input(
    context: &mut MLContext,
    tensor: &MLTensor,
    input: &OnnxInput,
) -> Result<(), OnnxError> {
    let result = match &input.data {
        TensorData::Float32(data) => context.write_tensor(tensor, data),
        TensorData::Float16(data) => context.write_tensor(tensor, data),
        TensorData::Int32(data) => context.write_tensor(tensor, data),
        TensorData::Int64(data) => context.write_tensor(tensor, data),
        TensorData::Uint8(data) => context.write_tensor(tensor, data),
        _ => {
            return Err(OnnxError::Validation(format!(
                "unsupported input write for {}",
                input.name
            )))
        }
    };
    result.map_err(|e| OnnxError::Validation(format!("failed to write {}: {e}", input.name)))
}

fn read_output(
    context: &mut MLContext,
    tensor: &MLTensor,
    desc: &OperandDescriptor,
) -> Result<Vec<f64>, OnnxError> {
    let count = desc.element_count().unwrap_or(1).max(1);
    macro_rules! read {
        ($type:ty) => {{
            let mut data = vec![<$type>::default(); count];
            context
                .read_tensor(tensor, &mut data)
                .map_err(|e| OnnxError::Validation(format!("failed to read output: {e}")))?;
            data.into_iter().map(|v| v as f64).collect()
        }};
    }
    Ok(match desc.data_type {
        rustnn::DataType::Float32 => read!(f32),
        rustnn::DataType::Float16 => {
            let mut data = vec![0u16; count];
            context
                .read_tensor(tensor, &mut data)
                .map_err(|e| OnnxError::Validation(format!("failed to read output: {e}")))?;
            data.into_iter()
                .map(|v| f64::from(f16::from_bits(v).to_f32()))
                .collect()
        }
        rustnn::DataType::Int32 => read!(i32),
        rustnn::DataType::Int64 => read!(i64),
        rustnn::DataType::Uint8 => read!(u8),
        other => {
            return Err(OnnxError::Validation(format!(
                "unsupported output dtype {other:?}"
            )))
        }
    })
}

fn dispatch_and_collect(
    validated: &mut ValidatedGraph,
    model: &ModelProto,
    inputs: &[OnnxInput],
) -> Result<HashMap<String, Vec<f64>>, OnnxError> {
    let graph_proto = graph(model)?;
    let input_names: HashSet<String> = feedable_inputs(model)?
        .iter()
        .map(|input| OnnxBuilder::webnn_id(&input.name))
        .collect();
    let mut input_storage = Vec::new();
    let mut input_keys = Vec::new();
    for input in inputs {
        let key = OnnxBuilder::webnn_id(&input.name);
        let desc = validated.graph.input_descriptors.get(&key).ok_or_else(|| {
            OnnxError::Validation(format!("cached graph missing input descriptor {key}"))
        })?;
        let tensor = validated
            .context
            .create_tensor(&tensor_descriptor(desc))
            .map_err(|e| OnnxError::Validation(format!("failed to create {key}: {e}")))?;
        write_input(&mut validated.context, &tensor, input)?;
        input_keys.push(key);
        input_storage.push(tensor);
    }
    let input_bindings: BTreeMap<&str, &MLTensor> = input_keys
        .iter()
        .zip(input_storage.iter())
        .map(|(name, tensor)| (name.as_str(), tensor))
        .collect();
    let mut output_storage = Vec::new();
    let mut output_keys = Vec::new();
    let mut output_map = HashMap::new();
    for output in &graph_proto.output {
        let key = OnnxBuilder::output_key_for(&output.name, &input_names);
        let desc = validated
            .graph
            .output_descriptors
            .get(&key)
            .ok_or_else(|| {
                OnnxError::Validation(format!("cached graph missing output descriptor {key}"))
            })?;
        let tensor = validated
            .context
            .create_tensor(&tensor_descriptor(desc))
            .map_err(|e| OnnxError::Validation(format!("failed to create {key}: {e}")))?;
        output_keys.push(key.clone());
        output_storage.push(tensor);
        output_map.insert(output.name.clone(), key);
    }
    let output_bindings: BTreeMap<&str, &MLTensor> = output_keys
        .iter()
        .zip(output_storage.iter())
        .map(|(name, tensor)| (name.as_str(), tensor))
        .collect();
    validated
        .context
        .dispatch(&mut validated.graph, &input_bindings, &output_bindings)
        .map_err(|e| OnnxError::Validation(format!("cached graph dispatch failed: {e}")))?;
    output_map
        .into_iter()
        .map(|(onnx_name, key)| {
            let desc = validated
                .graph
                .output_descriptors
                .get(&key)
                .expect("validated above");
            let tensor = output_bindings.get(key.as_str()).expect("bound above");
            read_output(&mut validated.context, tensor, desc).map(|values| (onnx_name, values))
        })
        .collect()
}

fn compare_outputs(
    model: &ModelProto,
    reference: &[rustnn::OnnxOutputWithData],
    actual: &HashMap<String, Vec<f64>>,
) -> Result<(), OnnxError> {
    let outputs = &graph(model)?.output;
    if outputs.len() != reference.len() {
        return Err(OnnxError::Validation(
            "native ORT output count mismatch".to_string(),
        ));
    }
    for (output, expected) in outputs.iter().zip(reference) {
        let got = actual.get(&output.name).ok_or_else(|| {
            OnnxError::Validation(format!("cached graph did not produce {}", output.name))
        })?;
        if expected.data.len() != got.len() {
            return Err(OnnxError::Validation(format!(
                "{} length mismatch: ORT={}, WebNN={}",
                output.name,
                expected.data.len(),
                got.len()
            )));
        }
        for (index, (expected, actual)) in expected.data.iter().zip(got).enumerate() {
            if expected.is_nan() && actual.is_nan() {
                continue;
            }
            let tolerance = 1e-5 + expected.abs() * 1e-4;
            if (expected - actual).abs() > tolerance {
                return Err(OnnxError::Validation(format!(
                    "{}[{index}] mismatch: ORT={expected}, WebNN={actual}, tolerance={tolerance}",
                    output.name
                )));
            }
        }
    }
    Ok(())
}
