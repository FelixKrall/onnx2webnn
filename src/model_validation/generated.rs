/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Deterministic generated-weight ONNX fixtures for numerical validation.

use crate::model_validation::full_model::cache_root;
use crate::model_validation::skeleton::{strip_model, HubSource, KEEP_BYTES};
use crate::onnx::ops::OpRegistry;
use crate::protos::onnx::{GraphProto, ModelProto, NodeProto, StringStringEntryProto, TensorProto};
use half::{bf16, f16};
use prost::Message;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use std::time::Duration;

pub const GENERATOR_VERSION: u32 = 4;

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
enum TensorRole {
    Exact,
    Learned,
    QuantScale,
    QuantZeroPoint,
}

#[derive(Debug, Deserialize, Serialize)]
struct TensorRecord {
    graph: String,
    name: String,
    role: TensorRole,
    reason: String,
    bytes: usize,
    digest: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct GeneratedMetadata {
    format: u32,
    source: String,
    revision: String,
    source_fingerprint: String,
    model_length: u64,
    data_length: u64,
    tensors: Vec<TensorRecord>,
}

#[derive(Clone, Debug)]
struct SourceLocation {
    location: String,
    offset: u64,
    length: usize,
    inline_source_offset: Option<u64>,
    inline_encoding: Option<String>,
}

pub fn cache_generated_model(
    file: &str,
    requested_revision: Option<&str>,
) -> Result<PathBuf, String> {
    let relative = safe_relative(file)?;
    let target = cache_root().join("generated").join(relative);
    let data_name = format!(
        "{}.generated.data",
        target
            .file_name()
            .and_then(|v| v.to_str())
            .unwrap_or("model.onnx")
    );
    let data_path = target.with_file_name(&data_name);
    let metadata_path = target.with_extension("generated.complete.json");
    let refresh = std::env::var_os("O2W_MODEL_CACHE_REFRESH").is_some();
    if !refresh && complete(&target, &data_path, &metadata_path, requested_revision) {
        return Ok(target);
    }
    if metadata_path.exists() {
        fs::remove_file(&metadata_path)
            .map_err(|e| format!("remove stale {}: {e}", metadata_path.display()))?;
    }

    let revision = match requested_revision {
        Some(revision) => revision.to_string(),
        None => resolve_revision(file)?,
    };
    let source = HubSource::open_revision(file, &revision)?;
    let (skeleton, _) = strip_model(source, KEEP_BYTES)?;
    let source_fingerprint = hex(&Sha256::digest(&skeleton));
    let mut model = ModelProto::decode(skeleton.as_slice())
        .map_err(|e| format!("decode generated-weight skeleton for {file}: {e}"))?;
    let source_spec = HubSpec::parse(file, &revision)?;
    let mut sidecar = Vec::new();
    let mut records = Vec::new();
    let mut graph_index = 0usize;
    if let Some(graph) = model.graph.as_mut() {
        materialize_graph(
            graph,
            "main",
            &source_spec,
            &source_fingerprint,
            &data_name,
            &mut graph_index,
            &mut sidecar,
            &mut records,
        )?;
    }

    let model_bytes = model.encode_to_vec();
    write_atomic(&data_path, &sidecar)?;
    write_atomic(&target, &model_bytes)?;
    let metadata = GeneratedMetadata {
        format: GENERATOR_VERSION,
        source: file.to_string(),
        revision,
        source_fingerprint,
        model_length: model_bytes.len() as u64,
        data_length: sidecar.len() as u64,
        tensors: records,
    };
    let metadata_bytes = serde_json::to_vec_pretty(&metadata).map_err(|e| e.to_string())?;
    write_atomic(&metadata_path, &metadata_bytes)?;
    Ok(target)
}

fn complete(model: &Path, data: &Path, metadata: &Path, requested_revision: Option<&str>) -> bool {
    let Ok(bytes) = fs::read(metadata) else {
        return false;
    };
    let Ok(meta) = serde_json::from_slice::<GeneratedMetadata>(&bytes) else {
        return false;
    };
    meta.format == GENERATOR_VERSION
        && requested_revision.is_none_or(|revision| meta.revision == revision)
        && fs::metadata(model)
            .map(|m| m.len() == meta.model_length)
            .unwrap_or(false)
        && fs::metadata(data)
            .map(|m| m.len() == meta.data_length)
            .unwrap_or(false)
}

#[allow(clippy::too_many_arguments)]
fn materialize_graph(
    graph: &mut GraphProto,
    graph_path: &str,
    source: &dyn ExactSource,
    fingerprint: &str,
    data_name: &str,
    graph_index: &mut usize,
    sidecar: &mut Vec<u8>,
    records: &mut Vec<TensorRecord>,
) -> Result<(), String> {
    let roles = classify_graph(graph, graph_path)?;
    for tensor in &mut graph.initializer {
        materialize_tensor(
            tensor,
            graph_path,
            &roles,
            source,
            fingerprint,
            data_name,
            sidecar,
            records,
        )?;
    }

    for (node_index, node) in graph.node.iter_mut().enumerate() {
        for (attr_index, attr) in node.attribute.iter_mut().enumerate() {
            if let Some(tensor) = attr.t.as_ref() {
                let length = tensor_byte_len(tensor)?;
                if length > KEEP_BYTES {
                    return Err(format!(
                        "ambiguous large Constant tensor at {graph_path}/node#{node_index}/attribute#{attr_index}; tensor-valued attributes are not generated"
                    ));
                }
                records.push(TensorRecord {
                    graph: graph_path.to_string(),
                    name: format!("{}:{}", node.name, attr.name),
                    role: TensorRole::Exact,
                    reason: "small tensor-valued node attribute".to_string(),
                    bytes: length,
                    digest: hex(&Sha256::digest(tensor.encode_to_vec())),
                });
            }
            if let Some(subgraph) = attr.g.as_mut() {
                *graph_index += 1;
                let path = format!("{graph_path}/{}:{}", node.name, graph_index);
                materialize_graph(
                    subgraph,
                    &path,
                    source,
                    fingerprint,
                    data_name,
                    graph_index,
                    sidecar,
                    records,
                )?;
            }
            for (sub_index, subgraph) in attr.graphs.iter_mut().enumerate() {
                *graph_index += 1;
                let path = format!("{graph_path}/{}:{}:{sub_index}", node.name, graph_index);
                materialize_graph(
                    subgraph,
                    &path,
                    source,
                    fingerprint,
                    data_name,
                    graph_index,
                    sidecar,
                    records,
                )?;
            }
        }
    }
    Ok(())
}

fn classify_graph(
    graph: &GraphProto,
    graph_path: &str,
) -> Result<HashMap<String, (TensorRole, String)>, String> {
    let producers: HashMap<&str, &NodeProto> = graph
        .node
        .iter()
        .flat_map(|node| {
            node.output
                .iter()
                .map(move |output| (output.as_str(), node))
        })
        .collect();
    let mut consumers: HashMap<&str, Vec<(&NodeProto, usize)>> = HashMap::new();
    let mut exact_values = BTreeSet::new();
    let mut quant_scales = BTreeSet::new();
    let mut quant_zero_points = BTreeSet::new();

    for node in &graph.node {
        for (index, input) in node
            .input
            .iter()
            .enumerate()
            .filter(|(_, input)| !input.is_empty())
        {
            consumers.entry(input).or_default().push((node, index));
            match special_role(&node.op_type, index) {
                Some(TensorRole::Exact) => {
                    exact_values.insert(input.clone());
                }
                Some(TensorRole::QuantScale) => {
                    quant_scales.insert(input.clone());
                }
                Some(TensorRole::QuantZeroPoint) => {
                    quant_zero_points.insert(input.clone());
                }
                _ => {}
            }
        }
    }

    let mut queue: Vec<String> = exact_values.iter().cloned().collect();
    while let Some(value) = queue.pop() {
        let Some(producer) = producers.get(value.as_str()) else {
            continue;
        };
        if producer.op_type == "Shape" {
            continue;
        }
        if !is_constant_control_transform(&producer.op_type) {
            continue;
        }
        for input in producer.input.iter().filter(|input| !input.is_empty()) {
            if exact_values.insert(input.clone()) {
                queue.push(input.clone());
            }
        }
    }

    let registry = OpRegistry::new();
    let mut result = HashMap::new();
    for tensor in &graph.initializer {
        let len = tensor_byte_len(tensor)?;
        let (role, reason) = if len <= KEEP_BYTES {
            (
                TensorRole::Exact,
                format!("small initializer ({len} <= {KEEP_BYTES} bytes)"),
            )
        } else if exact_values.contains(&tensor.name) {
            (TensorRole::Exact, "shape/control dependency".to_string())
        } else if quant_scales.contains(&tensor.name) {
            (TensorRole::QuantScale, "quantization scale".to_string())
        } else if quant_zero_points.contains(&tensor.name) {
            (
                TensorRole::QuantZeroPoint,
                "quantization zero point".to_string(),
            )
        } else {
            let uses = consumers
                .get(tensor.name.as_str())
                .cloned()
                .unwrap_or_default();
            if uses.is_empty() {
                return Err(format!(
                    "ambiguous initializer {graph_path}/{}: no consumers",
                    tensor.name
                ));
            }
            for (node, index) in &uses {
                if !registry.is_supported(&node.op_type) {
                    return Err(format!(
                        "ambiguous initializer {graph_path}/{}: unsupported consumer {}::{} input {index}",
                        tensor.name, node.domain, node.op_type
                    ));
                }
            }
            if !is_generatable_dtype(tensor.data_type) {
                return Err(format!(
                    "ambiguous initializer {graph_path}/{}: dtype {} is not safely generatable",
                    tensor.name, tensor.data_type
                ));
            }
            (
                TensorRole::Learned,
                "supported numeric data/parameter input".to_string(),
            )
        };
        result.insert(tensor.name.clone(), (role, reason));
    }
    Ok(result)
}

fn special_role(op: &str, index: usize) -> Option<TensorRole> {
    if index == 1
        && (op.starts_with("Reduce")
            || matches!(
                op,
                "CumProd"
                    | "Trilu"
                    | "Upsample"
                    | "CenterCropPad"
                    | "Compress"
                    | "ConstantOfShape"
                    | "AffineGrid"
                    | "DFT"
                    | "STFT"
                    | "HammingWindow"
                    | "HannWindow"
                    | "BlackmanWindow"
            ))
    {
        return Some(TensorRole::Exact);
    }
    if matches!(op, "LSTM" | "GRU" | "RNN") && index == 4 {
        return Some(TensorRole::Exact);
    }
    if op == "MelWeightMatrix" {
        return Some(TensorRole::Exact);
    }
    let exact = match op {
        "Reshape" | "Expand" | "Tile" | "Squeeze" | "Unsqueeze" | "Split" | "TopK" | "CumSum" => {
            index == 1
        }
        "Slice" => (1..=4).contains(&index),
        "Pad" => index == 1 || index == 3,
        "Resize" => (1..=3).contains(&index),
        "Gather"
        | "GatherBlockQuantized"
        | "GatherElements"
        | "GatherND"
        | "Scatter"
        | "ScatterElements"
        | "ScatterND" => index == 1,
        "OneHot" => index == 1 || index == 2,
        "Range" => index <= 2,
        "NonMaxSuppression" => (2..=4).contains(&index),
        "ReverseSequence" => index == 1,
        "If" => index == 0,
        "Loop" => index <= 1,
        _ => false,
    };
    if exact {
        return Some(TensorRole::Exact);
    }
    match op {
        "QuantizeLinear" | "DequantizeLinear" if index == 1 => Some(TensorRole::QuantScale),
        "QuantizeLinear" | "DequantizeLinear" if index == 2 => Some(TensorRole::QuantZeroPoint),
        "MatMulNBits" if index == 2 => Some(TensorRole::QuantScale),
        "MatMulNBits" if index == 3 => Some(TensorRole::QuantZeroPoint),
        "MatMulBnb4" if index == 2 => Some(TensorRole::QuantScale),
        "GatherBlockQuantized" if index == 2 => Some(TensorRole::QuantScale),
        "GatherBlockQuantized" if index == 3 => Some(TensorRole::QuantZeroPoint),
        "MatMulInteger" | "ConvInteger" if index == 2 || index == 3 => {
            Some(TensorRole::QuantZeroPoint)
        }
        "QLinearMatMul" if matches!(index, 1 | 4 | 6) => Some(TensorRole::QuantScale),
        "QLinearMatMul" if matches!(index, 2 | 5 | 7) => Some(TensorRole::QuantZeroPoint),
        "QLinearConv" if matches!(index, 1 | 4 | 6) => Some(TensorRole::QuantScale),
        "QLinearConv" if matches!(index, 2 | 5 | 7) => Some(TensorRole::QuantZeroPoint),
        _ => None,
    }
}

fn is_constant_control_transform(op: &str) -> bool {
    matches!(
        op,
        "Cast"
            | "Concat"
            | "Gather"
            | "GatherElements"
            | "GatherND"
            | "Slice"
            | "Squeeze"
            | "Unsqueeze"
            | "Reshape"
            | "Add"
            | "Sub"
            | "Mul"
            | "Div"
            | "Min"
            | "Max"
            | "Where"
            | "Range"
            | "ConstantOfShape"
    )
}

fn is_generatable_dtype(dtype: i32) -> bool {
    matches!(
        dtype,
        1 | 2 | 3 | 4 | 5 | 6 | 7 | 9 | 10 | 11 | 12 | 13 | 16 | 21 | 22
    )
}

#[allow(clippy::too_many_arguments)]
fn materialize_tensor(
    tensor: &mut TensorProto,
    graph_path: &str,
    roles: &HashMap<String, (TensorRole, String)>,
    source: &dyn ExactSource,
    fingerprint: &str,
    data_name: &str,
    sidecar: &mut Vec<u8>,
    records: &mut Vec<TensorRecord>,
) -> Result<(), String> {
    if tensor.data_location != 1 {
        let (role, reason) = roles.get(&tensor.name).cloned().ok_or_else(|| {
            format!(
                "ambiguous initializer {graph_path}/{}: missing classification",
                tensor.name
            )
        })?;
        let encoded = tensor.encode_to_vec();
        records.push(TensorRecord {
            graph: graph_path.to_string(),
            name: tensor.name.clone(),
            role,
            reason,
            bytes: tensor_byte_len(tensor)?,
            digest: hex(&Sha256::digest(encoded)),
        });
        return Ok(());
    }
    let location = source_location(tensor)?;
    let (role, reason) = roles.get(&tensor.name).cloned().ok_or_else(|| {
        format!(
            "ambiguous initializer {graph_path}/{}: missing classification",
            tensor.name
        )
    })?;
    let bytes = match role {
        TensorRole::Exact => fetch_exact(source, &location)?,
        TensorRole::Learned | TensorRole::QuantScale | TensorRole::QuantZeroPoint => {
            generate_tensor(tensor, role, graph_path, fingerprint, location.length)?
        }
    };
    if bytes.len() != location.length {
        return Err(format!(
            "{}: expected {} bytes, got {}",
            tensor.name,
            location.length,
            bytes.len()
        ));
    }
    let digest = hex(&Sha256::digest(&bytes));
    if bytes.len() <= KEEP_BYTES && role == TensorRole::Exact {
        tensor.raw_data = bytes;
        tensor.external_data.clear();
        tensor.data_location = 0;
    } else {
        let offset = sidecar.len();
        sidecar.extend_from_slice(&bytes);
        tensor.raw_data.clear();
        tensor.external_data = vec![
            external("location", data_name),
            external("offset", &offset.to_string()),
            external("length", &bytes.len().to_string()),
        ];
        tensor.data_location = 1;
    }
    records.push(TensorRecord {
        graph: graph_path.to_string(),
        name: tensor.name.clone(),
        role,
        reason,
        bytes: location.length,
        digest,
    });
    Ok(())
}

fn source_location(tensor: &TensorProto) -> Result<SourceLocation, String> {
    let values: HashMap<&str, &str> = tensor
        .external_data
        .iter()
        .map(|e| (e.key.as_str(), e.value.as_str()))
        .collect();
    let length = values
        .get("length")
        .and_then(|v| v.parse().ok())
        .unwrap_or(tensor_byte_len(tensor)?);
    Ok(SourceLocation {
        location: values
            .get("location")
            .copied()
            .ok_or_else(|| format!("external tensor {} has no location", tensor.name))?
            .to_string(),
        offset: values
            .get("offset")
            .and_then(|v| v.parse().ok())
            .unwrap_or(0),
        length,
        inline_source_offset: values.get("o2w_source_offset").and_then(|v| v.parse().ok()),
        inline_encoding: values.get("o2w_source_encoding").map(|v| (*v).to_string()),
    })
}

fn fetch_exact(source: &dyn ExactSource, location: &SourceLocation) -> Result<Vec<u8>, String> {
    if location.location == "skeleton.bin" {
        if location.inline_encoding.as_deref() != Some("raw_data") {
            return Err("cannot preserve a stripped non-raw inline tensor without downloading its protobuf payload".to_string());
        }
        let offset = location
            .inline_source_offset
            .ok_or_else(|| "stripped inline tensor has no source offset".to_string())?;
        return source.read_main(offset, location.length);
    }
    let parent = Path::new(source.repository_path())
        .parent()
        .unwrap_or_else(|| Path::new(""));
    let relative = safe_relative(parent.join(&location.location))?;
    source.read_relative(&relative, location.offset, location.length)
}

fn generate_tensor(
    tensor: &TensorProto,
    role: TensorRole,
    graph_path: &str,
    fingerprint: &str,
    length: usize,
) -> Result<Vec<u8>, String> {
    let mut seed = Sha256::new();
    seed.update(format!(
        "o2w-generated-v{GENERATOR_VERSION}\0{fingerprint}\0{graph_path}\0{}\0{}\0{:?}\0{role:?}",
        tensor.name, tensor.data_type, tensor.dims
    ));
    let seed = seed.finalize();
    let mut bytes = hash_stream(&seed, length);
    match role {
        TensorRole::QuantZeroPoint => bytes.fill(0),
        TensorRole::QuantScale => encode_positive_scales(&mut bytes, tensor.data_type)?,
        TensorRole::Learned => encode_bounded_values(&mut bytes, tensor.data_type)?,
        TensorRole::Exact => unreachable!(),
    }
    Ok(bytes)
}

fn hash_stream(seed: &[u8], length: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(length);
    let mut block = 0u64;
    while out.len() < length {
        let mut hash = Sha256::new();
        hash.update(seed);
        hash.update(block.to_le_bytes());
        out.extend_from_slice(&hash.finalize());
        block += 1;
    }
    out.truncate(length);
    out
}

fn unit_value(bytes: &[u8]) -> f32 {
    u16::from_le_bytes([bytes[0], bytes[1]]) as f32 / u16::MAX as f32
}

fn encode_bounded_values(bytes: &mut [u8], dtype: i32) -> Result<(), String> {
    match dtype {
        1 => {
            for chunk in bytes.chunks_exact_mut(4) {
                chunk.copy_from_slice(&((unit_value(chunk) - 0.5) * 0.1).to_le_bytes());
            }
        }
        10 => {
            for chunk in bytes.chunks_exact_mut(2) {
                chunk.copy_from_slice(
                    &f16::from_f32((unit_value(chunk) - 0.5) * 0.1)
                        .to_bits()
                        .to_le_bytes(),
                );
            }
        }
        16 => {
            for chunk in bytes.chunks_exact_mut(2) {
                chunk.copy_from_slice(
                    &bf16::from_f32((unit_value(chunk) - 0.5) * 0.1)
                        .to_bits()
                        .to_le_bytes(),
                );
            }
        }
        11 => {
            for chunk in bytes.chunks_exact_mut(8) {
                let v = (unit_value(chunk) as f64 - 0.5) * 0.1;
                chunk.copy_from_slice(&v.to_le_bytes());
            }
        }
        2 => {
            for byte in bytes {
                *byte %= 15;
            }
        }
        3 => {
            for byte in bytes {
                *byte = ((*byte % 15) as i8 - 7) as u8;
            }
        }
        4 => {
            for chunk in bytes.chunks_exact_mut(2) {
                let value = u16::from_le_bytes(chunk.try_into().unwrap()) % 15;
                chunk.copy_from_slice(&value.to_le_bytes());
            }
        }
        5 => {
            for chunk in bytes.chunks_exact_mut(2) {
                let value = (u16::from_le_bytes(chunk.try_into().unwrap()) % 15) as i16 - 7;
                chunk.copy_from_slice(&value.to_le_bytes());
            }
        }
        6 => {
            for chunk in bytes.chunks_exact_mut(4) {
                let value = (u32::from_le_bytes(chunk.try_into().unwrap()) % 15) as i32 - 7;
                chunk.copy_from_slice(&value.to_le_bytes());
            }
        }
        7 => {
            for chunk in bytes.chunks_exact_mut(8) {
                let value = (u64::from_le_bytes(chunk.try_into().unwrap()) % 15) as i64 - 7;
                chunk.copy_from_slice(&value.to_le_bytes());
            }
        }
        12 => {
            for chunk in bytes.chunks_exact_mut(4) {
                let value = u32::from_le_bytes(chunk.try_into().unwrap()) % 15;
                chunk.copy_from_slice(&value.to_le_bytes());
            }
        }
        13 => {
            for chunk in bytes.chunks_exact_mut(8) {
                let value = u64::from_le_bytes(chunk.try_into().unwrap()) % 15;
                chunk.copy_from_slice(&value.to_le_bytes());
            }
        }
        9 => {
            for (i, byte) in bytes.iter_mut().enumerate() {
                *byte = u8::from(i % 2 == 0);
            }
        }
        21 | 22 => {
            for byte in bytes {
                *byte &= 0x77;
            }
        }
        other => return Err(format!("unsupported generated dtype {other}")),
    }
    Ok(())
}

fn encode_positive_scales(bytes: &mut [u8], dtype: i32) -> Result<(), String> {
    match dtype {
        1 => {
            for chunk in bytes.chunks_exact_mut(4) {
                chunk.copy_from_slice(&(0.01 + unit_value(chunk) * 0.09).to_le_bytes());
            }
        }
        10 => {
            for chunk in bytes.chunks_exact_mut(2) {
                chunk.copy_from_slice(
                    &f16::from_f32(0.01 + unit_value(chunk) * 0.09)
                        .to_bits()
                        .to_le_bytes(),
                );
            }
        }
        16 => {
            for chunk in bytes.chunks_exact_mut(2) {
                chunk.copy_from_slice(
                    &bf16::from_f32(0.01 + unit_value(chunk) * 0.09)
                        .to_bits()
                        .to_le_bytes(),
                );
            }
        }
        other => {
            return Err(format!(
                "quantization scale {} has unsupported dtype {other}",
                bytes.len()
            ))
        }
    }
    Ok(())
}

fn tensor_byte_len(tensor: &TensorProto) -> Result<usize, String> {
    let count = tensor
        .dims
        .iter()
        .try_fold(1usize, |acc, dim| {
            usize::try_from(*dim).ok().and_then(|d| acc.checked_mul(d))
        })
        .ok_or_else(|| format!("{} has invalid dimensions {:?}", tensor.name, tensor.dims))?;
    let length = match tensor.data_type {
        1 | 6 | 12 => count.checked_mul(4),
        2 | 3 | 9 => Some(count),
        4 | 5 | 10 | 16 => count.checked_mul(2),
        7 | 11 | 13 => count.checked_mul(8),
        21 | 22 => Some(count.div_ceil(2)),
        other => return Err(format!("{} has unsupported dtype {other}", tensor.name)),
    };
    length.ok_or_else(|| format!("{} byte size overflow", tensor.name))
}

fn external(key: &str, value: &str) -> StringStringEntryProto {
    StringStringEntryProto {
        key: key.to_string(),
        value: value.to_string(),
    }
}

fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("create {}: {e}", parent.display()))?;
    }
    let part = path.with_extension(format!(
        "{}.part",
        path.extension().and_then(|x| x.to_str()).unwrap_or("file")
    ));
    let mut file =
        fs::File::create(&part).map_err(|e| format!("create {}: {e}", part.display()))?;
    file.write_all(bytes)
        .map_err(|e| format!("write {}: {e}", part.display()))?;
    file.flush()
        .map_err(|e| format!("flush {}: {e}", part.display()))?;
    if path.exists() {
        fs::remove_file(path).map_err(|e| format!("replace {}: {e}", path.display()))?;
    }
    fs::rename(&part, path)
        .map_err(|e| format!("move {} to {}: {e}", part.display(), path.display()))
}

fn safe_relative(path: impl AsRef<Path>) -> Result<PathBuf, String> {
    let path = path.as_ref();
    if path.as_os_str().is_empty() || path.is_absolute() {
        return Err(format!("unsafe path '{}'", path.display()));
    }
    let mut clean = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Normal(part) => clean.push(part),
            _ => return Err(format!("unsafe path '{}'", path.display())),
        }
    }
    Ok(clean)
}

fn resolve_revision(file: &str) -> Result<String, String> {
    let (org_repo, _) = file
        .split_once("/")
        .ok_or_else(|| format!("{file}: expected <org>--<repo>/<path>"))?;
    let repo = org_repo.replacen("--", "/", 1);
    let url = format!("https://huggingface.co/api/models/{repo}/revision/main");
    let agent = ureq::AgentBuilder::new()
        .timeout(Duration::from_secs(120))
        .build();
    let mut request = agent.get(&url);
    if let Ok(token) = std::env::var("HF_TOKEN") {
        if !token.is_empty() {
            request = request.set("Authorization", &format!("Bearer {token}"));
        }
    }
    let response = request
        .call()
        .map_err(|e| format!("resolve revision {url}: {e}"))?;
    let value: serde_json::Value = serde_json::from_reader(response.into_reader())
        .map_err(|e| format!("decode revision {url}: {e}"))?;
    value
        .get("sha")
        .and_then(|v| v.as_str())
        .map(str::to_string)
        .ok_or_else(|| format!("{url}: response has no sha"))
}

trait ExactSource {
    fn repository_path(&self) -> &str;
    fn read_main(&self, offset: u64, length: usize) -> Result<Vec<u8>, String>;
    fn read_relative(&self, relative: &Path, offset: u64, length: usize)
        -> Result<Vec<u8>, String>;
}

struct HubSpec {
    repo: String,
    repository_path: String,
    revision: String,
    agent: ureq::Agent,
}

impl HubSpec {
    fn parse(file: &str, revision: &str) -> Result<Self, String> {
        let (org_repo, repository_path) = file
            .split_once('/')
            .ok_or_else(|| format!("{file}: expected <org>--<repo>/<path>"))?;
        let (org, repo) = org_repo
            .split_once("--")
            .ok_or_else(|| format!("{file}: expected <org>--<repo>/<path>"))?;
        Ok(Self {
            repo: format!("{org}/{repo}"),
            repository_path: repository_path.to_string(),
            revision: revision.to_string(),
            agent: ureq::AgentBuilder::new()
                .timeout(Duration::from_secs(120))
                .build(),
        })
    }
}

impl ExactSource for HubSpec {
    fn repository_path(&self) -> &str {
        &self.repository_path
    }
    fn read_main(&self, offset: u64, length: usize) -> Result<Vec<u8>, String> {
        self.read_relative(Path::new(&self.repository_path), offset, length)
    }
    fn read_relative(
        &self,
        relative: &Path,
        offset: u64,
        length: usize,
    ) -> Result<Vec<u8>, String> {
        let relative = safe_relative(relative)?
            .to_string_lossy()
            .replace('\\', "/");
        let url = format!(
            "https://huggingface.co/{}/resolve/{}/{relative}",
            self.repo, self.revision
        );
        let end = offset
            .checked_add(length as u64)
            .and_then(|v| v.checked_sub(1))
            .ok_or_else(|| "invalid empty or overflowing range".to_string())?;
        let mut last = String::new();
        for attempt in 0..5 {
            let mut request = self
                .agent
                .get(&url)
                .set("Range", &format!("bytes={offset}-{end}"));
            if let Ok(token) = std::env::var("HF_TOKEN") {
                if !token.is_empty() {
                    request = request.set("Authorization", &format!("Bearer {token}"));
                }
            }
            match request.call() {
                Ok(response) => {
                    let status = response.status();
                    let mut body = Vec::new();
                    response
                        .into_reader()
                        .read_to_end(&mut body)
                        .map_err(|e| e.to_string())?;
                    if status == 206 && body.len() == length {
                        return Ok(body);
                    }
                    if status == 200 && body.len() as u64 >= offset + length as u64 {
                        return Ok(body[offset as usize..offset as usize + length].to_vec());
                    }
                    last = format!("range {offset}-{end}: HTTP {status}, {} bytes", body.len());
                }
                Err(error) => last = error.to_string(),
            }
            std::thread::sleep(Duration::from_millis(1500 * (attempt + 1)));
        }
        Err(format!("download {url}: {last}"))
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_generation_is_stable_and_versioned() {
        let tensor = TensorProto {
            name: "w".into(),
            data_type: 1,
            dims: vec![8],
            ..Default::default()
        };
        let a = generate_tensor(&tensor, TensorRole::Learned, "main", "source", 32).unwrap();
        let b = generate_tensor(&tensor, TensorRole::Learned, "main", "source", 32).unwrap();
        assert_eq!(a, b);
        assert_ne!(
            a,
            generate_tensor(&tensor, TensorRole::Learned, "main", "other", 32).unwrap()
        );
        assert!(a
            .chunks_exact(4)
            .all(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()).is_finite()));
    }

    #[test]
    fn quant_scales_are_positive_and_zero_points_are_zero() {
        let tensor = TensorProto {
            name: "scale".into(),
            data_type: 1,
            dims: vec![8],
            ..Default::default()
        };
        let scales =
            generate_tensor(&tensor, TensorRole::QuantScale, "main", "source", 32).unwrap();
        assert!(scales
            .chunks_exact(4)
            .all(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()) > 0.0));
        let zero = generate_tensor(
            &TensorProto {
                data_type: 2,
                ..tensor
            },
            TensorRole::QuantZeroPoint,
            "main",
            "source",
            8,
        )
        .unwrap();
        assert_eq!(zero, vec![0; 8]);
    }

    #[test]
    fn traversal_marks_large_reshape_shape_exact() {
        use crate::onnx::test_models::prelude::*;
        let shape = i64_init("shape", &[600], &vec![1; 600]);
        let graph = graph(
            "control",
            vec![f32_input("x", &[1])],
            vec![f32_output("y", &[600])],
            vec![node("Reshape", "reshape", &["x", "shape"], &["y"], &[])],
            vec![shape],
        );
        let roles = classify_graph(&graph, "main").unwrap();
        assert_eq!(roles["shape"].0, TensorRole::Exact);
    }

    #[test]
    fn generated_external_weight_matches_after_webnn_round_trip() {
        use crate::onnx::test_models::prelude::*;
        use crate::{convert_onnx, validate_cached_model, ConvertOptions};

        let mut weight = TensorProto {
            name: "weight".into(),
            dims: vec![1025],
            data_type: 1,
            data_location: 1,
            external_data: vec![
                external("location", "skeleton.bin"),
                external("length", &(1025 * 4).to_string()),
                external("o2w_source_offset", "0"),
                external("o2w_source_encoding", "raw_data"),
            ],
            ..Default::default()
        };
        let mut graph = graph(
            "generated",
            vec![f32_input("input", &[1, 1025])],
            vec![f32_output("output", &[1, 1025])],
            vec![node("Add", "add", &["input", "weight"], &["output"], &[])],
            vec![weight],
        );
        let source = HubSpec {
            repo: "unused/unused".into(),
            repository_path: "model.onnx".into(),
            revision: "unused".into(),
            agent: ureq::AgentBuilder::new().build(),
        };
        let mut sidecar = Vec::new();
        let mut records = Vec::new();
        materialize_graph(
            &mut graph,
            "main",
            &source,
            "fixture",
            "model.onnx.generated.data",
            &mut 0,
            &mut sidecar,
            &mut records,
        )
        .unwrap();
        weight = graph.initializer[0].clone();
        assert_eq!(weight.external_data[0].value, "model.onnx.generated.data");
        assert_eq!(records[0].role, TensorRole::Learned);

        let temp = tempfile::tempdir().unwrap();
        let onnx = temp.path().join("model.onnx");
        let data = temp.path().join("model.onnx.generated.data");
        let webnn = temp.path().join("model.webnn");
        fs::write(&data, sidecar).unwrap();
        fs::write(&onnx, model(17, graph).encode_to_vec()).unwrap();
        convert_onnx(
            &onnx,
            ConvertOptions {
                optimize: true,
                output_path: Some(webnn.clone()),
                ..Default::default()
            },
        )
        .unwrap();
        validate_cached_model(&onnx, &webnn).unwrap();
    }

    struct FakeSource {
        repository_path: String,
        main: Vec<u8>,
        files: HashMap<PathBuf, Vec<u8>>,
    }

    impl ExactSource for FakeSource {
        fn repository_path(&self) -> &str {
            &self.repository_path
        }
        fn read_main(&self, offset: u64, length: usize) -> Result<Vec<u8>, String> {
            Ok(self.main[offset as usize..offset as usize + length].to_vec())
        }
        fn read_relative(
            &self,
            relative: &Path,
            offset: u64,
            length: usize,
        ) -> Result<Vec<u8>, String> {
            let bytes = self
                .files
                .get(relative)
                .ok_or_else(|| format!("missing {}", relative.display()))?;
            Ok(bytes[offset as usize..offset as usize + length].to_vec())
        }
    }

    #[test]
    fn small_external_control_tensor_is_range_read_and_inlined_exactly() {
        let expected = 1.25f32.to_le_bytes();
        let mut tensor = TensorProto {
            name: "scale".into(),
            dims: vec![1],
            data_type: 1,
            data_location: 1,
            external_data: vec![
                external("location", "weights/model.data"),
                external("offset", "2"),
                external("length", "4"),
            ],
            ..Default::default()
        };
        let source = FakeSource {
            repository_path: "onnx/model.onnx".into(),
            main: Vec::new(),
            files: HashMap::from([(
                PathBuf::from("onnx/weights/model.data"),
                [vec![9, 9], expected.to_vec()].concat(),
            )]),
        };
        let roles = HashMap::from([(
            "scale".to_string(),
            (TensorRole::Exact, "small initializer".to_string()),
        )]);
        let mut sidecar = Vec::new();
        let mut records = Vec::new();
        materialize_tensor(
            &mut tensor,
            "main",
            &roles,
            &source,
            "fixture",
            "generated.data",
            &mut sidecar,
            &mut records,
        )
        .unwrap();
        assert_eq!(tensor.raw_data, expected);
        assert_eq!(tensor.data_location, 0);
        assert!(tensor.external_data.is_empty());
        assert!(sidecar.is_empty());
        assert_eq!(records[0].role, TensorRole::Exact);
    }

    #[test]
    fn ambiguous_large_initializer_reports_consumer_and_input() {
        use crate::onnx::test_models::prelude::*;
        let weight = TensorProto {
            name: "mystery".into(),
            dims: vec![1025],
            data_type: 1,
            ..Default::default()
        };
        let mut unsupported = node(
            "UnknownCustom",
            "unknown",
            &["input", "mystery"],
            &["output"],
            &[],
        );
        unsupported.domain = "vendor.test".into();
        let graph = graph(
            "ambiguous",
            vec![f32_input("input", &[1025])],
            vec![f32_output("output", &[1025])],
            vec![unsupported],
            vec![weight],
        );
        let error = classify_graph(&graph, "main").unwrap_err();
        assert!(error.contains("mystery"));
        assert!(error.contains("vendor.test::UnknownCustom input 1"));
    }
}
