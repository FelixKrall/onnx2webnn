/*
 * SPDX-License-Identifier: Apache-2.0
 */

//! End-to-end cache export, reload, dispatch, and native ORT comparison.

use std::fs;

use onnx2webnn::onnx::test_models::prelude::*;
use onnx2webnn::{
    cache_onnx_model, convert_onnx, validate_cached_model, validate_cached_model_with_options,
    ConvertOptions,
};
use prost::Message;

#[test]
fn saved_add_graph_reloads_and_matches_native_ort() {
    let dir = tempfile::tempdir().expect("temporary cache");
    let source = dir.path().join("source.onnx");
    let cached_onnx = dir.path().join("cached.onnx");
    let cached_webnn = dir.path().join("cached.webnn");
    let model = model(
        17,
        graph(
            "add",
            vec![f32_input("x", &[1, 2])],
            vec![f32_output("y", &[1, 2])],
            vec![node("Add", "add", &["x", "w"], &["y"], &[])],
            vec![f32_init("w", &[1, 2], &[0.5, 1.0])],
        ),
    );
    fs::write(&source, model.encode_to_vec()).expect("write source model");
    cache_onnx_model(&source, &cached_onnx, false).expect("cache ONNX");
    convert_onnx(
        &cached_onnx,
        ConvertOptions {
            output_path: Some(cached_webnn.clone()),
            ..ConvertOptions::default()
        },
    )
    .expect("convert and export");
    assert!(cached_webnn.with_extension("safetensors").exists());
    assert_eq!(
        validate_cached_model(&cached_onnx, &cached_webnn)
            .expect("validate")
            .output_count,
        1
    );
}

#[test]
fn accepted_integer_dtypes_round_trip_exactly() {
    use onnx2webnn::protos::onnx::TensorProto_DataType;

    for (label, elem_type) in [
        ("int8", TensorProto_DataType::Int8 as i32),
        ("uint32", TensorProto_DataType::Uint32 as i32),
        ("uint64", TensorProto_DataType::Uint64 as i32),
    ] {
        let dir = tempfile::tempdir().expect("temporary cache");
        let source = dir.path().join(format!("{label}.onnx"));
        let cached_webnn = dir.path().join(format!("{label}.webnn"));
        let model = model(
            17,
            graph(
                label,
                vec![tensor_input("x", elem_type, &[4])],
                vec![tensor_output("y", elem_type, &[4])],
                vec![node("Identity", "identity", &["x"], &["y"], &[])],
                vec![],
            ),
        );
        fs::write(&source, model.encode_to_vec()).expect("write dtype model");
        convert_onnx(
            &source,
            ConvertOptions {
                output_path: Some(cached_webnn.clone()),
                ..ConvertOptions::default()
            },
        )
        .unwrap_or_else(|error| panic!("convert {label}: {error}"));
        let summary = validate_cached_model(&source, &cached_webnn)
            .unwrap_or_else(|error| panic!("validate {label}: {error}"));
        assert_eq!(summary.input_count, 1);
        assert_eq!(summary.output_count, 1);
    }
}

#[test]
fn pinned_input_is_used_by_native_ort_but_not_dispatched() {
    let dir = tempfile::tempdir().expect("temporary cache");
    let source = dir.path().join("pinned.onnx");
    let cached_webnn = dir.path().join("pinned.webnn");
    let model = model(
        17,
        graph(
            "pinned",
            vec![f32_input("x", &[2]), f32_input("scale", &[])],
            vec![f32_output("y", &[2])],
            vec![node("Mul", "mul", &["x", "scale"], &["y"], &[])],
            vec![],
        ),
    );
    fs::write(&source, model.encode_to_vec()).expect("write pinned model");
    let pins = std::collections::HashMap::from([("scale".to_string(), 2)]);
    convert_onnx(
        &source,
        ConvertOptions {
            pinned_inputs: pins.clone(),
            output_path: Some(cached_webnn.clone()),
            ..ConvertOptions::default()
        },
    )
    .expect("convert pinned model");
    let summary = validate_cached_model_with_options(
        &source,
        &cached_webnn,
        &std::collections::HashMap::new(),
        &pins,
    )
    .expect("validate pinned model");
    assert_eq!(summary.input_count, 1);
    assert_eq!(summary.pinned_input_count, 1);
    assert_eq!(summary.output_count, 1);
}

#[test]
fn pinned_input_name_reused_as_output_uses_the_converted_output_key() {
    let dir = tempfile::tempdir().expect("temporary cache");
    let source = dir.path().join("pinned-output-collision.onnx");
    let cached_webnn = dir.path().join("pinned-output-collision.webnn");
    let model = model(
        17,
        graph(
            "pinned-output-collision",
            vec![f32_input("x", &[2]), f32_input("scale", &[])],
            vec![f32_output("y", &[2]), f32_output("scale", &[])],
            vec![node("Identity", "identity", &["x"], &["y"], &[])],
            vec![],
        ),
    );
    fs::write(&source, model.encode_to_vec()).expect("write pinned-output collision model");
    let pins = std::collections::HashMap::from([("scale".to_string(), 2)]);
    convert_onnx(
        &source,
        ConvertOptions {
            pinned_inputs: pins.clone(),
            output_path: Some(cached_webnn.clone()),
            ..ConvertOptions::default()
        },
    )
    .expect("convert pinned-output collision model");

    let summary = validate_cached_model_with_options(
        &source,
        &cached_webnn,
        &std::collections::HashMap::new(),
        &pins,
    )
    .expect("validate output whose name matches a pinned input");
    assert_eq!(summary.input_count, 1);
    assert_eq!(summary.pinned_input_count, 1);
    assert_eq!(summary.output_count, 2);
}

#[test]
fn external_data_model_round_trips_without_embedding_weights() {
    use onnx2webnn::protos::onnx::StringStringEntryProto;

    let dir = tempfile::tempdir().expect("temporary cache");
    let source = dir.path().join("external.onnx");
    let weights = dir.path().join("weights.bin");
    let cached_webnn = dir.path().join("external.webnn");
    let mut model = model(
        17,
        graph(
            "external",
            vec![f32_input("x", &[2])],
            vec![f32_output("y", &[2])],
            vec![node("Add", "add", &["x", "w"], &["y"], &[])],
            vec![f32_init("w", &[2], &[0.5, 1.0])],
        ),
    );
    let weight = &mut model.graph.as_mut().unwrap().initializer[0];
    weight.float_data.clear();
    weight.data_location = 1;
    weight.external_data = vec![
        StringStringEntryProto {
            key: "location".into(),
            value: "weights.bin".into(),
        },
        StringStringEntryProto {
            key: "offset".into(),
            value: "0".into(),
        },
        StringStringEntryProto {
            key: "length".into(),
            value: "8".into(),
        },
    ];
    fs::write(&source, model.encode_to_vec()).expect("write external-data model");
    fs::write(
        &weights,
        [0.5f32, 1.0f32]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>(),
    )
    .expect("write external weights");

    convert_onnx(
        &source,
        ConvertOptions {
            output_path: Some(cached_webnn.clone()),
            ..ConvertOptions::default()
        },
    )
    .expect("convert external-data model");
    validate_cached_model(&source, &cached_webnn)
        .expect("path-based native ORT resolves external weights");
}

fn validate_semantic_index_input(input_name: &str, length: i64) {
    let dir = tempfile::tempdir().expect("temporary cache");
    let source = dir.path().join(format!("{input_name}.onnx"));
    let cached_webnn = dir.path().join(format!("{input_name}.webnn"));
    let model = model(
        17,
        graph(
            input_name,
            vec![i64_input(input_name, &[length])],
            vec![f32_output("y", &[length, 1])],
            vec![node(
                "Gather",
                "gather",
                &["table", input_name],
                &["y"],
                &[],
            )],
            vec![f32_init("table", &[2, 1], &[0.25, 0.75])],
        ),
    );
    fs::write(&source, model.encode_to_vec()).expect("write semantic-input model");
    convert_onnx(
        &source,
        ConvertOptions {
            output_path: Some(cached_webnn.clone()),
            ..ConvertOptions::default()
        },
    )
    .expect("convert semantic-input model");
    validate_cached_model(&source, &cached_webnn).expect("semantic input stays in range");
}

#[test]
fn token_type_ids_stay_within_a_two_row_embedding() {
    validate_semantic_index_input("token_type_ids", 4);
}

#[test]
fn attention_mask_is_binary_and_valid_for_indexing() {
    validate_semantic_index_input("attention_mask", 65);
}

#[test]
fn zero_element_input_round_trips_without_becoming_a_scalar() {
    use onnx2webnn::protos::onnx::{
        tensor_shape_proto::dimension::Value as DimensionValue, type_proto::Value as TypeValue,
    };

    let dir = tempfile::tempdir().expect("temporary cache");
    let source = dir.path().join("zero-element-input.onnx");
    let cached_webnn = dir.path().join("zero-element-input.webnn");
    let mut empty_cache = f32_input("past_key", &[1, 2, 1, 4]);
    let Some(TypeValue::TensorType(tensor)) =
        empty_cache.r#type.as_mut().and_then(|ty| ty.value.as_mut())
    else {
        panic!("tensor input");
    };
    tensor.shape.as_mut().unwrap().dim[2].value =
        Some(DimensionValue::DimParam("past_sequence_length".into()));
    let model = model(
        17,
        graph(
            "zero-element-input",
            vec![empty_cache, f32_input("x", &[1])],
            vec![f32_output("y", &[1])],
            vec![node("Identity", "identity", &["x"], &["y"], &[])],
            vec![],
        ),
    );
    fs::write(&source, model.encode_to_vec()).expect("write zero-element-input model");
    let overrides = std::collections::HashMap::from([("past_sequence_length".to_string(), 0)]);
    convert_onnx(
        &source,
        ConvertOptions {
            free_dim_overrides: overrides.clone(),
            output_path: Some(cached_webnn.clone()),
            ..ConvertOptions::default()
        },
    )
    .expect("convert zero-element-input model");
    let summary = validate_cached_model_with_options(
        &source,
        &cached_webnn,
        &overrides,
        &std::collections::HashMap::new(),
    )
    .expect("zero-element input survives native and cached execution");
    assert_eq!(summary.input_count, 2);
    assert_eq!(summary.output_count, 1);
}
