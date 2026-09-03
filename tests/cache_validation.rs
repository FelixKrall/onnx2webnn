/*
 * SPDX-License-Identifier: Apache-2.0
 */

//! End-to-end cache export, reload, dispatch, and native ORT comparison.

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use onnx2webnn::onnx::test_models::prelude::*;
use onnx2webnn::{
    cache_onnx_model, convert_onnx, validate_cached_model, validate_cached_model_with_overrides,
    ConvertOptions,
};
use prost::Message;

const TINY_ROFORMER_URL: &str =
    "https://huggingface.co/Xenova/tiny-random-RoFormerForMultipleChoice/resolve/main/onnx/model.onnx";

fn cache_path(root: &str, name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(root).join(name)
}

fn download_if_missing(path: &Path) {
    if path.exists() {
        return;
    }
    fs::create_dir_all(path.parent().expect("cache parent")).expect("create ONNX cache");
    let response = ureq::get(TINY_ROFORMER_URL)
        .call()
        .unwrap_or_else(|error| panic!("download Tiny RoFormer: {error}"));
    let mut bytes = Vec::new();
    response
        .into_reader()
        .read_to_end(&mut bytes)
        .expect("read Tiny RoFormer");
    fs::write(path, bytes).expect("cache Tiny RoFormer");
}

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
fn tiny_roformer_cached_webnn_matches_native_ort() {
    let source = cache_path(".onnx-cache", "tiny-roformer-source.onnx");
    let cached_onnx = cache_path(".onnx-cache", "tiny-roformer-integration.onnx");
    let cached_webnn = cache_path(".webnn-cache", "tiny-roformer-integration.webnn");
    download_if_missing(&source);
    cache_onnx_model(&source, &cached_onnx, false).expect("cache self-contained ONNX");

    convert_onnx(
        &cached_onnx,
        ConvertOptions {
            free_dim_overrides: std::collections::HashMap::from([
                ("batch_size".to_string(), 1),
                ("num_choices".to_string(), 2),
                ("sequence_length".to_string(), 16),
            ]),
            optimize: true,
            output_path: Some(cached_webnn.clone()),
            ..ConvertOptions::default()
        },
    )
    .expect("convert and export Tiny RoFormer");

    assert!(cached_webnn.exists(), "WebNN cache was not written");
    assert!(
        cached_webnn.with_extension("safetensors").exists(),
        "Safetensors cache was not written"
    );

    let summary = validate_cached_model_with_overrides(
        &cached_onnx,
        &cached_webnn,
        &std::collections::HashMap::from([
            ("batch_size".to_string(), 1),
            ("num_choices".to_string(), 2),
            ("sequence_length".to_string(), 16),
        ]),
    )
    .expect("cached model validates");
    assert_eq!(summary.input_count, 3);
    assert_eq!(summary.output_count, 1);
}
