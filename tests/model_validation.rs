/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Manual, manifest-driven numerical validation using complete model weights.
//!
//! Set `O2W_MODEL_VALIDATION` to `smoke`, `extended`, `all`, or
//! `match=<text>`. Models are cached below `.onnx-cache`; converted WebNN and
//! Safetensors artifacts are overwritten below `.webnn-cache` on every run.

#[allow(dead_code)]
mod common;

use common::full_model::cache_full_model;
use common::manifest::{load_manifest, Entry, Selection, ValidationTier};
use onnx2webnn::{convert_onnx, validate_cached_model_with_options, ConvertOptions};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

const WORKER_STACK_BYTES: usize = 256 << 20;

fn webnn_cache_root() -> PathBuf {
    std::env::var_os("O2W_WEBNN_CACHE")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join(".webnn-cache"))
}

fn env_jobs() -> usize {
    std::env::var("O2W_MODEL_VALIDATION_JOBS")
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|&jobs| jobs > 0)
        .unwrap_or(1)
}

struct Sweep {
    models: Mutex<HashMap<String, Arc<OnceLock<Result<PathBuf, String>>>>>,
    passed: AtomicUsize,
    failures: Mutex<Vec<String>>,
}

impl Sweep {
    fn model(&self, file: &str) -> Result<PathBuf, String> {
        let cell = self
            .models
            .lock()
            .unwrap()
            .entry(file.to_string())
            .or_default()
            .clone();
        cell.get_or_init(|| cache_full_model(file)).clone()
    }

    fn validate(&self, index: usize, entry: &Entry) {
        let label = entry.label(index);
        if let Some(validation) = &entry.validation {
            if validation.tier == ValidationTier::Blocked {
                eprintln!(
                    "TRY            {label}\n               recorded blocker: {}",
                    validation.reason.as_deref().unwrap_or("missing reason")
                );
            }
        }
        let started = std::time::Instant::now();
        let result = self.validate_inner(entry);
        match result {
            Ok((inputs, pins, outputs)) => {
                self.passed.fetch_add(1, Ordering::Relaxed);
                eprintln!(
                    "ok   {:>8} ms  {label}\n               {inputs} inputs + {pins} pinned, {outputs} outputs",
                    started.elapsed().as_millis()
                );
            }
            Err(error) => {
                eprintln!(
                    "FAIL {:>8} ms  {label}\n               {error}",
                    started.elapsed().as_millis()
                );
                self.failures
                    .lock()
                    .unwrap()
                    .push(format!("{label}: {error}"));
            }
        }
    }

    fn validate_inner(&self, entry: &Entry) -> Result<(usize, usize, usize), String> {
        let onnx_path = self
            .model(&entry.file)
            .map_err(|error| format!("download: {error}"))?;
        let webnn_path = webnn_cache_root().join(format!("{}.webnn", entry.cache_key()));
        std::fs::create_dir_all(webnn_path.parent().expect("cache parent"))
            .map_err(|error| format!("export: create cache directory: {error}"))?;

        convert_onnx(
            &onnx_path,
            ConvertOptions {
                free_dim_overrides: entry.override_dims.clone(),
                optimize: true,
                experimental_dynamic_inputs: false,
                pinned_inputs: entry.pin_inputs.clone(),
                zero_fill_missing_external_data: false,
                output_path: Some(webnn_path.clone()),
            },
        )
        .map_err(|error| format!("conversion/export: {error}"))?;
        if !webnn_path.exists() || !webnn_path.with_extension("safetensors").exists() {
            return Err("export: expected .webnn and .safetensors artifacts".to_string());
        }

        let summary = validate_cached_model_with_options(
            &onnx_path,
            &webnn_path,
            &entry.override_dims,
            &entry.pin_inputs,
        )
        .map_err(|error| classify_validation_error(&error.to_string()))?;
        Ok((
            summary.input_count,
            summary.pinned_input_count,
            summary.output_count,
        ))
    }

    fn run(&self, entries: Vec<(usize, Entry)>, workers: usize) {
        let queue = Mutex::new(entries);
        std::thread::scope(|scope| {
            for _ in 0..workers {
                std::thread::Builder::new()
                    .stack_size(WORKER_STACK_BYTES)
                    .spawn_scoped(scope, || loop {
                        let Some((index, entry)) = queue.lock().unwrap().pop() else {
                            break;
                        };
                        self.validate(index, &entry);
                    })
                    .expect("spawn validation worker");
            }
        });
    }
}

fn classify_validation_error(error: &str) -> String {
    let stage = if error.contains("native ORT") {
        "native ORT"
    } else if error.contains("reload WebNN") || error.contains("graph build") {
        "reload"
    } else if error.contains("dispatch")
        || error.contains("write")
        || error.contains("read output")
        || error.contains("tensor")
    {
        "dispatch"
    } else if error.contains("mismatch") || error.contains("did not produce") {
        "comparison"
    } else {
        "validation"
    };
    format!("{stage}: {error}")
}

#[test]
fn manifest_models_round_trip_and_match_native_ort() {
    let Some(value) = std::env::var("O2W_MODEL_VALIDATION")
        .ok()
        .filter(|value| !value.is_empty())
    else {
        eprintln!(
            "skipping full-model validation: set O2W_MODEL_VALIDATION=smoke, extended, all, or match=<text>"
        );
        return;
    };
    let selection = Selection::parse(&value).unwrap_or_else(|error| panic!("{error}"));
    let entries = load_manifest().unwrap_or_else(|error| panic!("{error}"));
    let selected: Vec<_> = entries
        .into_iter()
        .enumerate()
        .filter(|(index, entry)| selection.includes(*index, entry))
        .collect();
    assert!(
        !selected.is_empty(),
        "validation selection matched no models"
    );
    let (heavy, light): (Vec<_>, Vec<_>) = selected.into_iter().partition(|(_, entry)| entry.heavy);
    let selected_count = light.len() + heavy.len();
    let sweep = Sweep {
        models: Mutex::new(HashMap::new()),
        passed: AtomicUsize::new(0),
        failures: Mutex::new(Vec::new()),
    };
    let started = std::time::Instant::now();
    sweep.run(light, env_jobs());
    sweep.run(heavy, 1);

    let failures = sweep.failures.into_inner().unwrap();
    eprintln!(
        "full-model validation: {} passed, {} failed, {} selected; {:.1} s",
        sweep.passed.load(Ordering::Relaxed),
        failures.len(),
        selected_count,
        started.elapsed().as_secs_f32()
    );
    assert!(
        failures.is_empty(),
        "full-model validations failed:\n{}",
        failures.join("\n")
    );
}
