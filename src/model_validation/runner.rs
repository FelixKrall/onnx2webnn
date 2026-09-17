/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use super::full_model::cache_full_model;
use super::generated::{cache_generated_model, GENERATOR_VERSION};
use super::manifest::{load_manifest_from, Entry, Selection};
use crate::{convert_onnx, validate_cached_model_with_options, ConvertOptions};
use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

const WORKER_STACK_BYTES: usize = 256 << 20;

type ModelCell = Arc<OnceLock<Result<PathBuf, String>>>;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum WeightMode {
    #[default]
    Real,
    Generated,
}

impl FromStr for WeightMode {
    type Err = String;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "real" => Ok(Self::Real),
            "generated" => Ok(Self::Generated),
            _ => Err(format!(
                "invalid weight mode {value:?}; expected real or generated"
            )),
        }
    }
}

impl fmt::Display for WeightMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Real => "real",
            Self::Generated => "generated",
        })
    }
}

#[derive(Clone, Debug)]
pub struct RunOptions {
    pub selection: Selection,
    pub weights: WeightMode,
    pub manifest: PathBuf,
    pub jobs: usize,
    pub webnn_cache: PathBuf,
}

impl RunOptions {
    pub fn new(selection: Selection, weights: WeightMode) -> Self {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        Self {
            selection,
            weights,
            manifest: std::env::var_os("O2W_MANIFEST")
                .filter(|v| !v.is_empty())
                .map(PathBuf::from)
                .unwrap_or_else(|| root.join("tests/models/manifest.json")),
            jobs: 1,
            webnn_cache: std::env::var_os("O2W_WEBNN_CACHE")
                .filter(|v| !v.is_empty())
                .map(PathBuf::from)
                .unwrap_or_else(|| root.join(".webnn-cache")),
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RunSummary {
    pub selected: usize,
    pub passed: usize,
}

pub fn run_manifest_validation(options: RunOptions) -> Result<RunSummary, String> {
    if options.jobs == 0 {
        return Err("validation jobs must be at least 1".to_string());
    }
    let entries = load_manifest_from(&options.manifest)?;
    let selected: Vec<_> = entries
        .into_iter()
        .enumerate()
        .filter(|(index, entry)| options.selection.includes(*index, entry))
        .collect();
    if selected.is_empty() {
        return Err("validation selection matched no models".to_string());
    }
    let selected_count = selected.len();
    let (heavy, light): (Vec<_>, Vec<_>) = selected.into_iter().partition(|(_, entry)| entry.heavy);
    let sweep = Sweep {
        options,
        models: Mutex::new(HashMap::new()),
        passed: AtomicUsize::new(0),
        failures: Mutex::new(Vec::new()),
    };
    sweep.run(light, sweep.options.jobs);
    sweep.run(heavy, 1);
    let passed = sweep.passed.load(Ordering::Relaxed);
    let failures = sweep.failures.into_inner().unwrap();
    if failures.is_empty() {
        Ok(RunSummary {
            selected: selected_count,
            passed,
        })
    } else {
        Err(format!(
            "{} of {selected_count} model validations failed:\n{}",
            failures.len(),
            failures.join("\n")
        ))
    }
}

struct Sweep {
    options: RunOptions,
    models: Mutex<HashMap<String, ModelCell>>,
    passed: AtomicUsize,
    failures: Mutex<Vec<String>>,
}

impl Sweep {
    fn model(&self, entry: &Entry) -> Result<PathBuf, String> {
        let key = format!("{}:{}", self.options.weights, entry.source_key());
        let cell = self.models.lock().unwrap().entry(key).or_default().clone();
        cell.get_or_init(|| match self.options.weights {
            WeightMode::Real => cache_full_model(entry),
            WeightMode::Generated => cache_generated_model(&entry.file, entry.revision.as_deref()),
        })
        .clone()
    }

    fn validate(&self, index: usize, entry: &Entry) {
        let label = entry.label(index);
        match self.validate_inner(entry) {
            Ok((inputs, pins, outputs)) => {
                self.passed.fetch_add(1, Ordering::Relaxed);
                eprintln!(
                    "ok   {label}\n     {inputs} inputs + {pins} pinned, {outputs} outputs ({})",
                    self.options.weights
                );
            }
            Err(error) => {
                eprintln!("FAIL {label}\n     {error}");
                self.failures
                    .lock()
                    .unwrap()
                    .push(format!("{label}: {error}"));
            }
        }
    }

    fn validate_inner(&self, entry: &Entry) -> Result<(usize, usize, usize), String> {
        let onnx_path = self
            .model(entry)
            .map_err(|e| format!("model preparation: {e}"))?;
        let version = if self.options.weights == WeightMode::Generated {
            format!("-g{GENERATOR_VERSION}")
        } else {
            String::new()
        };
        let webnn_path = self.options.webnn_cache.join(format!(
            "{}-{}{version}.webnn",
            entry.cache_key(),
            self.options.weights
        ));
        std::fs::create_dir_all(webnn_path.parent().expect("cache parent"))
            .map_err(|e| format!("export: {e}"))?;
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
        .map_err(|e| format!("conversion/export: {e}"))?;
        if !webnn_path.exists() || !webnn_path.with_extension("safetensors").exists() {
            return Err("export did not produce .webnn and .safetensors".to_string());
        }
        let summary = validate_cached_model_with_options(
            &onnx_path,
            &webnn_path,
            &entry.override_dims,
            &entry.pin_inputs,
        )
        .map_err(|e| classify(&e.to_string()))?;
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

fn classify(error: &str) -> String {
    let stage = if error.contains("native ORT") {
        "native ORT"
    } else if error.contains("reload WebNN") || error.contains("graph build") {
        "reload"
    } else if error.contains("dispatch") || error.contains("write") || error.contains("read output")
    {
        "dispatch"
    } else if error.contains("mismatch") || error.contains("did not produce") {
        "comparison"
    } else {
        "validation"
    };
    format!("{stage}: {error}")
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parses_weight_modes() {
        assert_eq!("real".parse(), Ok(WeightMode::Real));
        assert_eq!("generated".parse(), Ok(WeightMode::Generated));
        assert!("random".parse::<WeightMode>().is_err());
    }
    #[test]
    fn real_is_default() {
        assert_eq!(WeightMode::default(), WeightMode::Real);
    }
}
