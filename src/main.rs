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

use anyhow::Context;
use clap::{Args, Parser, Subcommand};
use rustnn::mlcontext::Backend;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use onnx2webnn::model_validation::manifest::Selection;
use onnx2webnn::model_validation::{run_manifest_validation, RunOptions, WeightMode};
use onnx2webnn::{
    cache_onnx_model, convert_onnx_with_backend, export_onnx,
    validate_cached_model_with_inputs_on_backend, validate_cached_model_with_options_on_backend,
    ConvertOptions,
};

#[derive(Parser)]
#[command(name = "onnx2webnn")]
#[command(
    about = "Convert ONNX models to WebNN via MLGraphBuilder",
    long_about = None
)]
struct Cli {
    #[command(subcommand)]
    cmd: Command,
}

#[derive(Args)]
struct CommonArgs {
    /// Enable debug output
    #[arg(long, global = true)]
    debug: bool,

    #[arg(long, global = true, help = backend_help())]
    backend: Option<Backend>,
}

fn backend_help() -> String {
    let names = Backend::compiled()
        .iter()
        .map(|backend| backend.as_ref().to_owned())
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "Execution backend for graph builds; omit for export only; required for validation. Compiled backends: {names}. Runtime libraries and hardware must also be available."
    )
}

#[derive(Subcommand)]
enum Command {
    /// Print the code-backed operator catalog as JSON
    OperatorCatalog {
        #[command(flatten)]
        common: CommonArgs,
    },
    /// Lower ONNX and either build or export the WebNN graph
    #[command(after_help = r#"Examples:
  onnx2webnn convert --model first.onnx --model second.onnx --output exported
  onnx2webnn convert --model model.onnx --backend onnx

Repeat --model for multiple files; comma-separated lists are not split.
Dimension overrides and input pins specialize the exported graph.
Run validate separately to supply execution inputs without modifying the graph."#)]
    Convert {
        #[command(flatten)]
        common: CommonArgs,
        /// ONNX model path; repeat --model for each file (no comma-separated lists)
        #[arg(long = "model", required = true)]
        models: Vec<PathBuf>,
        #[command(flatten)]
        settings: ModelSettingsArgs,

        /// Output .webnn file or extensionless directory (created as needed)
        #[arg(long, short)]
        output: Option<PathBuf>,

        /// Zero-fill external tensors whose data file is missing (weight-stripped
        /// skeleton models; graph structure only)
        #[arg(long = "allow-missing-external-data")]
        allow_missing_external_data: bool,

        #[command(flatten)]
        folding: FoldingArgs,

        /// Preserve unresolved symbolic input dims as dynamic metadata (experimental;
        /// with --backend, requires a build with --features dynamic-inputs)
        #[arg(long)]
        experimental_dynamic_inputs: bool,
    },

    /// Numerically validate manifest models against reloaded WebNN artifacts
    ValidateManifest {
        #[command(flatten)]
        common: CommonArgs,
        #[command(flatten)]
        folding: FoldingArgs,
        /// Manifest selection: all or match=<text>
        #[arg(long)]
        selection: String,

        /// Weight source. Only publisher weights are currently supported.
        #[arg(long, default_value_t = WeightMode::Real)]
        weights: WeightMode,

        /// Override tests/models/manifest.json
        #[arg(long)]
        manifest: Option<PathBuf>,

        /// Number of parallel light-model workers; heavy models remain sequential
        #[arg(long, default_value_t = 1)]
        jobs: usize,
    },

    /// Validate existing WebNN graphs against ONNX models without conversion
    #[command(after_help = r#"Examples:
  onnx2webnn validate --onnx first.onnx --onnx second.onnx --backend onnx
  onnx2webnn validate --onnx first.onnx --webnn first.webnn --input first.safetensors \
    --onnx second.onnx --webnn second.webnn --input second.safetensors --backend onnx

Repeat --onnx, --input and --webnn for multiple files; comma-separated lists are not split.
Input files contain SafeTensors execution tensors, paired with models in argument order.
Supply one --input per --onnx or omit all --input options. The same rule applies to --webnn;
omitting all --webnn options selects the cached graphs.
Dimension overrides and input files size execution feeds; they never rewrite the graph.
Pins must match the values used during conversion."#)]
    Validate {
        #[command(flatten)]
        common: CommonArgs,
        /// Reference ONNX model path; repeat --onnx for each file (no comma-separated lists)
        #[arg(long, required = true)]
        onnx: Vec<PathBuf>,
        #[command(flatten)]
        settings: ModelSettingsArgs,
        /// SafeTensors execution inputs; repeat --input for each model, paired by position
        #[arg(long = "input", short = 'i')]
        inputs: Vec<PathBuf>,
        /// Existing .webnn file, paired by position with each --onnx; omit for cache lookup
        #[arg(long)]
        webnn: Vec<PathBuf>,
    },
}

#[derive(Args)]
struct FoldingArgs {
    /// Disable optional constant-folding passes (enabled by default); required simplifications remain
    #[arg(long)]
    no_constant_fold: bool,
}

#[derive(Args)]
struct ModelSettingsArgs {
    /// Symbolic dimension binding for a single model, NAME=VALUE (repeatable)
    #[arg(long = "override-dim")]
    override_dims: Vec<String>,
    /// Freeze a graph input for a single model, NAME=VALUE (repeatable)
    #[arg(long = "pin-input")]
    pin_inputs: Vec<String>,
    /// JSON array of per-model dimension overrides and input pins
    #[arg(long = "override-dims-pininput-file")]
    settings_file: Option<PathBuf>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelConfig {
    model: PathBuf,
    #[serde(default)]
    override_dims: HashMap<String, u32>,
    #[serde(default)]
    pin_inputs: HashMap<String, serde_json::Value>,
}

#[derive(Clone, Default)]
struct ModelSettings {
    overrides: HashMap<String, u32>,
    pins: HashMap<String, i64>,
}

fn parse_overrides(specs: Vec<String>) -> anyhow::Result<HashMap<String, u32>> {
    let mut values = HashMap::new();
    for spec in specs {
        let (name, value) = spec.split_once('=').ok_or_else(|| {
            anyhow::anyhow!("invalid --override-dim {spec:?}; expected NAME=VALUE")
        })?;
        if name.trim().is_empty() {
            anyhow::bail!("empty --override-dim name");
        }
        let parsed = value.trim().parse::<u32>()?;
        if values.insert(name.trim().to_string(), parsed).is_some() {
            anyhow::bail!("duplicate --override-dim {name}");
        }
    }
    Ok(values)
}

fn parse_pins(specs: Vec<String>) -> anyhow::Result<HashMap<String, i64>> {
    let mut values = HashMap::new();
    for spec in specs {
        let (name, value) = onnx2webnn::onnx::convert::parse_pinned_input(&spec)?;
        if values.insert(name.clone(), value).is_some() {
            anyhow::bail!("duplicate --pin-input {name}");
        }
    }
    Ok(values)
}

fn cache_paths(input: &Path) -> anyhow::Result<(PathBuf, PathBuf)> {
    let canonical = input.canonicalize().unwrap_or_else(|_| input.to_path_buf());
    let mut hash = 0xcbf29ce484222325u64;
    for byte in canonical.to_string_lossy().bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    let stem = input
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("model");
    let safe_stem: String = stem
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    let key = format!("{safe_stem}-{hash:016x}");
    Ok((
        onnx2webnn::cache::onnx_cache_dir()
            .map_err(anyhow::Error::msg)?
            .join(format!("{key}.onnx")),
        onnx2webnn::cache::webnn_cache_dir()
            .map_err(anyhow::Error::msg)?
            .join(format!("{key}.webnn")),
    ))
}

fn conversion_paths(input: &Path, output: Option<&Path>) -> anyhow::Result<(PathBuf, PathBuf)> {
    let (onnx, cached_webnn) = cache_paths(input)?;
    let webnn = match output {
        None => cached_webnn,
        Some(path) => match path.extension() {
            Some(ext) if ext == "webnn" => {
                anyhow::ensure!(
                    !path.is_dir(),
                    "output .webnn path is a directory: {}",
                    path.display()
                );
                path.to_path_buf()
            }
            None => {
                anyhow::ensure!(
                    !path.is_file(),
                    "output directory path is a file: {}",
                    path.display()
                );
                let stem = input
                    .file_stem()
                    .ok_or_else(|| anyhow::anyhow!("input has no file name"))?;
                let mut filename = stem.to_os_string();
                filename.push(".webnn");
                path.join(filename)
            }
            Some(_) => anyhow::bail!(
                "output must be a .webnn file or an extensionless directory: {}",
                path.display()
            ),
        },
    };
    Ok((onnx, webnn))
}

fn conversion_targets(
    models: &[PathBuf],
    output: Option<&Path>,
) -> anyhow::Result<Vec<(PathBuf, PathBuf)>> {
    if models.len() > 1 {
        if let Some(output) = output {
            anyhow::ensure!(
                output.extension().is_none(),
                "multiple models require an output directory"
            );
        }
    }
    let targets = models
        .iter()
        .map(|model| conversion_paths(model, output))
        .collect::<anyhow::Result<Vec<_>>>()?;
    let mut seen = HashSet::new();
    for (_, graph) in &targets {
        anyhow::ensure!(
            seen.insert(normalized_output(graph)?),
            "models resolve to the same output: {}",
            graph.display()
        );
    }
    Ok(targets)
}

/// Resolve existing ancestors so symlinked directories cannot hide output collisions.
fn normalized_output(path: &Path) -> anyhow::Result<PathBuf> {
    let mut resolved = PathBuf::new();
    for component in std::env::current_dir()?.join(path).components() {
        match component {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                resolved.pop();
            }
            _ => {
                resolved.push(component.as_os_str());
                if let Ok(canonical) = resolved.canonicalize() {
                    resolved = canonical;
                }
            }
        }
    }
    Ok(resolved)
}

fn check_input_count(models: &[PathBuf], inputs: &[PathBuf]) -> anyhow::Result<()> {
    anyhow::ensure!(
        inputs.is_empty() || inputs.len() == models.len(),
        "--input must be omitted or supplied once per --onnx ({} models, {} input files)",
        models.len(),
        inputs.len()
    );
    Ok(())
}

/// Match configuration entries by model path, independently of argument order.
fn model_settings(
    models: &[PathBuf],
    override_dims: Vec<String>,
    pin_inputs: Vec<String>,
    settings_file: Option<&Path>,
) -> anyhow::Result<Vec<ModelSettings>> {
    let single_options = !override_dims.is_empty() || !pin_inputs.is_empty();
    anyhow::ensure!(
        models.len() == 1 || !single_options,
        "--override-dim and --pin-input require a single model; use --override-dims-pininput-file for multiple models"
    );
    let Some(path) = settings_file else {
        let settings = ModelSettings {
            overrides: parse_overrides(override_dims)?,
            pins: parse_pins(pin_inputs)?,
        };
        return Ok(vec![settings; models.len()]);
    };
    anyhow::ensure!(
        !single_options,
        "--override-dims-pininput-file cannot be combined with --override-dim or --pin-input"
    );
    let entries: Vec<ModelConfig> = serde_json::from_slice(&std::fs::read(path)?)
        .with_context(|| format!("invalid model settings file: {}", path.display()))?;
    let model_paths = models
        .iter()
        .map(|model| normalized_output(model))
        .collect::<anyhow::Result<Vec<_>>>()?;
    let mut settings = HashMap::new();
    for entry in entries {
        let model = normalized_output(
            &path
                .parent()
                .unwrap_or_else(|| Path::new(""))
                .join(&entry.model),
        )?;
        anyhow::ensure!(
            model_paths.contains(&model),
            "settings model was not selected: {}",
            model.display()
        );
        anyhow::ensure!(
            !settings.contains_key(&model),
            "duplicate settings for model: {}",
            model.display()
        );
        anyhow::ensure!(
            entry
                .override_dims
                .keys()
                .all(|name| !name.trim().is_empty()),
            "empty dimension name for model: {}",
            model.display()
        );
        anyhow::ensure!(
            entry.pin_inputs.keys().all(|name| !name.trim().is_empty()),
            "empty input pin name for model: {}",
            model.display()
        );
        let pins = parse_pins(
            entry
                .pin_inputs
                .into_iter()
                .map(|(name, value)| format!("{name}={value}"))
                .collect(),
        )
        .with_context(|| format!("invalid input pins for model: {}", model.display()))?;
        settings.insert(
            model,
            ModelSettings {
                overrides: entry.override_dims,
                pins,
            },
        );
    }
    model_paths
        .iter()
        .map(|model| {
            settings
                .get(model)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("missing settings for model: {}", model.display()))
        })
        .collect()
}

fn validate_graph(
    model: &Path,
    graph: &Path,
    overrides: &HashMap<String, u32>,
    pins: &HashMap<String, i64>,
    inputs: Option<Vec<rustnn::OnnxInput>>,
    backend: Backend,
) -> anyhow::Result<onnx2webnn::ValidationSummary> {
    let summary = if let Some(inputs) = inputs {
        validate_cached_model_with_inputs_on_backend(model, graph, inputs, pins, backend)
    } else {
        validate_cached_model_with_options_on_backend(model, graph, overrides, pins, backend)
    }?;
    Ok(summary)
}

fn finish_batch(action: &str, total: usize, failed: usize) -> anyhow::Result<()> {
    println!("{action}: {}/{} passed", total - failed, total);
    anyhow::ensure!(failed == 0, "{failed} model(s) failed");
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let common = match &cli.cmd {
        Command::OperatorCatalog { common }
        | Command::Convert { common, .. }
        | Command::Validate { common, .. }
        | Command::ValidateManifest { common, .. } => common,
    };
    let backend = common.backend;
    if !matches!(cli.cmd, Command::OperatorCatalog { .. }) {
        std::env::set_var("O2W_OPERATOR_REPORT", "1");
    }
    if common.debug {
        std::env::set_var("RUST_LOG", "debug");
    }
    pretty_env_logger::init();
    let validation_backend = if !matches!(cli.cmd, Command::OperatorCatalog { .. }) {
        let dynamic_inputs = matches!(
            cli.cmd,
            Command::Convert {
                experimental_dynamic_inputs: true,
                ..
            }
        );
        let validation = matches!(
            cli.cmd,
            Command::Validate { .. } | Command::ValidateManifest { .. }
        );
        onnx2webnn::capabilities::preflight(backend, dynamic_inputs, validation)?
    } else {
        None
    };
    match cli.cmd {
        Command::OperatorCatalog { .. } => println!(
            "{}",
            serde_json::to_string_pretty(onnx2webnn::onnx::ops::catalog::OPERATOR_CAPABILITIES)?
        ),
        Command::Convert {
            models,
            settings:
                ModelSettingsArgs {
                    override_dims,
                    pin_inputs,
                    settings_file,
                },
            output,
            allow_missing_external_data,
            folding: FoldingArgs { no_constant_fold },
            experimental_dynamic_inputs,
            ..
        } => {
            let settings =
                model_settings(&models, override_dims, pin_inputs, settings_file.as_deref())?;
            let targets = conversion_targets(&models, output.as_deref())?;
            let mut failed = 0;
            for (index, (model, (onnx, graph))) in models.iter().zip(&targets).enumerate() {
                let ModelSettings { overrides, pins } = &settings[index];
                let run = || -> anyhow::Result<()> {
                    cache_onnx_model(model, onnx, allow_missing_external_data)?;
                    let options = ConvertOptions {
                        free_dim_overrides: overrides.clone(),
                        optimize: !no_constant_fold,
                        experimental_dynamic_inputs,
                        pinned_inputs: pins.clone(),
                        zero_fill_missing_external_data: allow_missing_external_data,
                        output_path: Some(graph.clone()),
                    };
                    if let Some(backend) = backend {
                        let built = convert_onnx_with_backend(model, options, backend)?;
                        println!(
                            "✓ {:?} graph build succeeded for {}",
                            built.context.rustnn_backend(),
                            model.display()
                        );
                    } else {
                        export_onnx(model, options)?;
                        println!(
                            "✓ portable WebNN export succeeded for {} (no backend build)",
                            model.display()
                        );
                    }
                    println!("✓ cached ONNX at {}", onnx.display());
                    println!("✓ WebNN at {}", graph.display());
                    Ok(())
                };
                match run() {
                    Ok(()) => println!("PASS {}", model.display()),
                    Err(error) => {
                        eprintln!("FAIL {}: {error:#}", model.display());
                        failed += 1;
                    }
                }
            }
            finish_batch("Conversion", models.len(), failed)?;
        }
        Command::Validate {
            onnx: models,
            settings:
                ModelSettingsArgs {
                    override_dims,
                    pin_inputs,
                    settings_file,
                },
            inputs,
            webnn: graphs,
            ..
        } => {
            let backend = validation_backend.expect("validation backend established by preflight");
            check_input_count(&models, &inputs)?;
            anyhow::ensure!(
                graphs.is_empty() || graphs.len() == models.len(),
                "--webnn must be omitted or supplied once per --onnx"
            );
            let settings =
                model_settings(&models, override_dims, pin_inputs, settings_file.as_deref())?;
            let mut failed = 0;
            for (index, model) in models.iter().enumerate() {
                let ModelSettings { overrides, pins } = &settings[index];
                let run = || -> anyhow::Result<onnx2webnn::ValidationSummary> {
                    let graph = match graphs.get(index) {
                        Some(graph) => graph.clone(),
                        None => cache_paths(model)?.1,
                    };
                    anyhow::ensure!(
                        graph.is_file(),
                        "WebNN graph not found: {}; run convert first",
                        graph.display()
                    );
                    let (effective, supplied) = if let Some(input) = inputs.get(index) {
                        let (dims, tensors) =
                            onnx2webnn::load_safetensors_inputs(model, input, overrides, pins)?;
                        (dims, Some(tensors))
                    } else {
                        (overrides.clone(), None)
                    };
                    validate_graph(model, &graph, &effective, pins, supplied, backend)
                };
                match run() {
                    Ok(summary) => println!(
                        "PASS {} ({} inputs, {} outputs)",
                        model.display(),
                        summary.input_count,
                        summary.output_count
                    ),
                    Err(error) => {
                        eprintln!("FAIL {}: {error:#}", model.display());
                        failed += 1;
                    }
                }
            }
            finish_batch("Validation", models.len(), failed)?;
        }
        Command::ValidateManifest {
            folding: FoldingArgs { no_constant_fold },
            selection,
            weights,
            manifest,
            jobs,
            ..
        } => {
            let backend = validation_backend.expect("validation backend established by preflight");
            let selection = Selection::parse(&selection).map_err(anyhow::Error::msg)?;
            let mut options = RunOptions::new(selection, weights).map_err(anyhow::Error::msg)?;
            options.jobs = jobs;
            options.backend = backend;
            options.optimize = !no_constant_fold;
            if let Some(manifest) = manifest {
                options.manifest = manifest;
            }
            println!("Validating reloaded graphs with {backend:?} backend against native CPU ORT");
            let summary = run_manifest_validation(options).map_err(anyhow::Error::msg)?;
            println!("{summary}");
            anyhow::ensure!(
                !summary.has_failures(),
                "{} of {} model validations failed",
                summary.failed.len(),
                summary.selected
            );
        }
    }
    Ok(())
}
