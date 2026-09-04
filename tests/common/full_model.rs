/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Full Hugging Face model downloads for the manual numerical sweep.

#![allow(dead_code)]

use onnx2webnn::protos::onnx::{GraphProto, ModelProto};
use prost::Message;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fs;
use std::io::Write;
use std::path::{Component, Path, PathBuf};
use std::time::Duration;

const CACHE_FORMAT: u32 = 1;

#[derive(Debug, Deserialize, Serialize)]
struct CacheMetadata {
    format: u32,
    source: String,
    files: Vec<CachedFile>,
}

#[derive(Debug, Deserialize, Serialize)]
struct CachedFile {
    path: String,
    url: String,
    etag: Option<String>,
    length: u64,
}

pub fn cache_root() -> PathBuf {
    std::env::var_os("O2W_ONNX_CACHE")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join(".onnx-cache"))
}

pub fn cache_full_model(file: &str) -> Result<PathBuf, String> {
    let (repo, repository_path) = parse_manifest_file(file)?;
    let relative_cache_path = safe_relative_path(file)?;
    let target = cache_root().join(&relative_cache_path);
    let metadata_path = metadata_path(&target);
    let refresh = std::env::var_os("O2W_MODEL_CACHE_REFRESH").is_some();
    if !refresh && complete_cache(&metadata_path) {
        return Ok(target);
    }

    if metadata_path.exists() {
        fs::remove_file(&metadata_path)
            .map_err(|e| format!("remove stale {}: {e}", metadata_path.display()))?;
    }

    let agent = ureq::AgentBuilder::new()
        .timeout(Duration::from_secs(300))
        .build();
    let source_url = hub_url(&repo, &repository_path);
    let mut files = vec![download(
        &agent,
        &source_url,
        &target,
        &relative_cache_path,
    )?];

    let model_bytes = fs::read(&target).map_err(|e| format!("read {}: {e}", target.display()))?;
    let model = ModelProto::decode(model_bytes.as_slice())
        .map_err(|e| format!("decode {}: {e}", target.display()))?;
    let locations = external_locations(&model)?;
    let repository_parent = Path::new(&repository_path)
        .parent()
        .unwrap_or_else(|| Path::new(""));
    let cache_parent = relative_cache_path
        .parent()
        .unwrap_or_else(|| Path::new(""));
    for location in locations {
        let repository_sidecar = repository_parent.join(&location);
        let relative_sidecar = cache_parent.join(&location);
        let sidecar_url = hub_url(&repo, &path_for_url(&repository_sidecar)?);
        let sidecar_target = cache_root().join(&relative_sidecar);
        files.push(download(
            &agent,
            &sidecar_url,
            &sidecar_target,
            &relative_sidecar,
        )?);
    }

    let metadata = CacheMetadata {
        format: CACHE_FORMAT,
        source: source_url,
        files,
    };
    write_metadata(&metadata_path, &metadata)?;
    Ok(target)
}

fn parse_manifest_file(file: &str) -> Result<(String, String), String> {
    let (org_repo, relative) = file
        .split_once('/')
        .ok_or_else(|| format!("{file}: expected <org>--<repo>/<path>"))?;
    let (org, repo) = org_repo
        .split_once("--")
        .ok_or_else(|| format!("{file}: expected <org>--<repo>/<path>"))?;
    if org.is_empty() || repo.is_empty() {
        return Err(format!(
            "{file}: empty Hugging Face organization or repository"
        ));
    }
    let relative = path_for_url(&safe_relative_path(relative)?)?;
    Ok((format!("{org}/{repo}"), relative))
}

fn safe_relative_path(path: impl AsRef<Path>) -> Result<PathBuf, String> {
    let path = path.as_ref();
    if path.as_os_str().is_empty() || path.is_absolute() {
        return Err(format!("unsafe external-data path '{}'", path.display()));
    }
    let mut clean = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Normal(part) => clean.push(part),
            _ => return Err(format!("unsafe external-data path '{}'", path.display())),
        }
    }
    Ok(clean)
}

fn path_for_url(path: &Path) -> Result<String, String> {
    let safe = safe_relative_path(path)?;
    Ok(safe
        .components()
        .map(|component| component.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("/"))
}

fn hub_url(repo: &str, relative: &str) -> String {
    format!("https://huggingface.co/{repo}/resolve/main/{relative}")
}

fn metadata_path(model: &Path) -> PathBuf {
    let file_name = model
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("model.onnx");
    model.with_file_name(format!("{file_name}.complete.json"))
}

fn complete_cache(metadata_path: &Path) -> bool {
    let Ok(bytes) = fs::read(metadata_path) else {
        return false;
    };
    let Ok(metadata) = serde_json::from_slice::<CacheMetadata>(&bytes) else {
        return false;
    };
    metadata.format == CACHE_FORMAT
        && metadata.files.iter().all(|entry| {
            let Ok(relative) = safe_relative_path(&entry.path) else {
                return false;
            };
            fs::metadata(cache_root().join(relative))
                .map(|metadata| metadata.is_file() && metadata.len() == entry.length)
                .unwrap_or(false)
        })
}

fn download(
    agent: &ureq::Agent,
    url: &str,
    target: &Path,
    relative_path: &Path,
) -> Result<CachedFile, String> {
    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("create {}: {e}", parent.display()))?;
    }
    let file_name = target
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("download");
    let part = target.with_file_name(format!("{file_name}.part"));
    let mut last_error = String::new();
    for attempt in 0..5 {
        let mut request = agent.get(url);
        if let Ok(token) = std::env::var("HF_TOKEN") {
            if !token.is_empty() {
                request = request.set("Authorization", &format!("Bearer {token}"));
            }
        }
        let result = (|| {
            let response = request.call().map_err(|e| e.to_string())?;
            let etag = response.header("ETag").map(str::to_string);
            let mut reader = response.into_reader();
            let mut output =
                fs::File::create(&part).map_err(|e| format!("create {}: {e}", part.display()))?;
            let length = std::io::copy(&mut reader, &mut output)
                .map_err(|e| format!("download {url}: {e}"))?;
            output
                .flush()
                .map_err(|e| format!("flush {}: {e}", part.display()))?;
            if target.exists() {
                fs::remove_file(target)
                    .map_err(|e| format!("replace {}: {e}", target.display()))?;
            }
            fs::rename(&part, target)
                .map_err(|e| format!("move {} to {}: {e}", part.display(), target.display()))?;
            Ok(CachedFile {
                path: path_for_url(relative_path)?,
                url: url.to_string(),
                etag,
                length,
            })
        })();
        match result {
            Ok(file) => return Ok(file),
            Err(error) => {
                last_error = error;
                std::thread::sleep(Duration::from_millis(1500 * (attempt + 1)));
            }
        }
    }
    Err(format!("download {url}: {last_error}"))
}

fn write_metadata(path: &Path, metadata: &CacheMetadata) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("create {}: {e}", parent.display()))?;
    }
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("complete.json");
    let part = path.with_file_name(format!("{file_name}.part"));
    let bytes = serde_json::to_vec_pretty(metadata).map_err(|e| e.to_string())?;
    fs::write(&part, bytes).map_err(|e| format!("write {}: {e}", part.display()))?;
    if path.exists() {
        fs::remove_file(path).map_err(|e| format!("replace {}: {e}", path.display()))?;
    }
    fs::rename(&part, path)
        .map_err(|e| format!("move {} to {}: {e}", part.display(), path.display()))
}

fn external_locations(model: &ModelProto) -> Result<Vec<PathBuf>, String> {
    fn walk(graph: &GraphProto, locations: &mut BTreeSet<PathBuf>) -> Result<(), String> {
        for tensor in &graph.initializer {
            if tensor.data_location != 1 {
                continue;
            }
            let location = tensor
                .external_data
                .iter()
                .find(|entry| entry.key == "location")
                .map(|entry| entry.value.as_str())
                .ok_or_else(|| format!("external tensor '{}' has no location", tensor.name))?;
            locations.insert(safe_relative_path(location)?);
        }
        for node in &graph.node {
            for attribute in &node.attribute {
                if let Some(graph) = &attribute.g {
                    walk(graph, locations)?;
                }
                for graph in &attribute.graphs {
                    walk(graph, locations)?;
                }
            }
        }
        Ok(())
    }

    let mut locations = BTreeSet::new();
    if let Some(graph) = &model.graph {
        walk(graph, &mut locations)?;
    }
    Ok(locations.into_iter().collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx2webnn::protos::onnx::{StringStringEntryProto, TensorProto};

    #[test]
    fn rejects_external_paths_that_escape_the_model_directory() {
        assert!(safe_relative_path("../weights.bin").is_err());
        assert!(safe_relative_path("/weights.bin").is_err());
        assert_eq!(
            safe_relative_path("weights/model.data").unwrap(),
            Path::new("weights/model.data")
        );
    }

    #[test]
    fn finds_unique_external_data_locations() {
        let tensor = TensorProto {
            name: "weight".into(),
            data_location: 1,
            external_data: vec![StringStringEntryProto {
                key: "location".into(),
                value: "model.onnx_data".into(),
            }],
            ..Default::default()
        };
        let model = ModelProto {
            graph: Some(GraphProto {
                initializer: vec![tensor.clone(), tensor],
                ..Default::default()
            }),
            ..Default::default()
        };
        assert_eq!(
            external_locations(&model).unwrap(),
            vec![PathBuf::from("model.onnx_data")]
        );
    }
}
