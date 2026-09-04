/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Shared parsing and selection for the transformers.js model manifest.

#![allow(dead_code)]

use serde::Deserialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub struct Entry {
    pub file: String,
    #[serde(default)]
    pub heavy: bool,
    #[serde(default)]
    pub override_dims: HashMap<String, u32>,
    #[serde(default)]
    pub pin_inputs: HashMap<String, i64>,
    #[serde(default)]
    pub validation: Option<ValidationConfig>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub struct ValidationConfig {
    pub tier: ValidationTier,
    #[serde(default)]
    pub reason: Option<String>,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ValidationTier {
    Smoke,
    Extended,
    Blocked,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Selection {
    Smoke,
    Extended,
    All,
    Match(String),
}

impl Selection {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "smoke" => Ok(Self::Smoke),
            "extended" => Ok(Self::Extended),
            "all" => Ok(Self::All),
            value if value.starts_with("match=") && value.len() > "match=".len() => {
                Ok(Self::Match(value["match=".len()..].to_string()))
            }
            _ => Err(format!(
                "O2W_MODEL_VALIDATION={value}: expected smoke, extended, all, or match=<text>"
            )),
        }
    }

    pub fn includes(&self, index: usize, entry: &Entry) -> bool {
        match self {
            Self::Smoke => matches!(
                entry.validation.as_ref().map(|v| v.tier),
                Some(ValidationTier::Smoke)
            ),
            Self::Extended => matches!(
                entry.validation.as_ref().map(|v| v.tier),
                Some(ValidationTier::Smoke | ValidationTier::Extended)
            ),
            Self::All => true,
            Self::Match(needle) => entry.label(index).contains(needle),
        }
    }
}

impl Entry {
    pub fn label(&self, index: usize) -> String {
        format!(
            "#{index} {} dims={:?} pins={:?}",
            self.file, self.override_dims, self.pin_inputs
        )
    }

    /// Stable cache identity. Map entries are sorted before hashing, so two
    /// cases using the same file but different overrides or pins cannot clash.
    pub fn cache_key(&self) -> String {
        let mut canonical = format!("webnn-cache-v1\nfile={}\n", self.file);
        let mut dims: Vec<_> = self.override_dims.iter().collect();
        dims.sort_unstable_by_key(|(name, _)| *name);
        for (name, value) in dims {
            canonical.push_str(&format!("dim={name}:{value}\n"));
        }
        let mut pins: Vec<_> = self.pin_inputs.iter().collect();
        pins.sort_unstable_by_key(|(name, _)| *name);
        for (name, value) in pins {
            canonical.push_str(&format!("pin={name}:{value}\n"));
        }
        let hash = canonical.bytes().fold(0xcbf29ce484222325u64, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3)
        });
        let stem = Path::new(&self.file)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("model")
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
            .collect::<String>();
        format!("{stem}-{hash:016x}")
    }
}

pub fn manifest_path() -> PathBuf {
    std::env::var_os("O2W_MANIFEST")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/models/manifest.json"))
}

pub fn load_manifest() -> Result<Vec<Entry>, String> {
    let path = manifest_path();
    let text =
        std::fs::read_to_string(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
    parse_manifest(&text).map_err(|e| format!("parse {}: {e}", path.display()))
}

pub fn parse_manifest(text: &str) -> Result<Vec<Entry>, String> {
    let entries: Vec<Entry> = serde_json::from_str(text).map_err(|e| e.to_string())?;
    for (index, entry) in entries.iter().enumerate() {
        if entry.file.is_empty() {
            return Err(format!("manifest entry #{index} has an empty file"));
        }
        if let Some(validation) = &entry.validation {
            match (validation.tier, validation.reason.as_deref()) {
                (ValidationTier::Blocked, Some(reason)) if !reason.trim().is_empty() => {}
                (ValidationTier::Blocked, _) => {
                    return Err(format!(
                        "manifest entry #{index} is validation-blocked without a reason"
                    ));
                }
                _ => {}
            }
        }
    }
    Ok(entries)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry() -> Entry {
        serde_json::from_str(
            r#"{"file":"org--repo/onnx/model.onnx","override_dims":{"b":1},"pin_inputs":{"branch":0},"validation":{"tier":"smoke"}}"#,
        )
        .unwrap()
    }

    #[test]
    fn parses_and_selects_validation_tiers() {
        let smoke = entry();
        assert!(Selection::Smoke.includes(0, &smoke));
        assert!(Selection::Extended.includes(0, &smoke));
        assert!(Selection::All.includes(0, &smoke));
        assert!(Selection::Match("model.onnx".into()).includes(0, &smoke));
        assert!(!Selection::Match("missing".into()).includes(0, &smoke));

        let mut extended = smoke.clone();
        extended.validation.as_mut().unwrap().tier = ValidationTier::Extended;
        assert!(!Selection::Smoke.includes(0, &extended));
        assert!(Selection::Extended.includes(0, &extended));
    }

    #[test]
    fn cache_key_is_stable_and_configuration_specific() {
        let first = entry();
        let reordered: Entry = serde_json::from_str(
            r#"{"pin_inputs":{"branch":0},"override_dims":{"b":1},"file":"org--repo/onnx/model.onnx"}"#,
        )
        .unwrap();
        assert_eq!(first.cache_key(), reordered.cache_key());

        let mut changed = first.clone();
        changed.override_dims.insert("b".into(), 2);
        assert_ne!(first.cache_key(), changed.cache_key());
        changed = first.clone();
        changed.pin_inputs.insert("branch".into(), 1);
        assert_ne!(first.cache_key(), changed.cache_key());
    }

    #[test]
    fn blocked_entries_require_a_reason() {
        let error =
            parse_manifest(r#"[{"file":"org--repo/model.onnx","validation":{"tier":"blocked"}}]"#)
                .unwrap_err();
        assert!(error.contains("without a reason"));

        let entries = parse_manifest(
            r#"[{"file":"org--repo/model.onnx","validation":{"tier":"blocked","reason":"unsupported op"}}]"#,
        )
        .unwrap();
        assert_eq!(
            entries[0].validation.as_ref().unwrap().tier,
            ValidationTier::Blocked
        );
    }

    #[test]
    fn selector_rejects_unknown_values() {
        assert!(Selection::parse("quick").is_err());
        assert!(Selection::parse("match=").is_err());
    }
}
