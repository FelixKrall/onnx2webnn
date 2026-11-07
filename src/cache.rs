/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Cache location for exported WebNN graphs and Safetensors archives.

use std::path::PathBuf;

/// Return the WebNN export directory, honoring `O2W_WEBNN_CACHE`.
pub fn webnn_cache_dir() -> Result<PathBuf, String> {
    resolve_webnn_cache(
        std::env::var_os("O2W_WEBNN_CACHE")
            .filter(|value| !value.is_empty())
            .map(PathBuf::from),
        dirs::cache_dir(),
    )
    .ok_or_else(|| {
        "cannot determine the operating-system cache directory; set O2W_WEBNN_CACHE".to_string()
    })
}

fn resolve_webnn_cache(specific: Option<PathBuf>, os_cache: Option<PathBuf>) -> Option<PathBuf> {
    specific.or_else(|| os_cache.map(|root| root.join("onnx2webnn/webnn")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn webnn_cache_prefers_override_then_os() {
        assert_eq!(
            resolve_webnn_cache(Some("/specific".into()), Some("/os-cache".into())),
            Some(PathBuf::from("/specific"))
        );
        assert_eq!(
            resolve_webnn_cache(None, Some("/os-cache".into())),
            Some(PathBuf::from("/os-cache/onnx2webnn/webnn"))
        );
        assert_eq!(resolve_webnn_cache(None, None), None);
    }
}
