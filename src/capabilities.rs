//! Build and runtime capability checks shared by CLI preflight and error reporting.

use crate::onnx::convert::{backend_context_options, OnnxError};
use rustnn::mlcontext::{Backend, MLContext};
use std::path::PathBuf;

/// Describe compile-time support without loading native runtimes.
pub fn backend_summary() -> String {
    let names = |backends: &[Backend]| {
        backends
            .iter()
            .map(Backend::as_ref)
            .collect::<Vec<_>>()
            .join(", ")
    };
    format!(
        "enabled backends: [{}], all backends: [{}]",
        names(&Backend::compiled()),
        names(Backend::compilable())
    )
}

/// Parse a known, compiled-in backend without probing its runtime.
pub fn parse_backend(name: &str) -> Result<Backend, OnnxError> {
    let backend = name.parse::<Backend>().map_err(|_| {
        OnnxError::Backend(format!(
            "could not use {name:?} because it is not a known backend; {}",
            backend_summary()
        ))
    })?;
    check_backend_build(backend)?;
    Ok(backend)
}

/// Check CLI capabilities before processing models and return the validation backend, if needed.
pub fn preflight(
    backend: Option<Backend>,
    dynamic_inputs: bool,
    validation: bool,
) -> Result<Option<Backend>, OnnxError> {
    if validation && backend.is_none() {
        return Err(OnnxError::Backend(
            "--backend is required for validation".into(),
        ));
    }
    if let Some(backend) = backend {
        check_backend_build(backend)?;
        check_dynamic_inputs(dynamic_inputs)?;
    }
    if validation && backend != Some(Backend::Onnx) {
        check_backend_build(Backend::Onnx)?;
    }
    if let Some(backend) = backend {
        check_runtime(backend)?;
    }
    if validation && backend != Some(Backend::Onnx) {
        check_runtime(Backend::Onnx)?;
    }
    Ok(if validation { backend } else { None })
}

fn check_backend_build(backend: Backend) -> Result<(), OnnxError> {
    if !Backend::compiled().contains(&backend) {
        return Err(OnnxError::Backend(format!(
            "could not use {:?} because it is compile-time disabled on this target; {}",
            backend.as_ref(),
            backend_summary()
        )));
    }
    Ok(())
}

pub(crate) fn check_dynamic_inputs(dynamic_inputs: bool) -> Result<(), OnnxError> {
    if dynamic_inputs && !rustnn::graph::dynamic_inputs_enabled() {
        return Err(OnnxError::Backend(format!(
            "{}; rebuild with `cargo build --release --features dynamic-inputs` or use a graph with all dimensions bound",
            rustnn::error::GraphError::DynamicInputsFeatureDisabled
        )));
    }
    Ok(())
}

fn check_runtime(backend: Backend) -> Result<(), OnnxError> {
    MLContext::create(&backend_context_options(backend))
        .map(|_| ())
        .map_err(|error| context_creation_error(backend.as_ref(), error))
}

/// Distinguish missing compiled support from runtime or device initialization failures.
pub(crate) fn context_creation_error(backend: &str, error: rustnn::error::Error) -> OnnxError {
    if let rustnn::error::Error::NoBackendAvailableForBackendHint { backend_hint, .. } = &error {
        let name = backend_hint.as_ref();
        if let Err(error) = check_backend_build(*backend_hint) {
            return error;
        }
        let mut detail = format!(
            "{name} backend is compiled in, but runtime initialization or device discovery failed"
        );
        match backend_hint {
            Backend::Onnx => {
                let path = std::env::var_os("ORT_DYLIB_PATH").filter(|path| !path.is_empty());
                let location = match path {
                    Some(path) => format!("ORT_DYLIB_PATH={}", PathBuf::from(path).display()),
                    None => format!(
                        "ORT_DYLIB_PATH is unset; default library is {}onnxruntime{}",
                        std::env::consts::DLL_PREFIX,
                        std::env::consts::DLL_SUFFIX
                    ),
                };
                detail.push_str(&format!(
                    ". {location}. Check the ONNX Runtime shared library and its dependencies; set ORT_DYLIB_PATH to a compatible runtime library"
                ));
            }
            Backend::Trtx => detail.push_str(
                ". Check the TensorRT-RTX shared libraries, CUDA driver, and an NVIDIA GPU with compute capability >= 8.0; RustNN does not expose which initialization step failed",
            ),
            Backend::Litert => detail.push_str(
                ". LiteRT reported no usable runtime environment; check the LiteRT runtime installation. RustNN does not expose the initialization cause",
            ),
            _ => {}
        }
        return OnnxError::Backend(detail);
    }
    OnnxError::Backend(format!("{backend} context creation failed: {error}"))
}
