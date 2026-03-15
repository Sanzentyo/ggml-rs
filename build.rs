use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=GGML_RS_LIB_DIR");
    println!("cargo:rerun-if-env-changed=GGML_RS_LIB_DIRS");
    println!("cargo:rerun-if-env-changed=GGML_RS_LIBS");
    println!("cargo:rerun-if-env-changed=GGML_RS_GGML_INCLUDE_DIR");

    let include_dir = resolve_ggml_include_dir();
    generate_bindings(&include_dir)
        .unwrap_or_else(|error| panic!("failed to generate bindgen FFI bindings: {error}"));

    if env::var_os("CARGO_FEATURE_LINK_SYSTEM").is_none() {
        return;
    }

    if let Some(lib_dir) = env::var_os("GGML_RS_LIB_DIR") {
        println!(
            "cargo:rustc-link-search=native={}",
            lib_dir.to_string_lossy()
        );
    }

    if let Some(lib_dirs) = env::var_os("GGML_RS_LIB_DIRS") {
        for lib_dir in env::split_paths(&lib_dirs) {
            println!("cargo:rustc-link-search=native={}", lib_dir.display());
        }
    }

    let libs = env::var("GGML_RS_LIBS").unwrap_or_else(|_| "ggml-cpu,ggml-base,ggml".to_string());

    for lib in libs
        .split(',')
        .map(str::trim)
        .filter(|name| !name.is_empty())
    {
        println!("cargo:rustc-link-lib={lib}");
    }
}

fn resolve_ggml_include_dir() -> PathBuf {
    if let Some(dir) = env::var_os("GGML_RS_GGML_INCLUDE_DIR") {
        let path = PathBuf::from(dir);
        if path.exists() {
            return path;
        }
        panic!(
            "GGML_RS_GGML_INCLUDE_DIR does not exist: {}",
            path.display()
        );
    }

    let manifest_dir =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR must be set"));
    let candidates = [
        manifest_dir.join("vendor/ggml/include"),
        manifest_dir.join("target/vendor/ggml/include"),
    ];

    candidates
        .into_iter()
        .find(|path| path.exists())
        .unwrap_or_else(|| {
            panic!(
                "ggml include directory was not found. Set GGML_RS_GGML_INCLUDE_DIR, or initialize submodules with `git submodule update --init --recursive` (expected: vendor/ggml/include)"
            )
        })
}

fn generate_bindings(include_dir: &Path) -> Result<(), String> {
    let out_dir = PathBuf::from(env::var("OUT_DIR").map_err(|error| error.to_string())?);
    let wrapper_path = out_dir.join("ggml_bindgen_wrapper.h");
    let bindings_path = out_dir.join("ffi_bindings.rs");
    let allowlist_file = format!("{}/.*", include_dir.display());

    fs::write(
        &wrapper_path,
        "#include <ggml.h>\n#include <ggml-cpu.h>\n#include <ggml-backend.h>\n#include <ggml-alloc.h>\n#include <gguf.h>\n",
    )
    .map_err(|error| format!("failed to write bindgen wrapper header: {error}"))?;

    let bindings = bindgen::Builder::default()
        .header(wrapper_path.to_string_lossy())
        .clang_arg(format!("-I{}", include_dir.display()))
        .allowlist_file(allowlist_file)
        .allowlist_function("ggml_.*")
        .allowlist_function("gguf_.*")
        .allowlist_type("ggml_.*")
        .allowlist_type("gguf_.*")
        .allowlist_var("GGML_.*")
        .allowlist_var("GGUF_.*")
        .blocklist_type("max_align_t")
        .generate_comments(false)
        .layout_tests(false)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .map_err(|error| error.to_string())?;

    bindings
        .write_to_file(bindings_path)
        .map_err(|error| error.to_string())
}
