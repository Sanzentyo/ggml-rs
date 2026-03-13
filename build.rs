use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const FFI_CONST_NAMES: &[&str] = &[
    "GGML_TYPE_F32",
    "GGML_TYPE_I32",
    "GGML_STATUS_SUCCESS",
    "GGML_BACKEND_DEVICE_TYPE_CPU",
    "GGML_BACKEND_DEVICE_TYPE_GPU",
    "GGML_BACKEND_DEVICE_TYPE_IGPU",
    "GGUF_TYPE_UINT8",
    "GGUF_TYPE_INT8",
    "GGUF_TYPE_UINT16",
    "GGUF_TYPE_INT16",
    "GGUF_TYPE_UINT32",
    "GGUF_TYPE_INT32",
    "GGUF_TYPE_FLOAT32",
    "GGUF_TYPE_BOOL",
    "GGUF_TYPE_STRING",
    "GGUF_TYPE_ARRAY",
    "GGUF_TYPE_UINT64",
    "GGUF_TYPE_INT64",
    "GGUF_TYPE_FLOAT64",
];

fn main() {
    println!("cargo:rerun-if-env-changed=GGML_RS_LIB_DIR");
    println!("cargo:rerun-if-env-changed=GGML_RS_LIB_DIRS");
    println!("cargo:rerun-if-env-changed=GGML_RS_LIBS");
    println!("cargo:rerun-if-env-changed=GGML_RS_DISABLE_BINDGEN");
    println!("cargo:rerun-if-env-changed=GGML_RS_AUTOGEN_FFI_CONSTS");
    println!("cargo:rerun-if-env-changed=GGML_RS_GGML_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=CC");
    println!("cargo:rustc-check-cfg=cfg(ggml_rs_bindgen)");
    println!("cargo:rustc-check-cfg=cfg(ggml_rs_autogen_ffi_consts)");

    maybe_generate_bindings();
    maybe_generate_ffi_constants();

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

fn maybe_generate_bindings() {
    if env::var_os("GGML_RS_DISABLE_BINDGEN").is_some() {
        println!("cargo:warning=GGML_RS_DISABLE_BINDGEN is set; using fallback handwritten FFI");
        return;
    }

    let include_dir = match resolve_ggml_include_dir() {
        Ok(path) => path,
        Err(error) => {
            println!("cargo:warning=bindgen skipped ({error}); using fallback handwritten FFI");
            return;
        }
    };

    generate_bindings(&include_dir)
        .unwrap_or_else(|error| panic!("failed to generate bindgen FFI bindings: {error}"));
    println!("cargo:rustc-cfg=ggml_rs_bindgen");
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

fn maybe_generate_ffi_constants() {
    if env::var_os("GGML_RS_AUTOGEN_FFI_CONSTS").is_none() {
        return;
    }

    generate_ffi_constants()
        .unwrap_or_else(|error| panic!("failed to auto-generate FFI constants: {error}"));
    println!("cargo:rustc-cfg=ggml_rs_autogen_ffi_consts");
}

fn generate_ffi_constants() -> Result<(), String> {
    let include_dir = resolve_ggml_include_dir()?;
    let out_dir = PathBuf::from(env::var("OUT_DIR").map_err(|error| error.to_string())?);
    let c_source = out_dir.join("ggml_const_probe.c");
    let binary = out_dir.join(format!("ggml_const_probe{}", env::consts::EXE_SUFFIX));
    let generated = out_dir.join("ffi_generated_consts.rs");

    fs::write(&c_source, build_probe_source()).map_err(|error| {
        format!(
            "writing probe source {} failed: {error}",
            c_source.display()
        )
    })?;

    compile_probe_binary(&c_source, &binary, &include_dir)?;
    let values = run_probe_binary(&binary)?;
    write_generated_constants(&generated, &values)?;

    Ok(())
}

fn resolve_ggml_include_dir() -> Result<PathBuf, String> {
    if let Some(dir) = env::var_os("GGML_RS_GGML_INCLUDE_DIR") {
        let path = PathBuf::from(dir);
        return if path.exists() {
            Ok(path)
        } else {
            Err(format!(
                "GGML_RS_GGML_INCLUDE_DIR does not exist: {}",
                path.display()
            ))
        };
    }

    let manifest_dir =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").map_err(|error| error.to_string())?);
    let candidates = [manifest_dir.join("target/vendor/ggml/include")];
    candidates
        .into_iter()
        .find(|path| path.exists())
        .ok_or_else(|| {
            "no ggml include directory found (set GGML_RS_GGML_INCLUDE_DIR or vendor ggml under target/vendor/ggml/include)".to_string()
        })
}

fn build_probe_source() -> String {
    let mut source = String::from(
        "#include <stdio.h>\n#include <ggml.h>\n#include <ggml-backend.h>\n#include <gguf.h>\n\nint main(void) {\n",
    );
    for constant in FFI_CONST_NAMES {
        source.push_str(&format!("    printf(\"{constant}=%d\\n\", {constant});\n"));
    }
    source.push_str("    return 0;\n}\n");
    source
}

fn compile_probe_binary(source: &Path, binary: &Path, include_dir: &Path) -> Result<(), String> {
    let cc = env::var("CC").unwrap_or_else(|_| "cc".to_owned());
    let output = Command::new(cc)
        .arg("-std=c11")
        .arg(format!("-I{}", include_dir.display()))
        .arg(source)
        .arg("-o")
        .arg(binary)
        .output()
        .map_err(|error| format!("failed to run C compiler: {error}"))?;
    if output.status.success() {
        Ok(())
    } else {
        Err(format!(
            "C compiler failed:\n{}",
            String::from_utf8_lossy(&output.stderr)
        ))
    }
}

fn run_probe_binary(binary: &Path) -> Result<BTreeMap<String, i32>, String> {
    let output = Command::new(binary).output().map_err(|error| {
        format!(
            "failed to execute probe binary {}: {error}",
            binary.display()
        )
    })?;
    if !output.status.success() {
        return Err(format!(
            "probe binary failed:\n{}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    let stdout = String::from_utf8(output.stdout).map_err(|error| error.to_string())?;
    let mut values = BTreeMap::new();
    for line in stdout.lines() {
        let Some((name, value)) = line.split_once('=') else {
            continue;
        };
        let parsed = value
            .parse::<i32>()
            .map_err(|error| format!("invalid integer in probe output `{line}`: {error}"))?;
        values.insert(name.to_owned(), parsed);
    }
    for constant in FFI_CONST_NAMES {
        if !values.contains_key(*constant) {
            return Err(format!("probe output missing constant `{constant}`"));
        }
    }
    Ok(values)
}

fn write_generated_constants(
    generated_path: &Path,
    values: &BTreeMap<String, i32>,
) -> Result<(), String> {
    let mut rust = String::from("// @generated by build.rs (GGML_RS_AUTOGEN_FFI_CONSTS=1)\n");
    for constant in FFI_CONST_NAMES {
        let value = values
            .get(*constant)
            .ok_or_else(|| format!("missing generated value for constant `{constant}`"))?;
        rust.push_str(&format!(
            "pub const {constant}: ::std::os::raw::c_int = {value};\n"
        ));
    }
    fs::write(generated_path, rust).map_err(|error| {
        format!(
            "writing generated constants {} failed: {error}",
            generated_path.display()
        )
    })?;
    Ok(())
}
