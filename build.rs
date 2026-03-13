use std::env;

fn main() {
    println!("cargo:rerun-if-env-changed=GGML_RS_LIB_DIR");
    println!("cargo:rerun-if-env-changed=GGML_RS_LIB_DIRS");
    println!("cargo:rerun-if-env-changed=GGML_RS_LIBS");

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
