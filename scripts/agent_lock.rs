#!/usr/bin/env -S cargo +nightly -Zscript
---
[package]
edition = "2024"
---

use std::env;
use std::ffi::{OsStr, OsString};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

fn main() -> ExitCode {
    match run() {
        Ok(code) => code,
        Err(message) => {
            eprintln!("{message}");
            ExitCode::from(1)
        }
    }
}

fn run() -> Result<ExitCode, String> {
    let mut raw_args: Vec<OsString> = env::args_os().skip(1).collect();
    if raw_args.first().is_some_and(|value| value == "--") {
        raw_args.remove(0);
    }
    let mut args = raw_args.into_iter();
    let lock_name = args
        .next()
        .ok_or_else(|| "usage: agent_lock <lock-name> <command> [args...]".to_string())?;
    let command = args
        .next()
        .ok_or_else(|| "usage: agent_lock <lock-name> <command> [args...]".to_string())?;
    let command_args: Vec<OsString> = args.collect();

    let lock_name = lock_name
        .into_string()
        .map_err(|_| "lock name must be valid UTF-8".to_string())?;
    if lock_name.is_empty() {
        return Err("lock name must not be empty".to_string());
    }

    let lock_dir = env::var_os("GGML_RS_AGENT_LOCK_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(".agent-locks"));
    let wait_sec = read_env_u64("GGML_RS_AGENT_LOCK_WAIT_SEC", 1800)?;
    let poll_sec = read_env_u64("GGML_RS_AGENT_LOCK_POLL_SEC", 1)?;
    if poll_sec == 0 {
        return Err("GGML_RS_AGENT_LOCK_POLL_SEC must be >= 1".to_string());
    }

    fs::create_dir_all(&lock_dir)
        .map_err(|error| format!("failed to create lock dir {}: {error}", lock_dir.display()))?;
    let lock_path = lock_dir.join(format!("{lock_name}.lockdir"));

    let _guard = acquire_lock(&lock_path, wait_sec, poll_sec)?;

    let mut child = Command::new(&command)
        .args(command_args)
        .spawn()
        .map_err(|error| {
            format!(
                "failed to spawn command `{}`: {error}",
                os_to_display(&command)
            )
        })?;
    let status = child
        .wait()
        .map_err(|error| format!("failed to wait for child process: {error}"))?;

    Ok(exit_code_from_status(status.code()))
}

fn acquire_lock(lock_path: &Path, wait_sec: u64, poll_sec: u64) -> Result<LockGuard, String> {
    let start = Instant::now();
    loop {
        match fs::create_dir(lock_path) {
            Ok(()) => {
                let owner_path = lock_path.join("owner");
                let timestamp = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|duration| duration.as_secs())
                    .unwrap_or(0);
                let owner_body =
                    format!("pid={}\nstarted_unix={}\n", std::process::id(), timestamp);
                fs::write(&owner_path, owner_body).map_err(|error| {
                    format!(
                        "failed to write owner file {}: {error}",
                        owner_path.display()
                    )
                })?;
                return Ok(LockGuard {
                    lock_path: lock_path.to_path_buf(),
                });
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                if start.elapsed() >= Duration::from_secs(wait_sec) {
                    return Err(format!(
                        "timeout waiting for lock {} after {} seconds",
                        lock_path.display(),
                        wait_sec
                    ));
                }
                thread::sleep(Duration::from_secs(poll_sec));
            }
            Err(error) => {
                return Err(format!(
                    "failed to acquire lock {}: {error}",
                    lock_path.display()
                ));
            }
        }
    }
}

fn read_env_u64(key: &str, default: u64) -> Result<u64, String> {
    match env::var(key) {
        Ok(raw) => raw
            .parse::<u64>()
            .map_err(|_| format!("{key} must be an unsigned integer")),
        Err(env::VarError::NotPresent) => Ok(default),
        Err(env::VarError::NotUnicode(_)) => Err(format!("{key} must be valid UTF-8")),
    }
}

fn os_to_display(value: &OsStr) -> String {
    value.to_string_lossy().into_owned()
}

fn exit_code_from_status(code: Option<i32>) -> ExitCode {
    match code {
        Some(raw) => {
            if raw <= 0 {
                return ExitCode::from(0);
            }
            if raw > u8::MAX as i32 {
                return ExitCode::from(1);
            }
            ExitCode::from(raw as u8)
        }
        None => ExitCode::from(1),
    }
}

struct LockGuard {
    lock_path: PathBuf,
}

impl Drop for LockGuard {
    fn drop(&mut self) {
        let owner = self.lock_path.join("owner");
        let _ = fs::remove_file(owner);
        let _ = fs::remove_dir(&self.lock_path);
    }
}
