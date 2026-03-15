use clap::{Parser, ValueEnum};
use ggml_rs::{Context, GgufArrayValue, GgufValue, GgufWriter, Length, Shape2D, Tensor, Type};
use llama_rs::{GgufReport, inspect_gguf};
use std::error::Error as StdError;
use thiserror::Error;

const USAGE: &str = "usage: cargo run -p llama-rs --example gguf --features link-system -- <file.gguf> <w|r|r0|r1> [n|--no-check]";
const TEST_TENSOR_COUNT: usize = 10;
const TEST_TENSOR_BASE: f32 = 100.0;
const TEST_KV_COUNT: usize = 15;
const RNG_SEED: u32 = 123_456;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReadCheckPolicy {
    CheckData,
    NoCheckData,
}

impl ReadCheckPolicy {
    const fn should_check_data(self) -> bool {
        matches!(self, Self::CheckData)
    }
}

#[derive(Debug, Clone, Copy)]
struct Lcg {
    state: u32,
}

impl Lcg {
    const fn new(seed: u32) -> Self {
        Self { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(1_664_525)
            .wrapping_add(1_013_904_223);
        self.state
    }
}

fn main() -> Result<(), ExampleError> {
    let cli = Cli::parse();
    let path = cli.path.as_str();

    match cli.mode {
        Mode::W => {
            ensure_no_read_flags(&cli, "write")?;
            write_fixture(path)?;
        }
        Mode::R => {
            let policy = parse_read_policy(&cli)?;
            read_fixture_metadata(path)?;
            read_fixture_data(path, policy)?;
        }
        Mode::R0 => {
            ensure_no_read_flags(&cli, "r0")?;
            read_fixture_metadata(path)?;
        }
        Mode::R1 => {
            let policy = parse_read_policy(&cli)?;
            read_fixture_data(path, policy)?;
        }
    }

    Ok(())
}

#[derive(Debug, Error)]
enum ExampleError {
    #[error(transparent)]
    Llama(#[from] llama_rs::LlamaError),
    #[error(transparent)]
    Ggml(#[from] ggml_rs::Error),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Boxed(#[from] Box<dyn StdError>),
}

fn ensure_no_read_flags(cli: &Cli, mode_name: &str) -> Result<(), Box<dyn StdError>> {
    if cli.no_check || cli.check || cli.legacy_read_flag.is_some() {
        return Err(io_error(format!(
            "{mode_name} mode does not accept read-check flags"
        )));
    }
    Ok(())
}

fn parse_read_policy(cli: &Cli) -> Result<ReadCheckPolicy, Box<dyn StdError>> {
    if cli.no_check {
        if cli.legacy_read_flag.is_some() {
            return Err(io_error("cannot mix --no-check with legacy `n` flag"));
        }
        return Ok(ReadCheckPolicy::NoCheckData);
    }
    if cli.check {
        if cli.legacy_read_flag.is_some() {
            return Err(io_error("cannot mix --check with legacy `n` flag"));
        }
        return Ok(ReadCheckPolicy::CheckData);
    }
    if let Some(flag) = &cli.legacy_read_flag {
        return match flag.as_str() {
            "n" => Ok(ReadCheckPolicy::NoCheckData),
            "--no-check" => Ok(ReadCheckPolicy::NoCheckData),
            "--check" => Ok(ReadCheckPolicy::CheckData),
            _ => Err(io_error(
                "read mode supports no flag, `n`, `--no-check`, or `--check`",
            )),
        };
    }
    Ok(ReadCheckPolicy::CheckData)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Mode {
    #[value(name = "w")]
    W,
    #[value(name = "r")]
    R,
    #[value(name = "r0")]
    R0,
    #[value(name = "r1")]
    R1,
}

#[derive(Debug, Parser)]
#[command(
    about = "GGUF write/read fixture parity helper",
    version,
    after_help = USAGE
)]
struct Cli {
    /// Target GGUF file path.
    path: String,
    /// Operation mode: write/read/read_0/read_1.
    mode: Mode,
    /// Skip payload validation in read modes.
    #[arg(long = "no-check", conflicts_with = "check")]
    no_check: bool,
    /// Force payload validation in read modes.
    #[arg(long, conflicts_with = "no_check")]
    check: bool,
    /// Legacy positional read flag support: `n`, `--no-check`, or `--check`.
    legacy_read_flag: Option<String>,
}

fn expected_kv_entries() -> Vec<(String, GgufValue)> {
    vec![
        ("some.parameter.uint8".to_owned(), GgufValue::U8(0x12)),
        ("some.parameter.int8".to_owned(), GgufValue::I8(-0x13)),
        ("some.parameter.uint16".to_owned(), GgufValue::U16(0x1234)),
        ("some.parameter.int16".to_owned(), GgufValue::I16(-0x1235)),
        (
            "some.parameter.uint32".to_owned(),
            GgufValue::U32(0x1234_5678),
        ),
        (
            "some.parameter.int32".to_owned(),
            GgufValue::I32(-0x1234_5679),
        ),
        (
            "some.parameter.float32".to_owned(),
            GgufValue::F32(0.123_456_79),
        ),
        (
            "some.parameter.uint64".to_owned(),
            GgufValue::U64(0x1234_5678_9abc_def0),
        ),
        (
            "some.parameter.int64".to_owned(),
            GgufValue::I64(-0x1234_5678_9abc_def1),
        ),
        (
            "some.parameter.float64".to_owned(),
            GgufValue::F64(0.123_456_789_012_345_68),
        ),
        ("some.parameter.bool".to_owned(), GgufValue::Bool(true)),
        (
            "some.parameter.string".to_owned(),
            GgufValue::String("hello world".to_owned()),
        ),
        (
            "some.parameter.arr.i16".to_owned(),
            GgufValue::Array(GgufArrayValue::I16(vec![1, 2, 3, 4])),
        ),
        (
            "some.parameter.arr.f32".to_owned(),
            GgufValue::Array(GgufArrayValue::F32(vec![3.145, 2.718, 1.414])),
        ),
        (
            "some.parameter.arr.str".to_owned(),
            GgufValue::Array(GgufArrayValue::String(vec![
                "hello".to_owned(),
                "world".to_owned(),
                "!".to_owned(),
            ])),
        ),
    ]
}

fn write_fixture(path: &str) -> Result<(), Box<dyn StdError>> {
    let ctx = Context::new(128 * 1024 * 1024)?;
    let mut writer = GgufWriter::new()?;

    let kv_entries = expected_kv_entries();
    writer.set_values(kv_entries.iter().map(|(key, value)| (key.as_str(), value)))?;
    writer.set_value("some.parameter.remove_me", &GgufValue::I32(-1))?;
    let removed = writer.remove_key("some.parameter.remove_me")?;
    if removed.is_none() {
        return Err(io_error(
            "failed to remove temporary key `some.parameter.remove_me`",
        ));
    }

    let tensors = build_fixture_tensors(&ctx)?;
    for tensor in &tensors {
        writer.add_tensor(tensor);
    }
    writer.write_data_to_file(path)?;

    println!(
        "gguf_ex_write: wrote file '{}' (n_kv={}, n_tensors={})",
        path, TEST_KV_COUNT, TEST_TENSOR_COUNT
    );
    Ok(())
}

fn read_fixture_metadata(path: &str) -> Result<GgufReport, Box<dyn StdError>> {
    let report = inspect_gguf(path)?;
    println!("gguf_ex_read_0: version: {}", report.version);
    println!("gguf_ex_read_0: alignment: {}", report.alignment);
    println!("gguf_ex_read_0: data offset: {}", report.data_offset);

    println!("gguf_ex_read_0: n_kv: {}", report.kv_entries.len());
    for (index, entry) in report.kv_entries.iter().enumerate() {
        println!("gguf_ex_read_0: kv[{index}]: key = {}", entry.key);
    }
    if let Some(entry) = report
        .kv_entries
        .iter()
        .find(|entry| entry.key == "some.parameter.string")
    {
        println!(
            "gguf_ex_read_0: find key: some.parameter.string found, value = {:?}",
            entry.value
        );
    }

    println!("gguf_ex_read_0: n_tensors: {}", report.tensors.len());
    for (index, tensor) in report.tensors.iter().enumerate() {
        println!(
            "gguf_ex_read_0: tensor[{index}]: name = {}, size = {}, offset = {}, type = {} ({})",
            tensor.name, tensor.size, tensor.offset, tensor.ggml_type_name, tensor.ggml_type_raw
        );
    }

    Ok(report)
}

fn read_fixture_data(path: &str, policy: ReadCheckPolicy) -> Result<(), Box<dyn StdError>> {
    let report = inspect_gguf(path)?;
    let file_bytes = std::fs::read(path)?;

    println!("gguf_ex_read_1: n_tensors: {}", report.tensors.len());
    for (index, tensor) in report.tensors.iter().enumerate() {
        let payload = tensor_payload_slice(report.data_offset, &file_bytes, tensor)?;
        let preview = payload
            .chunks_exact(std::mem::size_of::<f32>())
            .take(10)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect::<Vec<_>>();
        println!(
            "gguf_ex_read_1: tensor[{index}]: name = {}, n_elts = {}, preview = {:?}",
            tensor.name,
            payload.len() / std::mem::size_of::<f32>(),
            preview
        );
    }

    if policy.should_check_data() {
        validate_fixture_report(&report)?;
        validate_fixture_tensor_values(&file_bytes, &report)?;
        println!("check=ok");
    }

    Ok(())
}

fn build_fixture_tensors<'ctx>(ctx: &'ctx Context) -> ggml_rs::Result<Vec<Tensor<'ctx>>> {
    let mut rng = Lcg::new(RNG_SEED);
    let mut tensors = Vec::with_capacity(TEST_TENSOR_COUNT);

    for index in 0..TEST_TENSOR_COUNT {
        let n_dims = (rng.next_u32() % 4 + 1) as usize;
        let mut ne = [1usize; 4];
        for dim in ne.iter_mut().take(n_dims) {
            *dim = (rng.next_u32() % 10 + 1) as usize;
        }

        let tensor = match n_dims {
            1 => ctx.new_f32_tensor_1d_len(Length::new(ne[0]))?,
            2 => ctx.new_f32_tensor_2d_shape(Shape2D::new(ne[0], ne[1]))?,
            3 => ctx.new_tensor_3d(Type::F32, ne[0], ne[1], ne[2])?,
            4 => ctx.new_tensor_4d(Type::F32, ne[0], ne[1], ne[2], ne[3])?,
            _ => unreachable!(),
        };
        fill_named_tensor(
            tensor,
            &format!("tensor_{index}"),
            TEST_TENSOR_BASE + index as f32,
        )?;
        tensors.push(tensor);
    }

    Ok(tensors)
}

fn fill_named_tensor(tensor: Tensor<'_>, name: &str, fill: f32) -> ggml_rs::Result<()> {
    tensor.set_name(name)?;
    let values = vec![fill; tensor.element_count()?];
    tensor.write_data(&values)
}

fn validate_fixture_report(report: &GgufReport) -> Result<(), Box<dyn StdError>> {
    let expected = expected_kv_entries();
    if report.kv_entries.len() != expected.len() {
        return Err(io_error(format!(
            "unexpected kv count: expected {} but got {}",
            expected.len(),
            report.kv_entries.len()
        )));
    }
    for (key, value) in expected {
        require_kv(report, &key, &value)?;
    }
    if report.tensors.len() != TEST_TENSOR_COUNT {
        return Err(io_error(format!(
            "unexpected tensor count: expected {} but got {}",
            TEST_TENSOR_COUNT,
            report.tensors.len()
        )));
    }
    if report
        .kv_entries
        .iter()
        .any(|entry| entry.key == "some.parameter.remove_me")
    {
        return Err(io_error(
            "temporary key `some.parameter.remove_me` should be absent",
        ));
    }

    Ok(())
}

fn validate_fixture_tensor_values(
    file_bytes: &[u8],
    report: &GgufReport,
) -> Result<(), Box<dyn StdError>> {
    for index in 0..TEST_TENSOR_COUNT {
        let tensor_name = format!("tensor_{index}");
        let expected = TEST_TENSOR_BASE + index as f32;
        let tensor = report
            .tensors
            .iter()
            .find(|tensor| tensor.name == tensor_name)
            .ok_or_else(|| io_error(format!("missing tensor `{tensor_name}`")))?;
        if tensor.ggml_type_name != "f32" {
            return Err(io_error(format!(
                "tensor `{tensor_name}` expected f32 type, got {} ({})",
                tensor.ggml_type_name, tensor.ggml_type_raw
            )));
        }

        let payload = tensor_payload_slice(report.data_offset, file_bytes, tensor)?;
        if payload.len() % std::mem::size_of::<f32>() != 0 {
            return Err(io_error(format!(
                "tensor `{tensor_name}` payload length {} is not f32-aligned",
                payload.len()
            )));
        }
        if payload.is_empty() {
            return Err(io_error(format!("tensor `{tensor_name}` has no payload")));
        }

        for (value_index, chunk) in payload.chunks_exact(4).enumerate() {
            let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            if (value - expected).abs() > 1e-6_f32 {
                return Err(io_error(format!(
                    "tensor `{tensor_name}` mismatch at index {value_index}: expected {expected}, got {value}",
                )));
            }
        }
    }
    Ok(())
}

fn tensor_payload_slice<'a>(
    data_offset: usize,
    file_bytes: &'a [u8],
    tensor: &ggml_rs::GgufTensorInfo,
) -> Result<&'a [u8], Box<dyn StdError>> {
    let payload_start = data_offset
        .checked_add(tensor.offset)
        .ok_or_else(|| io_error("tensor payload offset overflow"))?;
    let payload_end = payload_start
        .checked_add(tensor.size)
        .ok_or_else(|| io_error("tensor payload size overflow"))?;
    if payload_end > file_bytes.len() {
        return Err(io_error(format!(
            "tensor `{}` payload range is out of file bounds",
            tensor.name
        )));
    }
    Ok(&file_bytes[payload_start..payload_end])
}

fn require_kv(
    report: &GgufReport,
    key: &str,
    expected: &GgufValue,
) -> Result<(), Box<dyn StdError>> {
    let Some(entry) = report.kv_entries.iter().find(|entry| entry.key == key) else {
        return Err(io_error(format!("missing key `{key}`")));
    };

    if &entry.value != expected {
        return Err(io_error(format!(
            "kv mismatch for `{key}`: expected {:?}, got {:?}",
            expected, entry.value
        )));
    }

    Ok(())
}

fn io_error(message: impl Into<String>) -> Box<dyn StdError> {
    Box::new(std::io::Error::other(message.into()))
}
