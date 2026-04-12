# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "huggingface_hub",
# ]
# ///

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import hf_hub_download


ASSETS = [
    {
        "repo_id": "Aldaris/Qwen3-8B-Q4_K_M-GGUF",
        "filename": "qwen3-8b-q4_k_m.gguf",
        "dir": "target/models/qwen3_8b_q4_k_m",
        "local_name": "Qwen3-8B-Q4_K_M.gguf",
    },
    {
        "repo_id": "LocalScribe/Qwen3.5-4B-Q4_K_M.gguf",
        "filename": "Qwen3.5-4B-Q4_K_M.gguf",
        "dir": "target/models/qwen3_5_4b_q4_k_m",
        "local_name": "Qwen3.5-4B-Q4_K_M.gguf",
    },
    {
        "repo_id": "keisuke-miyako/Llama-3-ELYZA-JP-8B-gguf-q4_k_m",
        "filename": "Llama-3-ELYZA-JP-8B-Q4_K_M.gguf",
        "dir": "target/models/elyza_llama3_jp_8b_q4_k_m",
        "local_name": "Llama-3-ELYZA-JP-8B-q4_k_m.gguf",
    },
    {
        "repo_id": "greenwich157/Llama-3.1-Minitron-4B-Width-Base-Q4_0-GGUF",
        "filename": "llama-3.1-minitron-4b-width-base-q4_0.gguf",
        "dir": "target/models/llama_minitron_4b_q4_0",
        "local_name": "Llama-3.1-Minitron-4B-Width-Base-Q4_0.gguf",
    },
    {
        "repo_id": "mradermacher/KaLM-Embedding-Gemma3-12B-2511-GGUF",
        "filename": "KaLM-Embedding-Gemma3-12B-2511.Q2_K.gguf",
        "dir": "target/models/kalm_embedding_gemma3_12b_2511_q2_k",
        "local_name": "KaLM-Embedding-Gemma3-12B-2511.Q2_K.gguf",
    },
    {
        "repo_id": "muhammedsaidckr/InternVL3-8B-Q4_K_M-GGUF",
        "filename": "internvl3-8b-q4_k_m.gguf",
        "dir": "target/models/internvl3_8b_q4_k_m",
        "local_name": "InternVL3-8B-Q4_K_M.gguf",
    },
]


def main() -> None:
    # Prefer direct blob downloads over xet-backed local-dir symlink behavior.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    for asset in ASSETS:
        local_dir = Path(asset["dir"])
        local_dir.mkdir(parents=True, exist_ok=True)
        local_path = local_dir / asset["local_name"]
        if local_path.exists() or local_path.is_symlink():
            local_path.unlink()

        print(f"DOWNLOAD {asset['repo_id']}::{asset['filename']}", flush=True)
        cache_path = Path(
            hf_hub_download(
                repo_id=asset["repo_id"],
                filename=asset["filename"],
            )
        ).resolve()
        os.symlink(cache_path, local_path)
        print(f"READY {local_path} -> {cache_path}", flush=True)

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
