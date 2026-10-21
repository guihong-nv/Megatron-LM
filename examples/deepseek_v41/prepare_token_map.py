# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Build and validate Engram identities from the exact released tokenizer."""

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    """Prepare a CPU int64 token map in the Megatron training environment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True, help="Local released tokenizer directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Explicitly allow a Hugging Face identifier and network downloads",
    )
    args = parser.parse_args()

    import torch
    from transformers import AutoTokenizer

    from megatron.core.models.deepseek_v41.engram_hash import build_compressed_token_map

    reference_path = Path(__file__).with_name("config_flash.json")
    reference = json.loads(reference_path.read_text(encoding="utf-8"))["text_config"]
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=not args.allow_download, trust_remote_code=False
    )
    lookup, count = build_compressed_token_map(tokenizer)
    if len(lookup) != reference["vocab_size"] or count != reference["engram_compressed_vocab_size"]:
        raise ValueError(
            f"Tokenizer mismatch: got {len(lookup)} raw / {count} compressed identities; "
            f"expected {reference['vocab_size']} / {reference['engram_compressed_vocab_size']}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.tensor(lookup, dtype=torch.int64), args.output)
    metadata = {
        "tokenizer": args.tokenizer,
        "raw_vocab_size": len(lookup),
        "compressed_vocab_size": count,
        "config_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        "token_map_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    args.output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    # Emit machine-readable provenance on the standalone CLI's stdout.
    print(json.dumps(metadata, indent=2))  # pylint: disable=bad-builtin


if __name__ == "__main__":
    main()
