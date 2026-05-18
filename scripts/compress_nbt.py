"""
Minecraft JSON to NBT Converter

=====================================
Takes a JSON file (as produced by the schematic generator) and writes a
gzip-compressed Minecraft NBT file usable in-game.

Usage:
    python compress_nbt.py <input.json>
    python compress_nbt.py <input.json> --output custom.nbt

Requires: nbtlib (pip install nbtlib)
"""

import argparse
import json
import pathlib
import sys

import nbtlib


def infer_nbt_tag(value):
    """Convert a Python/JSON value to the appropriate nbtlib tag."""
    if isinstance(value, dict):
        return nbtlib.Compound({k: infer_nbt_tag(v) for k, v in value.items()})
    elif isinstance(value, list):
        if not value:
            return nbtlib.List[nbtlib.Int]()  # empty list default
        # If all elements are ints, use Int (common for pos/size arrays)
        if all(isinstance(v, int) for v in value):
            return nbtlib.List[nbtlib.Int](value)
        # If all floats, use Float
        if all(isinstance(v, (int, float)) for v in value):
            return nbtlib.List[nbtlib.Float]([float(v) for v in value])
        # Mixed or complex — use Compound list
        return nbtlib.List[nbtlib.Compound]([infer_nbt_tag(v) for v in value])
    elif isinstance(value, float):
        return nbtlib.Double(value)
    elif isinstance(value, int):
        return nbtlib.Int(value)
    elif isinstance(value, str):
        return nbtlib.String(value)
    elif isinstance(value, bool):
        return nbtlib.Byte(1 if value else 0)
    else:
        return nbtlib.String(str(value))


def json_to_nbt(json_path: pathlib.Path, nbt_path: pathlib.Path) -> None:
    """Read a JSON file and write a gzip-compressed NBT file."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    root_tag = infer_nbt_tag(data)
    nbt_file = nbtlib.File(root_tag)

    # Write gzip-compressed binary NBT (nbtlib handles gzip internally)
    nbt_file.save(str(nbt_path), gzipped=True)

    size_kb = nbt_path.stat().st_size / 1024
    print(f"NBT written: {nbt_path.name} ({size_kb:.1f} KB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a JSON schematic to a gzip-compressed Minecraft NBT file")
    parser.add_argument("input", help="Path to the JSON schematic file")
    parser.add_argument("--output", "-o", default=None, help="Output NBT path (default: <input>.nbt)")
    args = parser.parse_args()

    src = pathlib.Path(args.input)
    if not src.exists():
        print(f"Error: Input file not found: {src}", file=sys.stderr)
        sys.exit(1)

    out = pathlib.Path(args.output) if args.output else src.with_suffix(".nbt")

    try:
        json_to_nbt(src, out)
    except Exception as e:
        print(f"Error: Failed to convert JSON to NBT: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
