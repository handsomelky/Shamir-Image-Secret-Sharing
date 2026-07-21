import argparse
from dataclasses import dataclass
import math
import os
import secrets
import struct
import sys
import time
import zlib
from pathlib import Path

import numpy as np
from PIL import Image
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text


FIELD_SIZE = 256
MAX_SHARES = FIELD_SIZE - 1
PAYLOAD_MAGIC = b"SISSIMG1"
PAYLOAD_HEADER_FORMAT = ">8sBQQ"
PAYLOAD_HEADER_SIZE = struct.calcsize(PAYLOAD_HEADER_FORMAT)
PAYLOAD_FLAG_ZLIB = 1
SHARE_IMAGE_WIDTH = 1024
CONSOLE = Console()
ERROR_CONSOLE = Console(stderr=True)


@dataclass(frozen=True)
class FileResult:
    path: Path
    size: str


@dataclass(frozen=True)
class EncodeResult:
    input_path: Path
    payload_size: int
    original_size: int
    stored_size: int
    compressed: bool
    shares: list[FileResult]


@dataclass(frozen=True)
class DecodeResult:
    path: Path
    size: str


def gf_mul_byte(left, right):
    result = 0
    left = int(left)
    right = int(right)
    while right:
        if right & 1:
            result ^= left
        left <<= 1
        if left & FIELD_SIZE:
            left ^= 0x11B
        right >>= 1
    return result & 0xFF


def build_gf_tables():
    exp_table = np.empty(510, dtype=np.uint8)
    log_table = np.zeros(FIELD_SIZE, dtype=np.uint16)
    value = 1
    for exponent in range(MAX_SHARES):
        exp_table[exponent] = value
        log_table[value] = exponent
        value = gf_mul_byte(value, 3)
    exp_table[MAX_SHARES:] = exp_table[:MAX_SHARES]
    return exp_table, log_table


GF_EXP, GF_LOG = build_gf_tables()


def gf_inverse(value):
    if value == 0:
        raise ZeroDivisionError("Cannot invert zero in GF(256)")
    return int(GF_EXP[MAX_SHARES - int(GF_LOG[value])])


def gf_mul_array(values, scalar):
    scalar = int(scalar)
    if scalar == 0:
        return np.zeros_like(values, dtype=np.uint8)
    if scalar == 1:
        return values.astype(np.uint8, copy=True)

    result = np.zeros_like(values, dtype=np.uint8)
    nonzero = values != 0
    result[nonzero] = GF_EXP[GF_LOG[values[nonzero]] + int(GF_LOG[scalar])]
    return result


def decompress_limited(compressed, max_output_size):
    if max_output_size is None:
        return zlib.decompress(compressed)

    decompressor = zlib.decompressobj()
    payload = decompressor.decompress(compressed, max_output_size + 1)
    if len(payload) > max_output_size or decompressor.unconsumed_tail:
        raise ValueError("Shamir payload exceeds the expected image size")

    payload += decompressor.flush(max_output_size + 1 - len(payload))
    if len(payload) > max_output_size or not decompressor.eof:
        raise ValueError("Shamir payload exceeds the expected image size")
    return payload


def read_image_file(path):
    with Image.open(path) as img:
        img.verify()
    return Path(path).read_bytes()


def build_payload(image_path):
    image_bytes = read_image_file(image_path)
    compressed = zlib.compress(image_bytes, level=9)
    if len(compressed) < len(image_bytes):
        flags = PAYLOAD_FLAG_ZLIB
        stored = compressed
    else:
        flags = 0
        stored = image_bytes

    header = struct.pack(PAYLOAD_HEADER_FORMAT, PAYLOAD_MAGIC, flags, len(image_bytes), len(stored))
    return header + stored


def read_payload_header(payload):
    if len(payload) < PAYLOAD_HEADER_SIZE:
        raise ValueError("Payload is too small")
    magic, flags, original_size, stored_size = struct.unpack(
        PAYLOAD_HEADER_FORMAT,
        payload[:PAYLOAD_HEADER_SIZE],
    )
    if magic != PAYLOAD_MAGIC:
        raise ValueError("Invalid payload header")
    return flags, original_size, stored_size


def parse_payload(payload):
    if len(payload) < PAYLOAD_HEADER_SIZE:
        raise ValueError("Reconstructed payload is too small")

    magic, flags, original_size, stored_size = struct.unpack(
        PAYLOAD_HEADER_FORMAT,
        payload[:PAYLOAD_HEADER_SIZE],
    )
    if magic != PAYLOAD_MAGIC:
        raise ValueError("Invalid reconstructed payload header")
    if flags not in (0, PAYLOAD_FLAG_ZLIB):
        raise ValueError("Unsupported reconstructed payload flags")
    if stored_size > len(payload) - PAYLOAD_HEADER_SIZE:
        raise ValueError("Reconstructed payload is truncated")

    stored = payload[PAYLOAD_HEADER_SIZE : PAYLOAD_HEADER_SIZE + stored_size]
    if flags & PAYLOAD_FLAG_ZLIB:
        image_bytes = decompress_limited(stored, original_size)
        if len(image_bytes) != original_size:
            raise ValueError("Reconstructed payload has an invalid decompressed size")
        return image_bytes
    if stored_size != original_size:
        raise ValueError("Reconstructed payload has inconsistent sizes")
    return stored


def payload_to_share_image(payload):
    width = min(SHARE_IMAGE_WIDTH, max(1, len(payload)))
    height = math.ceil(len(payload) / width)
    padded_size = width * height
    padded = np.zeros(padded_size, dtype=np.uint8)
    padded[: len(payload)] = np.frombuffer(payload, dtype=np.uint8)
    return padded.reshape((height, width))


def share_image_to_bytes(path):
    with Image.open(path) as img:
        if img.mode != "L":
            raise ValueError(f"Share image must be a grayscale PNG: {path}")
        return np.asarray(img, dtype=np.uint8).flatten()


def save_share_image(path, share_bytes):
    Image.fromarray(payload_to_share_image(share_bytes)).save(path, compress_level=9)


def secure_random_bytes(shape):
    count = math.prod(shape)
    if count == 0:
        return np.empty(shape, dtype=np.uint8)
    return np.frombuffer(secrets.token_bytes(count), dtype=np.uint8).reshape(shape)


def iter_polynomial_shares(secret, n, r):
    validate_share_parameters(n, r)
    secret = np.asarray(secret, dtype=np.uint8)
    coefficients = secure_random_bytes((secret.shape[0], r - 1))

    for x_value in range(1, n + 1):
        share = coefficients[:, -1].copy()
        for degree in range(r - 3, -1, -1):
            share = gf_mul_array(share, x_value) ^ coefficients[:, degree]
        yield gf_mul_array(share, x_value) ^ secret


def polynomial(secret, n, r):
    return np.array(list(iter_polynomial_shares(secret, n, r)), dtype=np.uint8)


def format_size(size_bytes):
    if size_bytes == 0:
        return "0 B"
    size_names = ("B", "KB", "MB", "GB", "TB")
    unit_index = min(int(math.log(size_bytes, 1024)), len(size_names) - 1)
    scaled_size = round(size_bytes / (1024 ** unit_index), 2)
    return f"{scaled_size} {size_names[unit_index]}"


def get_file_size(file_path):
    try:
        return format_size(os.path.getsize(file_path))
    except OSError as exc:
        return f"Error: {exc}"


def build_progress(console):
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
        transient=not console.is_terminal,
    )


def render_file_results(title, rows, console):
    table = Table(title=title)
    table.add_column("#", justify="right", style="cyan", no_wrap=True)
    table.add_column("Path", style="green", overflow="fold")
    table.add_column("Size", justify="right", style="magenta")
    for index, result in enumerate(rows, start=1):
        table.add_row(str(index), Text(str(result.path)), result.size)
    console.print(table)


def render_kv_table(title, values, console):
    table = Table(title=title)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    for key, value in values:
        table.add_row(str(key), Text(str(value)))
    console.print(table)


def validate_share_parameters(n, r):
    if r is None:
        raise ValueError("Threshold number 'r' is required")
    if n is None:
        raise ValueError("Total number 'n' of shares is required")
    if r < 2:
        raise ValueError("Threshold 'r' must be at least 2")
    if n < r:
        raise ValueError("Total number 'n' must be greater than or equal to threshold 'r'")
    if n > MAX_SHARES:
        raise ValueError(f"Total number 'n' cannot exceed {MAX_SHARES} when using GF({FIELD_SIZE})")


def validate_indices(indices, r):
    if not indices:
        raise ValueError("At least 'r' share indexes are required for decoding")
    if len(indices) < r:
        raise ValueError("The number of share indexes must be greater than or equal to threshold 'r'")
    if len(set(indices)) != len(indices):
        raise ValueError("Share indexes must be unique")
    invalid = [index for index in indices if index < 1 or index > MAX_SHARES]
    if invalid:
        raise ValueError(f"Share indexes must be between 1 and {MAX_SHARES}: {invalid}")


def lagrange_weights_at_zero(indices):
    weights = []
    for current in indices:
        numerator = 1
        denominator = 1
        for other in indices:
            if current == other:
                continue
            numerator = gf_mul_byte(numerator, other)
            denominator = gf_mul_byte(denominator, current ^ other)
        weights.append(gf_mul_byte(numerator, gf_inverse(denominator)))
    return np.array(weights, dtype=np.uint8)


def decode(shares, index, r):
    if shares.shape[0] < r:
        raise ValueError("Not enough shares to reconstruct the image")

    selected_shares = shares[:r].astype(np.uint8)
    selected_index = index[:r]

    weights = lagrange_weights_at_zero(selected_index)
    secret = np.zeros(selected_shares.shape[1], dtype=np.uint8)
    for share, weight in zip(selected_shares, weights):
        secret ^= gf_mul_array(share, int(weight))
    return secret


def compare_images(image1_path, image2_path):
    image1 = np.array(Image.open(image1_path), dtype=np.int32)
    image2 = np.array(Image.open(image2_path), dtype=np.int32)
    if image1.shape != image2.shape:
        raise ValueError(f"Image shapes differ: {image1.shape} != {image2.shape}")

    diff = np.abs(image1 - image2)
    return {
        "Mean difference": round(float(np.mean(diff)), 4),
        "Max difference": round(float(np.max(diff)), 4),
        "Min difference": round(float(np.min(diff)), 4),
        "Standard deviation": round(float(np.std(diff)), 4),
    }


def build_parser():
    parser = argparse.ArgumentParser(description="Shamir Secret Image Sharing")
    parser.add_argument("-e", "--encode", help="Path to the image to be encoded")
    parser.add_argument("-d", "--decode", help="Path for the origin image to be saved")
    parser.add_argument("-n", type=int, help="The total number of shares")
    parser.add_argument("-r", type=int, help="The threshold number of shares to reconstruct the image")
    parser.add_argument("-i", "--index", nargs="+", type=int, help="The index of shares to use for decoding")
    parser.add_argument("-c", "--compare", nargs=2, help="Compare two images")
    parser.add_argument("--output-dir", default=".", help="Directory where encoded shares are saved")
    parser.add_argument("--share-dir", default=".", help="Directory where shares are read during decoding")
    parser.add_argument("--share-prefix", default="secret", help="Prefix used for share file names")
    return parser


def encode_image(args, console=CONSOLE):
    validate_share_parameters(args.n, args.r)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = build_payload(args.encode)
    flags, original_size, stored_size = read_payload_header(payload)
    compressed = bool(flags & PAYLOAD_FLAG_ZLIB)
    console.print(
        Panel.fit(
            Text(
                "\n".join(
                    [
                        f"Input: {args.encode}",
                        f"Shares: {args.n}",
                        f"Threshold: {args.r}",
                        f"Payload: {format_size(len(payload))}",
                        f"Compression: {'zlib' if compressed else 'none'}",
                    ]
                )
            ),
            title="Encoding",
            border_style="cyan",
        )
    )

    saved_shares = []
    shares = iter_polynomial_shares(np.frombuffer(payload, dtype=np.uint8), n=args.n, r=args.r)
    with build_progress(console) as progress:
        task = progress.add_task("Generating share PNGs", total=args.n)
        for i, share_bytes in enumerate(shares, start=1):
            secret_img_path = output_dir / f"{args.share_prefix}_{i}.png"
            save_share_image(secret_img_path, share_bytes)
            saved_shares.append(FileResult(secret_img_path, get_file_size(secret_img_path)))
            progress.advance(task)

    render_file_results("Generated Shares", saved_shares, console)
    return EncodeResult(
        input_path=Path(args.encode),
        payload_size=len(payload),
        original_size=original_size,
        stored_size=stored_size,
        compressed=compressed,
        shares=saved_shares,
    )


def decode_image(args, console=CONSOLE):
    if args.r is None:
        raise ValueError("Threshold number 'r' is required")
    validate_indices(args.index, args.r)

    console.print(
        Panel.fit(
            Text(
                "\n".join(
                    [
                        f"Output: {args.decode}",
                        f"Threshold: {args.r}",
                        f"Indexes: {', '.join(str(index) for index in args.index)}",
                    ]
                )
            ),
            title="Decoding",
            border_style="cyan",
        )
    )

    share_dir = Path(args.share_dir)
    input_shares = []
    share_size = None

    with build_progress(console) as progress:
        task = progress.add_task("Reading share PNGs", total=len(args.index))
        for index in args.index:
            secret_img_path = share_dir / f"{args.share_prefix}_{index}.png"
            share_bytes = share_image_to_bytes(secret_img_path)
            if share_size is not None and share_bytes.shape[0] != share_size:
                raise ValueError("All shares must have the same encoded byte length")
            share_size = share_bytes.shape[0]
            input_shares.append(share_bytes)
            progress.advance(task)

    with console.status("Reconstructing payload...", spinner="dots"):
        payload = decode(np.array(input_shares, dtype=np.uint8), args.index, r=args.r)

    with console.status("Writing recovered image...", spinner="dots"):
        Path(args.decode).write_bytes(parse_payload(payload.tobytes()))

    result = DecodeResult(Path(args.decode), get_file_size(args.decode))
    render_file_results("Recovered Image", [result], console)
    return result


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not any([args.encode, args.decode, args.compare]):
        parser.print_help()
        return

    try:
        if args.encode:
            start_time = time.time()
            CONSOLE.rule("[bold cyan]Image Encoding")
            encode_image(args)
            CONSOLE.print(f"[bold green]Encoding completed[/bold green] in {time.time() - start_time:.2f}s")

        if args.decode:
            start_time = time.time()
            CONSOLE.rule("[bold cyan]Image Decoding")
            decode_image(args)
            CONSOLE.print(f"[bold green]Decoding completed[/bold green] in {time.time() - start_time:.2f}s")

        if args.compare:
            CONSOLE.rule("[bold cyan]Image Comparison")
            stats = compare_images(args.compare[0], args.compare[1])
            render_kv_table("Pixel Difference", stats.items(), CONSOLE)
            CONSOLE.print("[bold green]Comparison completed[/bold green]")
    except (OSError, ValueError) as exc:
        ERROR_CONSOLE.print(Panel(Text(str(exc), style="bold red"), title="Error", border_style="red"))
        sys.exit(1)


if __name__ == "__main__":
    main()
