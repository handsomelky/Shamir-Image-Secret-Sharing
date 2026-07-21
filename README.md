# Shamir's Secret Sharing for Images

This project implements lossless image secret sharing based on Shamir's Secret Sharing. It splits an image into multiple PNG share images and reconstructs the original image from any threshold-sized subset of those shares.

[简体中文](readme/README.zh-CN.md) | [繁體中文](readme/README.zh-TW.md) | English

## Features

- Lossless reconstruction of the original image file bytes.
- Shamir sharing over GF(256), operating directly on image payload bytes.
- Compact grayscale PNG share containers instead of same-size noise images.
- Optional zlib payload compression when it reduces the shared data size.
- Cryptographically secure random polynomial coefficients.
- Vectorized NumPy reconstruction for better decoding performance.
- Rich-powered CLI status panels, progress bars, result tables, and error messages.
- Configurable share output directory, share input directory, and share filename prefix.
- Built-in image comparison command and unit tests.

## Installation

Requires Python 3.12 by default and `uv` for environment and dependency management. Dependencies are declared in `pyproject.toml`.

```shell
uv sync
```

Runtime dependencies:

- NumPy
- Pillow
- Rich

## Usage

Run the CLI through uv:

```shell
uv run python Shamir.py [options]
```

Command-line options:

| Option | Usage |
| --- | --- |
| `-e`, `--encode <image-path>` | Encode an input image into share images. Requires `-n` and `-r`. |
| `-d`, `--decode <output-path>` | Reconstruct the original image and save it to the given path. Requires `-r` and `-i`. |
| `-n <number-of-shares>` | Total number of shares to generate. Must be greater than or equal to `r` and no greater than 255. |
| `-r <threshold>` | Minimum number of shares required for reconstruction. Must be at least 2. |
| `-i`, `--index <share-indexes...>` | Share indexes used for decoding, for example `-i 1 4 5`. Values must be unique and within `1..255`. |
| `-c`, `--compare <image-a> <image-b>` | Compare two images and report mean, max, min, and standard deviation of pixel differences. |
| `--output-dir <dir>` | Directory where encoded shares are written. Defaults to the current directory. |
| `--share-dir <dir>` | Directory where shares are read during decoding. Defaults to the current directory. |
| `--share-prefix <prefix>` | Prefix for share filenames. Defaults to `secret`, producing names such as `secret_1.png`. |
| `-h`, `--help` | Show the full command-line help. |

## Example

```shell
# Generate 5 shares with a reconstruction threshold of 3
uv run python Shamir.py -e avatar.png -n 5 -r 3 --output-dir shares

# Reconstruct the image from shares 1, 4, and 5
uv run python Shamir.py -d avatar_recover.png -r 3 -i 1 4 5 --share-dir shares

# Confirm lossless recovery
uv run python Shamir.py -c avatar.png avatar_recover.png
```

## Algorithm

Each input is validated with Pillow as an image, then the original image file bytes are used as the secret payload. The payload is zlib-compressed only when compression makes it smaller.

The algorithm works over GF(256), so every shared value is a byte from 0 to 255. For each payload byte `s`, the encoder creates a random polynomial:

```text
f(x) = s + a1*x + a2*x^2 + ... + a(r-1)*x^(r-1) in GF(256)
```

For share index `x`, the stored share byte is `f(x)`. Share bytes are packed into grayscale PNG images. The share PNG dimensions are container dimensions for the encoded payload and do not match the original image dimensions.

## Share Container Format

Each share PNG is a grayscale byte container. The reconstructed byte payload starts with an internal binary header containing a magic value, compression flag, original byte length, and stored byte length. Padding bytes added to fit the PNG rectangle are ignored after reconstruction.

## Reconstruction

During reconstruction, the decoder:

1. Reads the selected grayscale share PNGs and flattens their bytes.
2. Computes Lagrange weights once for the selected share indices.
3. Reconstructs the shared payload bytes with NumPy vectorized GF(256) arithmetic.
4. Parses the payload header, decompresses when needed, and writes the original image file bytes.

At least `r` unique shares are required. Share indices must be between 1 and 255, and all selected shares must have the same encoded byte length.

## Validation

The CLI validates:

- `r >= 2`
- `n >= r`
- `n <= 255`
- decode indexes are present, unique, and within `1..255`
- selected shares are grayscale PNGs with matching encoded byte lengths
- reconstructed payload headers and sizes are valid
- compared images have matching shapes

## Testing

Run the test suite with:

```shell
uv run python -m unittest discover
```

The tests cover RGB round-trip recovery, GF(256) reconstruction, payload parsing, parameter validation, invalid decode indexes, share container behavior, and image comparison shape checks.

## Project Structure

```text
Shamir.py              Main CLI and algorithm implementation
pyproject.toml         Project metadata and dependencies
tests/                Unit tests
readme/               Localized README files
```

## Contributions

Contributions are welcome. Please open an issue or submit a pull request for bug fixes, tests, documentation updates, or implementation improvements.

## License

This project is licensed under the GNU General Public License v3.0. See [LICENSE](LICENSE) for details.
