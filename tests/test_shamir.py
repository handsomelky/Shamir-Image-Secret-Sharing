import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from rich.console import Console

import Shamir


class ShamirTests(unittest.TestCase):
    def test_round_trip_rgb_image_with_any_threshold_subset(self):
        image = np.array(
            [
                [[0, 1, 2], [253, 254, 255]],
                [[10, 20, 30], [40, 50, 60]],
            ],
            dtype=np.uint8,
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "source.png"
            recovered = tmp_path / "recovered.png"
            Image.fromarray(image).save(source)

            encode_args = argparse_args(
                encode=str(source),
                n=5,
                r=3,
                output_dir=str(tmp_path),
                share_dir=str(tmp_path),
            )
            Shamir.encode_image(encode_args, console=quiet_console())

            decode_args = argparse_args(
                decode=str(recovered),
                r=3,
                index=[1, 4, 5],
                output_dir=str(tmp_path),
                share_dir=str(tmp_path),
            )
            Shamir.decode_image(decode_args, console=quiet_console())

            recovered_image = np.array(Image.open(recovered), dtype=np.uint8)
            self.assertTrue(np.array_equal(image, recovered_image))
            self.assertEqual(source.read_bytes(), recovered.read_bytes())

    def test_rejects_unsafe_share_parameters(self):
        with self.assertRaisesRegex(ValueError, "at least 2"):
            Shamir.validate_share_parameters(3, 1)
        with self.assertRaisesRegex(ValueError, "cannot exceed 255"):
            Shamir.validate_share_parameters(256, 3)
        with self.assertRaisesRegex(ValueError, "greater than or equal"):
            Shamir.validate_share_parameters(2, 3)

    def test_rejects_invalid_decode_indices(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            Shamir.validate_indices([1, 1, 2], 3)
        with self.assertRaisesRegex(ValueError, "At least"):
            Shamir.validate_indices(None, 3)
        with self.assertRaisesRegex(ValueError, "between"):
            Shamir.validate_indices([1, 2, 300], 3)

    def test_gf256_polynomial_round_trip(self):
        secret = np.frombuffer(bytes(range(64)), dtype=np.uint8)
        shares = np.array(list(Shamir.iter_polynomial_shares(secret, n=5, r=3)), dtype=np.uint8)

        recovered = Shamir.decode(shares[[0, 3, 4]], [1, 4, 5], r=3)

        self.assertTrue(np.array_equal(secret, recovered))

    def test_payload_round_trip_preserves_image_file_bytes(self):
        image = np.zeros((8, 8, 3), dtype=np.uint8)

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.png"
            Image.fromarray(image).save(source)

            payload = Shamir.build_payload(source)

            self.assertEqual(source.read_bytes(), Shamir.parse_payload(payload))

    def test_encoded_share_is_grayscale_payload_container(self):
        image = np.zeros((2, 2, 3), dtype=np.uint8)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "source.png"
            Image.fromarray(image).save(source)

            Shamir.encode_image(
                argparse_args(encode=str(source), n=3, r=2, output_dir=str(tmp_path)),
                console=quiet_console(),
            )

            with Image.open(tmp_path / "secret_1.png") as share:
                self.assertEqual("L", share.mode)
                self.assertNotEqual((2, 2), share.size)

    def test_rejects_invalid_reconstructed_payload(self):
        with self.assertRaisesRegex(ValueError, "Invalid reconstructed payload header"):
            Shamir.parse_payload(b"x" * Shamir.PAYLOAD_HEADER_SIZE)

    def test_compare_images_rejects_shape_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            first = tmp_path / "first.png"
            second = tmp_path / "second.png"
            Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(first)
            Image.fromarray(np.zeros((3, 2, 3), dtype=np.uint8)).save(second)

            with self.assertRaisesRegex(ValueError, "shapes differ"):
                Shamir.compare_images(first, second)

    def test_compare_images_returns_difference_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            first = tmp_path / "first.png"
            second = tmp_path / "second.png"
            Image.fromarray(np.zeros((2, 2), dtype=np.uint8)).save(first)
            Image.fromarray(np.ones((2, 2), dtype=np.uint8)).save(second)

            stats = Shamir.compare_images(first, second)

            self.assertEqual(1.0, stats["Mean difference"])
            self.assertEqual(1.0, stats["Max difference"])


def argparse_args(**overrides):
    defaults = {
        "encode": None,
        "decode": None,
        "n": None,
        "r": None,
        "index": None,
        "compare": None,
        "output_dir": ".",
        "share_dir": ".",
        "share_prefix": "secret",
    }
    defaults.update(overrides)
    return type("Args", (), defaults)()


def quiet_console():
    return Console(file=io.StringIO(), force_terminal=False, width=120)


if __name__ == "__main__":
    unittest.main()
