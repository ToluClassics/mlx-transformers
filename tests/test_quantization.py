import unittest

from src.mlx_transformers.quantization import (
    QuantizationConfig,
    QuantizationInfo,
)


class TestQuantizationConfig(unittest.TestCase):
    def test_affine_defaults_are_explicit(self):
        config = QuantizationConfig()

        self.assertEqual(config.group_size, 64)
        self.assertEqual(config.bits, 4)
        self.assertEqual(config.mode, "affine")
        self.assertEqual(
            config.as_dict(),
            {
                "group_size": 64,
                "bits": 4,
                "mode": "affine",
                "quantize_input": False,
            },
        )

    def test_specialized_modes_normalize_their_defaults(self):
        self.assertEqual(
            (QuantizationConfig(mode="mxfp4").group_size,
             QuantizationConfig(mode="mxfp4").bits),
            (32, 4),
        )
        self.assertEqual(
            (QuantizationConfig(mode="mxfp8").group_size,
             QuantizationConfig(mode="mxfp8").bits),
            (32, 8),
        )
        self.assertEqual(
            (QuantizationConfig(mode="nvfp4").group_size,
             QuantizationConfig(mode="nvfp4").bits),
            (16, 4),
        )

    def test_invalid_affine_settings_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "group_size must be"):
            QuantizationConfig(group_size=16)
        with self.assertRaisesRegex(ValueError, "bits must be"):
            QuantizationConfig(bits=7)

    def test_invalid_specialized_mode_settings_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "mxfp4.*requires"):
            QuantizationConfig(mode="mxfp4", group_size=64)
        with self.assertRaisesRegex(ValueError, "mxfp8.*requires"):
            QuantizationConfig(mode="mxfp8", bits=4)

    def test_quantize_input_is_restricted(self):
        with self.assertRaisesRegex(ValueError, "quantize_input=True"):
            QuantizationConfig(quantize_input=True)

        self.assertTrue(
            QuantizationConfig(mode="nvfp4", quantize_input=True).quantize_input
        )

    def test_checkpoint_mapping_and_info_are_normalized(self):
        config = QuantizationConfig.from_mapping(
            {"group_size": 32, "bits": 4}
        )
        info = QuantizationInfo.from_config(config, source="checkpoint")

        self.assertEqual(info.source, "checkpoint")
        self.assertEqual(info.mode, "affine")
        self.assertEqual(info.as_dict()["group_size"], 32)
