"""ROBIN-style latent watermarking helpers for ControlNet."""

from .fft_watermark import ROBINWatermarkConfig

__all__ = ["ROBINWatermarkConfig", "ControlNetROBINPipeline"]


def __getattr__(name):
    if name == "ControlNetROBINPipeline":
        from .pipeline import ControlNetROBINPipeline

        return ControlNetROBINPipeline
    raise AttributeError(name)
