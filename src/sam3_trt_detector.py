"""
SAM3 TensorRT detector for the main_simple_reid pipeline.

Wraps the TensorRT FP16 engines (vision-encoder, text-encoder, decoder)
built by https://github.com/Kishan200308/SAM3-TENSORRT-PYTHON and exposes
the same detect(frame) API as SAM3Detector so that main_simple_reid.py can
use it as a drop-in replacement.

Key optimisation over the PyTorch path:
  - All three sub-models run as TensorRT FP16 engines (~2.5x faster)
  - The text-encoder is run ONCE at init and its output is cached; since
    the prompt is fixed for the lifetime of the detector this eliminates
    one engine call per frame.

Prerequisites
-------------
1. TensorRT installed and on PATH (tested with TRT 10.x, CUDA 12.x)
2. pip install tensorrt tokenizers torch
3. TRT engines built with Build_Engines.py from the TRT repo:
       hf download --local-dir "Onnx-Models" kishanstar2003/SAM3_ONNX_FP16
       python Build_Engines.py --onnx "Onnx-Models" --engine "Engines"
4. The engines directory must also contain tokenizer.json

Usage (via main_simple_reid.py)
--------------------------------
python main_simple_reid.py \\
    --detector sam3-trt \\
    --sam3-engines /path/to/Engines \\
    --sam3-prompt "person" \\
    --conf-threshold 0.5
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Optional, Tuple

from profiling import profile_function


# ---------------------------------------------------------------------------
# TRTModule – thin PyTorch-based wrapper around a serialised TensorRT engine.
# Adapted from SAM3_TensorRT_Inference.py in Kishan200308/SAM3-TENSORRT-PYTHON.
# ---------------------------------------------------------------------------

class TRTModule:
    """Load and run a serialised TensorRT engine using PyTorch CUDA tensors."""

    def __init__(self, engine_path: str):
        import tensorrt as trt  # lazy import – not available on all machines

        self._trt = trt
        self.logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(self.logger, "")

        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError(f"Failed to deserialise TRT engine: {engine_path}")

        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()

        # Cache IO tensor metadata
        self.io_info: Dict[str, dict] = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            self.io_info[name] = {
                "mode": self.engine.get_tensor_mode(name),
                "dtype": self.engine.get_tensor_dtype(name),
                "shape": self.engine.get_tensor_shape(name),
            }

    # ------------------------------------------------------------------
    def _trt_dtype_to_torch(self, trt_dtype) -> torch.dtype:
        trt = self._trt
        mapping = {
            trt.float32: torch.float32,
            trt.float16: torch.float16,
            trt.int32:   torch.int32,
            trt.int64:   torch.int64,
            trt.bool:    torch.bool,
        }
        return mapping.get(trt_dtype, torch.float32)

    def get_tensor_shape(self, name: str) -> Tuple:
        if name in self.io_info:
            return tuple(self.io_info[name]["shape"])
        raise KeyError(f"Tensor '{name}' not found in engine.")

    def __call__(self, **inputs: np.ndarray) -> Dict[str, np.ndarray]:
        trt = self._trt

        # Bind inputs
        for name, data in inputs.items():
            if name in self.io_info and self.io_info[name]["mode"] == trt.TensorIOMode.INPUT:
                dtype = self._trt_dtype_to_torch(self.io_info[name]["dtype"])
                tensor = torch.from_numpy(data).cuda().to(dtype).contiguous()
                self.context.set_input_shape(name, tuple(tensor.shape))
                self.context.set_tensor_address(name, int(tensor.data_ptr()))

        # Allocate outputs
        torch_outputs: Dict[str, torch.Tensor] = {}
        for name, info in self.io_info.items():
            if info["mode"] == trt.TensorIOMode.OUTPUT:
                shape = tuple(self.context.get_tensor_shape(name))
                dtype = self._trt_dtype_to_torch(info["dtype"])
                out = torch.empty(shape, dtype=dtype, device="cuda")
                torch_outputs[name] = out
                self.context.set_tensor_address(name, int(out.data_ptr()))

        self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        self.stream.synchronize()
        return {k: v.cpu().numpy() for k, v in torch_outputs.items()}


# ---------------------------------------------------------------------------
# SAM3TRTDetector
# ---------------------------------------------------------------------------

class SAM3TRTDetector:
    """
    SAM3 text-prompted detector backed by TensorRT FP16 engines.

    Returns detections in the same format as SAM3Detector:
        [x1, y1, x2, y2, confidence, class_id=0, mask_tensor_or_None]
    where coordinates are in **pixel space** of the *input* frame.
    """

    def __init__(
        self,
        prompt: str = "person",
        conf_threshold: float = 0.5,
        engines_dir: str = "Engines",
        device: str = "cuda",
    ):
        """
        Args:
            prompt:         Fixed text prompt used for every frame (e.g. "person").
            conf_threshold: Minimum combined confidence to keep a detection.
            engines_dir:    Directory containing vision-encoder.engine,
                            text-encoder.engine, decoder.engine, tokenizer.json.
            device:         Must be 'cuda'; TensorRT requires a CUDA device.
        """
        if device != "cuda" or not torch.cuda.is_available():
            raise RuntimeError(
                "SAM3TRTDetector requires a CUDA GPU. "
                "Pass --device cuda or use --detector sam3 for CPU fallback."
            )

        self.prompt = prompt
        self.conf_threshold = conf_threshold
        engines_path = Path(engines_dir)

        if not engines_path.exists():
            raise FileNotFoundError(
                f"Engines directory not found: {engines_dir}\n"
                "Build TRT engines first:\n"
                "  hf download --local-dir Onnx-Models kishanstar2003/SAM3_ONNX_FP16\n"
                "  python Build_Engines.py --onnx Onnx-Models --engine Engines"
            )

        print(f"[SAM3-TRT] Loading TensorRT engines from {engines_path} …")
        self._vision_encoder = self._load_engine(engines_path, "vision-encoder")
        self._text_encoder   = self._load_engine(engines_path, "text-encoder")
        self._decoder        = self._load_engine(engines_path, "decoder")

        # Auto-detect model input resolution
        try:
            ishape = self._vision_encoder.get_tensor_shape("images")
            self._target_h = int(ishape[2])
            self._target_w = int(ishape[3])
        except (KeyError, IndexError):
            self._target_h = self._target_w = 1024
            print("[SAM3-TRT] WARN: could not read input shape; defaulting to 1024×1024")
        print(f"[SAM3-TRT] Input resolution: {self._target_w}×{self._target_h}")

        # Tokenizer
        tokenizer_path = engines_path / "tokenizer.json"
        if not tokenizer_path.exists():
            raise FileNotFoundError(
                f"tokenizer.json not found in {engines_dir}. "
                "It should have been copied there by Build_Engines.py."
            )
        from tokenizers import Tokenizer
        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self._tokenizer.enable_padding(length=32, pad_id=49407)
        self._tokenizer.enable_truncation(max_length=32)

        # Cache text features once – prompt is fixed for the entire run
        print(f"[SAM3-TRT] Caching text features for prompt: '{prompt}'")
        self._cached_text_feats = self._encode_text(prompt)
        print("[SAM3-TRT] Ready.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_engine(engines_path: Path, name: str) -> TRTModule:
        """Find and load a named engine, with or without SM-suffix."""
        for p in engines_path.glob(f"{name}*.engine"):
            return TRTModule(str(p))
        raise FileNotFoundError(
            f"Engine for '{name}' not found in {engines_path}. "
            "Expected a file matching '{name}*.engine'."
        )

    def _encode_text(self, prompt: str) -> Dict[str, np.ndarray]:
        """Run the text encoder and return its outputs as numpy arrays."""
        tokens = self._tokenizer.encode(prompt)
        return self._text_encoder(
            input_ids=np.array([tokens.ids], dtype=np.int64),
            attention_mask=np.array([tokens.attention_mask], dtype=np.int64),
        )

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    # ------------------------------------------------------------------
    # Public API (matches SAM3Detector.detect)
    # ------------------------------------------------------------------

    @profile_function
    def detect(self, frame: np.ndarray) -> list:
        """
        Detect instances matching the text prompt in the frame.

        Args:
            frame: BGR image as numpy array (H, W, 3).

        Returns:
            List of detections, each as:
                [x1, y1, x2, y2, confidence, class_id=0, mask_tensor_or_None]
            Coordinates are pixel values in the *input* frame.
            mask_tensor is a float32 torch.Tensor [H, W] (input frame size)
            or None if the engine does not output masks.
        """
        h_orig, w_orig = frame.shape[:2]

        # ---- 1. Preprocess -----------------------------------------------
        # BGR → RGB, resize, normalise to [-1, 1], NCHW float32
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self._target_w, self._target_h),
                             interpolation=cv2.INTER_LINEAR)
        pixel_values = (resized.astype(np.float32) / 127.5 - 1.0).transpose(2, 0, 1)[None]

        # ---- 2. Vision encoder -------------------------------------------
        v_feats = self._vision_encoder(images=pixel_values)

        # ---- 3. Decoder (uses cached text features) ----------------------
        out = self._decoder(
            fpn_feat_0=v_feats["fpn_feat_0"],
            fpn_feat_1=v_feats["fpn_feat_1"],
            fpn_feat_2=v_feats["fpn_feat_2"],
            fpn_pos_2=v_feats["fpn_pos_2"],
            prompt_features=self._cached_text_feats["text_features"],
            prompt_mask=self._cached_text_feats["text_mask"],
        )

        # ---- 4. Post-process ---------------------------------------------
        # Combined score: sigmoid(pred_logits) * sigmoid(presence_logits[..., 0])
        scores = (self._sigmoid(out["pred_logits"][0]) *
                  self._sigmoid(out["presence_logits"][0, 0]))
        keep = scores > self.conf_threshold

        boxes_norm = out["pred_boxes"][0][keep]          # [M, 4] normalised xyxy
        kept_scores = scores[keep]

        has_masks = "pred_masks" in out
        kept_masks = out["pred_masks"][0][keep] if has_masks else None

        # ---- 5. Build detection list -------------------------------------
        detections = []
        for i, (box, score) in enumerate(zip(boxes_norm, kept_scores)):
            x1 = float(box[0]) * w_orig
            y1 = float(box[1]) * h_orig
            x2 = float(box[2]) * w_orig
            y2 = float(box[3]) * h_orig

            mask_tensor: Optional[torch.Tensor] = None
            if kept_masks is not None:
                # kept_masks[i] shape: [1, H_enc, W_enc] float logits
                mask_prob = self._sigmoid(kept_masks[i])
                if mask_prob.ndim == 3:
                    mask_prob = mask_prob[0]                  # → [H_enc, W_enc]
                mask_bin = (mask_prob > 0.5).astype(np.float32)
                # Resize mask back to original frame resolution
                mask_resized = cv2.resize(
                    mask_bin, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST
                )
                mask_tensor = torch.from_numpy(mask_resized)

            detections.append([x1, y1, x2, y2, float(score), 0, mask_tensor])

        return detections
