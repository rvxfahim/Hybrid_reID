"""
SAM3-based human detection module.

Replaces YOLODetector with SAM3 text-prompted detection.
Uses a global text prompt (e.g. "humans") to detect and segment all
persons in each frame, returning results in the same format as YOLODetector.detect().

Requirements:
    - sam3 installed: pip install -e ".\sam3[notebooks]"
    - PyTorch 2.7+, CUDA 12.6+
    - HuggingFace authentication for SAM3 checkpoint: hf auth login
"""

import cv2
import numpy as np
import torch
import time
from PIL import Image

from profiling import profile_function


class SAM3Detector:
    def __init__(
        self,
        prompt: str = "humans",
        conf_threshold: float = 0.5,
        device: str = "cuda",
        fp16: bool = False,
        compile: bool = False,
    ):
        """
        Initialize SAM3 detector with a fixed text prompt.

        Args:
            prompt: Open-vocabulary text concept to detect, e.g. "humans", "person".
            conf_threshold: Minimum confidence threshold to keep a detection (default 0.5).
            device: Compute device ('cuda' or 'cpu').
            fp16: Cast model weights to float16 for faster inference on CUDA (default False).
            compile: Enable torch.compile on the model for faster repeated inference (default False).
        """
        self.prompt = prompt
        self.conf_threshold = conf_threshold
        self.device = device

        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        # torch.compile requires Triton; check availability and warn if missing
        if compile:
            try:
                import triton  # noqa: F401
            except ImportError:
                print("WARNING: --sam3-compile requested but Triton is not installed. "
                      "Falling back to eager mode. Install Triton to enable compilation.")
                compile = False

        extras = ", ".join(x for x in (["fp16"] if fp16 else []) + (["compile"] if compile else []))
        print(f"Loading SAM3 image model on {device}" + (f" ({extras})" if extras else "") + "...")
        self.model = build_sam3_image_model(device=device, compile=compile)
        if fp16 and device == "cuda":
            self.model = self.model.half()
        self.processor = Sam3Processor(self.model, device=device, confidence_threshold=conf_threshold)
        if fp16 and device == "cuda":
            # The processor's transform produces float32; append a half-cast so
            # the image dtype matches the fp16 model weights.
            import torchvision.transforms.v2 as T
            self.processor.transform = T.Compose(
                list(self.processor.transform.transforms)
                + [T.ToDtype(torch.float16, scale=False)]
            )
        print(f"SAM3 loaded. Prompt: '{self.prompt}'")

    @profile_function
    def detect(self, frame: np.ndarray) -> list:
        """
        Detect instances matching the text prompt in the frame.

        Args:
            frame: BGR image as a numpy array (H, W, 3).

        Returns:
            List of detections, each as:
                [x1, y1, x2, y2, confidence, class_id, mask_tensor]
            where mask_tensor is a float torch.Tensor of shape [H, W]
            or None if no mask is available. class_id is always 0 (person).
        """
        # Convert BGR (OpenCV) → RGB PIL Image required by SAM3
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        state = self.processor.set_image(pil_image)
        state = self.processor.set_text_prompt(prompt=self.prompt, state=state)

        boxes  = state.get("boxes")   # [N, 4] xyxy, scaled to original image size
        masks  = state.get("masks")   # [N, 1, H, W] bool tensor
        scores = state.get("scores")  # [N] float tensor

        detections = []

        if boxes is None or scores is None or len(boxes) == 0:
            return detections

        boxes  = boxes.detach().cpu()
        scores = scores.detach().cpu()
        if masks is not None:
            masks = masks.detach().cpu()

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i].tolist()
            score = float(scores[i])

            mask_tensor = None
            if masks is not None and i < len(masks):
                # masks[i] shape: [1, H, W] bool → squeeze to [H, W] float32
                mask_tensor = masks[i].squeeze(0).float()

            detections.append([x1, y1, x2, y2, score, 0, mask_tensor])

        return detections
