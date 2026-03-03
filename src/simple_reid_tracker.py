"""
Experimental per-frame appearance-based ReID tracker using DINOv2.

Bypasses DeepSORT entirely.  Each frame:
  1. YOLOv8-seg masks are applied to crops before DINOv2 feature extraction.
  2. A cosine-distance cost matrix is built between current detections and gallery.
  3. Hungarian assignment matches detections to existing IDs.
  4. Unmatched detections get a new ID; matched ones update the gallery.

No Kalman filter, no motion model.  Pure appearance-based reID.
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from collections import defaultdict, deque

from feature_extractor import FeatureExtractor
from profiling import profile_function


class SimpleReIDTracker:
    def __init__(
        self,
        match_threshold: float = 0.35,
        max_gallery_size: int = 10,
        min_confidence: float = 0.3,
        model_path=None,
    ):
        """
        Args:
            match_threshold: Maximum cosine distance to consider a match (0-2 range,
                             0 = identical, 2 = opposite).  Values <= threshold are matched.
            max_gallery_size: Number of feature vectors to keep per track ID.
            min_confidence: Minimum YOLO detection confidence to accept.
            model_path: Optional path to a custom DINOv2 model.
        """
        self.match_threshold = match_threshold
        self.max_gallery_size = max_gallery_size
        self.min_confidence = min_confidence

        # Feature extractor (DINOv2)
        self.feature_extractor = FeatureExtractor(model_path)
        self.feature_dim = self.feature_extractor.feature_dim

        # Track state
        self.gallery: dict[int, deque] = {}          # track_id -> deque of feature vecs
        self.last_seen: dict[int, int] = {}           # track_id -> frame index
        self.bboxes: dict[int, list] = {}             # track_id -> last [x1,y1,x2,y2]
        self.track_history: dict = defaultdict(lambda: deque(maxlen=50))
        self.next_id = 1
        self.frame_count = 0

        print(
            f"SimpleReIDTracker initialised | match_threshold={match_threshold} "
            f"| max_gallery_size={max_gallery_size}"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _mean_feature(self, track_id: int) -> np.ndarray:
        """Return the mean feature vector for a track."""
        feats = list(self.gallery[track_id])
        return np.mean(feats, axis=0).astype(np.float32)

    def _cosine_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine distance in [0, 2]."""
        a_norm = a / (np.linalg.norm(a) + 1e-8)
        b_norm = b / (np.linalg.norm(b) + 1e-8)
        return float(1.0 - np.dot(a_norm, b_norm))

    def _build_cost_matrix(
        self, det_features: list[np.ndarray], active_ids: list[int]
    ) -> np.ndarray:
        """
        Build an (N_det x M_tracks) cosine distance cost matrix using GPU.
        Falls back to CPU if CUDA not available.
        """
        n = len(det_features)
        m = len(active_ids)
        cost = np.full((n, m), fill_value=2.0, dtype=np.float32)

        if n == 0 or m == 0:
            return cost

        gallery_feats = np.stack([self._mean_feature(tid) for tid in active_ids])
        det_feats = np.stack(det_features)

        if torch.cuda.is_available():
            d = torch.tensor(det_feats, dtype=torch.float32).cuda()
            g = torch.tensor(gallery_feats, dtype=torch.float32).cuda()
            d = F.normalize(d, p=2, dim=1)
            g = F.normalize(g, p=2, dim=1)
            sim = torch.mm(d, g.t())            # (N, M)
            cost = (1.0 - sim).cpu().numpy()
        else:
            # CPU fallback
            d_norm = det_feats / (np.linalg.norm(det_feats, axis=1, keepdims=True) + 1e-8)
            g_norm = gallery_feats / (
                np.linalg.norm(gallery_feats, axis=1, keepdims=True) + 1e-8
            )
            cost = 1.0 - np.dot(d_norm, g_norm.T)

        return cost.astype(np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @profile_function
    def update(self, frame: np.ndarray, detections: list) -> list:
        """
        Process one frame.

        Args:
            frame: BGR image (numpy array).
            detections: List from YOLODetector.detect():
                        [x1, y1, x2, y2, conf, class_id, mask_tensor_or_None]

        Returns:
            List of [x1, y1, x2, y2, track_id, class_id] for all matched/new tracks.
        """
        self.frame_count += 1

        # ---- 1. Filter detections by confidence ----------------------
        valid_dets = [d for d in detections if len(d) >= 6 and d[4] >= self.min_confidence]

        if not valid_dets:
            return []

        bboxes = [d[:4] for d in valid_dets]
        class_ids = [d[5] for d in valid_dets]
        masks = [d[6] if len(d) > 6 else None for d in valid_dets]

        # ---- 2. Extract DINOv2 features (batch, with masks) ----------
        det_features = self.feature_extractor.extract_features_batch(frame, bboxes, masks)

        # Filter out zero-feature detections (invalid crops)
        valid_feat_mask = [not np.all(f == 0) for f in det_features]

        # ---- 3. Match detections against gallery ---------------------
        active_ids = list(self.gallery.keys())

        det_to_id: dict[int, int] = {}   # det_index -> assigned track_id

        if active_ids and any(valid_feat_mask):
            # Only match detections that have valid features
            valid_indices = [i for i, v in enumerate(valid_feat_mask) if v]
            valid_features = [det_features[i] for i in valid_indices]

            cost = self._build_cost_matrix(valid_features, active_ids)

            # Hungarian assignment on the sub-matrix of valid detections
            row_ind, col_ind = linear_sum_assignment(cost)

            matched_tracks = set()
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] <= self.match_threshold:
                    det_global_idx = valid_indices[r]
                    track_id = active_ids[c]
                    det_to_id[det_global_idx] = track_id
                    matched_tracks.add(c)

        # ---- 4. Assign IDs and update gallery -----------------------
        result = []
        for i, (bbox, cls_id, feat) in enumerate(zip(bboxes, class_ids, det_features)):
            if i in det_to_id:
                track_id = det_to_id[i]
            else:
                # New detection → new ID
                track_id = self.next_id
                self.next_id += 1
                self.gallery[track_id] = deque(maxlen=self.max_gallery_size)

            # Update gallery with new feature (only if valid)
            if not np.all(feat == 0):
                self.gallery[track_id].append(feat)

            # Update state
            self.last_seen[track_id] = self.frame_count
            self.bboxes[track_id] = list(bbox)

            # Update track history (centre point)
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            self.track_history[track_id].append((cx, cy))

            x1, y1, x2, y2 = [int(v) for v in bbox]
            result.append([x1, y1, x2, y2, track_id, cls_id])

        return result
