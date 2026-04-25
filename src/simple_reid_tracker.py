"""
Experimental per-frame appearance-based ReID tracker using DINOv2.

Bypasses DeepSORT entirely.  Each frame:
  1. YOLOv8-seg masks are applied to crops before DINOv2 feature extraction.
  2. Two-stage matching against gallery:
       Stage 1 – Hungarian assignment on EMA feature cost matrix (fast, recency-biased).
       Stage 2 – Min-distance over full gallery for detections still unmatched after
                 stage 1 (robust to occlusion and appearance drift).
  3. Unmatched detections get a new ID; matched ones update the gallery and EMA.

No Kalman filter, no motion model.  Pure appearance-based reID.
"""

import time
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
        dino_model: str = "dinov2_vitb14_reg",
        ema_alpha: float = 0.9,
        merge_threshold: float = 0.15,
        dormant_timeout: float = 5.0,
        fps: float | None = None,
    ):
        """
        Args:
            match_threshold: Maximum cosine distance to consider a match (0-2 range,
                             0 = identical, 2 = opposite).  Values <= threshold are matched.
            max_gallery_size: Number of feature vectors to keep per track ID.
            min_confidence: Minimum YOLO detection confidence to accept.
            model_path: Optional path to a custom DINOv2 model file.
            dino_model: Model name from DINO_MODEL_REGISTRY in feature_extractor.py,
                        e.g. 'dinov2_vitb14_reg', 'dinov3_vitb16'.
            ema_alpha: Weight for the existing EMA feature when blending in a new feature
                       (0-1).  Higher = slower adaptation, more recency bias towards the
                       track's historical appearance.  Used in stage-1 matching.
            merge_threshold: EMA cosine distance below which two tracks are considered the
                             same person and merged (smaller gallery into larger).  Should
                             be stricter (lower) than match_threshold.  Set to 0 to disable.
            dormant_timeout: Seconds without a detection before a track is deleted.
                             When fps is provided the timeout is measured in video time
                             (frame-count / fps), making it correct for recorded video.
                             When fps is None, wall-clock time is used (live streams only).
                             Set to 0 to disable pruning.
            fps: Frames-per-second of the video source.  Supply this for recorded video
                 so the dormant timeout is based on video time rather than machine time.
                 Leave as None for live webcam streams.
        """
        self.match_threshold = match_threshold
        self.max_gallery_size = max_gallery_size
        self.min_confidence = min_confidence
        self.ema_alpha = ema_alpha
        self.merge_threshold = merge_threshold
        self.dormant_timeout = dormant_timeout
        self.fps = fps  # None → wall-clock pruning; set → frame-count-based pruning

        # Feature extractor (DINOv2 / DINOv3)
        self.feature_extractor = FeatureExtractor(model_path, dino_model=dino_model)
        self.feature_dim = self.feature_extractor.feature_dim

        # Track state
        self.gallery: dict[int, deque] = {}          # track_id -> deque of feature vecs
        self.ema_features: dict[int, np.ndarray] = {}  # track_id -> EMA feature vector
        self.last_seen: dict[int, int] = {}           # track_id -> frame index
        self.last_seen_time: dict[int, float] = {}    # track_id -> wall-clock time (seconds)
        self.bboxes: dict[int, list] = {}             # track_id -> last [x1,y1,x2,y2]
        self.track_history: dict = defaultdict(lambda: deque(maxlen=50))
        self.next_id = 1
        self.frame_count = 0

        print(
            f"SimpleReIDTracker initialised | match_threshold={match_threshold} "
            f"| max_gallery_size={max_gallery_size} | ema_alpha={ema_alpha} "
            f"| merge_threshold={merge_threshold} | dormant_timeout={dormant_timeout}s "
            f"({'frame-based @ ' + str(fps) + ' fps' if fps else 'wall-clock'})"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _min_gallery_distance(self, query_feat: np.ndarray, track_id: int) -> float:
        """Minimum cosine distance from query_feat to any feature in the gallery.

        More discriminative than mean: preserves appearance modes (e.g. front/side
        view) and is unaffected by noisy crops that would corrupt a mean.
        """
        return min(self._cosine_distance(query_feat, f) for f in self.gallery[track_id])

    def _cosine_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine distance in [0, 2]."""
        a_norm = a / (np.linalg.norm(a) + 1e-8)
        b_norm = b / (np.linalg.norm(b) + 1e-8)
        return float(1.0 - np.dot(a_norm, b_norm))

    def _build_cost_matrix(
        self, det_features: list[np.ndarray], active_ids: list[int]
    ) -> np.ndarray:
        """
        Build an (N_det x M_tracks) cosine distance cost matrix using EMA features.
        Uses GPU when available, falls back to CPU.
        """
        n = len(det_features)
        m = len(active_ids)
        cost = np.full((n, m), fill_value=2.0, dtype=np.float32)

        if n == 0 or m == 0:
            return cost

        # Stage-1 uses EMA features (single vector per track, recency-weighted)
        gallery_feats = np.stack([self.ema_features[tid] for tid in active_ids]).reshape(m, -1)
        det_feats = np.stack(det_features).reshape(n, -1)

        if torch.cuda.is_available():
            d = torch.tensor(det_feats, dtype=torch.float32).cuda().reshape(n, -1)
            g = torch.tensor(gallery_feats, dtype=torch.float32).cuda().reshape(m, -1)
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
    # Internal helpers (continued)
    # ------------------------------------------------------------------

    def _prune_dormant_tracks(self) -> None:
        """Remove tracks that have not been seen for longer than dormant_timeout seconds.

        Uses video-time (frame count / fps) when fps is set — correct for recorded video.
        Falls back to wall-clock time when fps is None — suitable for live streams.
        """
        if self.dormant_timeout <= 0:
            return
        if self.fps and self.fps > 0:
            # Frame-based: accurate regardless of processing speed
            dormant_frames = self.dormant_timeout * self.fps
            to_delete = [
                tid for tid, f in self.last_seen.items()
                if self.frame_count - f > dormant_frames
            ]
        else:
            # Wall-clock fallback: use for live streams where there is no fixed fps
            now = time.time()
            to_delete = [
                tid for tid, t in self.last_seen_time.items()
                if now - t > self.dormant_timeout
            ]
        for tid in to_delete:
            self.gallery.pop(tid, None)
            self.ema_features.pop(tid, None)
            self.last_seen.pop(tid, None)
            self.last_seen_time.pop(tid, None)
            self.bboxes.pop(tid, None)
            self.track_history.pop(tid, None)

    def _merge_duplicate_tracks(self) -> dict[int, int]:
        """Merge track pairs whose EMA features are within merge_threshold of each other.

        The track with the smaller gallery is absorbed into the one with the larger
        gallery (IDY → IDX).  Returns a mapping {IDY: IDX} so callers can remap IDs.
        """
        if self.merge_threshold <= 0:
            return {}

        track_ids = list(self.ema_features.keys())
        merged: dict[int, int] = {}  # absorbed_id -> surviving_id

        # Work on a stable list; skip IDs already absorbed this round
        for i in range(len(track_ids)):
            id_a = track_ids[i]
            if id_a in merged:
                continue
            for j in range(i + 1, len(track_ids)):
                id_b = track_ids[j]
                if id_b in merged:
                    continue

                dist = self._cosine_distance(
                    self.ema_features[id_a], self.ema_features[id_b]
                )
                if dist > self.merge_threshold:
                    continue

                # Decide which is the survivor (larger gallery = IDX)
                len_a = len(self.gallery.get(id_a, []))
                len_b = len(self.gallery.get(id_b, []))
                if len_a >= len_b:
                    survivor, absorbed = id_a, id_b
                else:
                    survivor, absorbed = id_b, id_a

                # Merge gallery entries from absorbed into survivor
                for feat in self.gallery.get(absorbed, []):
                    self.gallery[survivor].append(feat)

                # Re-blend EMA: equal-weight average of both EMAs
                self.ema_features[survivor] = 0.5 * (
                    self.ema_features[survivor] + self.ema_features[absorbed]
                )

                # Merge track history (centre points)
                for pt in self.track_history.get(absorbed, []):
                    self.track_history[survivor].append(pt)

                # Remove absorbed track
                self.gallery.pop(absorbed, None)
                self.ema_features.pop(absorbed, None)
                self.last_seen.pop(absorbed, None)
                self.last_seen_time.pop(absorbed, None)
                self.bboxes.pop(absorbed, None)
                self.track_history.pop(absorbed, None)

                merged[absorbed] = survivor

                # If id_a was absorbed, it no longer exists — stop the inner loop
                if absorbed == id_a:
                    break

        return merged

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

        # ---- 0. Prune dormant tracks (before matching) ---------------
        self._prune_dormant_tracks()

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

            # --- Stage 1: Hungarian on EMA cost matrix ----------------
            cost = self._build_cost_matrix(valid_features, active_ids)
            row_ind, col_ind = linear_sum_assignment(cost)

            matched_track_cols = set()
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] <= self.match_threshold:
                    det_to_id[valid_indices[r]] = active_ids[c]
                    matched_track_cols.add(c)

            # --- Stage 2: min-gallery fallback for still-unmatched ----
            # Detections that failed stage 1 are compared to every stored
            # feature of every unmatched track; the nearest gallery vector
            # decides the match.  This recovers IDs after occlusion or
            # appearance drift where the EMA has moved away from the current
            # detection.
            unmatched_det_indices = [i for i in valid_indices if i not in det_to_id]
            unmatched_track_cols  = [c for c in range(len(active_ids))
                                     if c not in matched_track_cols]

            if unmatched_det_indices and unmatched_track_cols:
                for det_idx in unmatched_det_indices:
                    feat = det_features[det_idx]
                    best_dist = float('inf')
                    best_col  = -1
                    for col in unmatched_track_cols:
                        dist = self._min_gallery_distance(feat, active_ids[col])
                        if dist < best_dist:
                            best_dist = dist
                            best_col  = col
                    if best_dist <= self.match_threshold and best_col >= 0:
                        det_to_id[det_idx] = active_ids[best_col]
                        unmatched_track_cols.remove(best_col)

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

            # Update gallery and EMA with new feature (only if valid)
            if not np.all(feat == 0):
                self.gallery[track_id].append(feat)
                if track_id in self.ema_features:
                    # Blend: keep most of existing EMA, add small fraction of new feat
                    self.ema_features[track_id] = (
                        self.ema_alpha * self.ema_features[track_id]
                        + (1.0 - self.ema_alpha) * feat
                    )
                else:
                    # First observation: initialise EMA directly from this feature
                    self.ema_features[track_id] = feat.copy()

            # Update state
            self.last_seen[track_id] = self.frame_count
            self.last_seen_time[track_id] = time.time()
            self.bboxes[track_id] = list(bbox)

            # Update track history (centre point)
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            self.track_history[track_id].append((cx, cy))

            x1, y1, x2, y2 = [int(v) for v in bbox]
            result.append([x1, y1, x2, y2, track_id, cls_id])

        # ---- 5. Merge duplicate tracks (post-update) -----------------
        merged_ids = self._merge_duplicate_tracks()
        if merged_ids:
            result = [
                [x1, y1, x2, y2, merged_ids.get(tid, tid), cls]
                for x1, y1, x2, y2, tid, cls in result
            ]

        return result
