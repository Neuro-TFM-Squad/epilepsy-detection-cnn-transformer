from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict, Any

import numpy as np

Event = Tuple[float, float]


@dataclass
class PostProcessConfig:
    window_step_s: float = 1.0

    smoothing: str = "moving_average"   # "none", "moving_average", "majority_vote"
    smoothing_size: int = 5

    threshold: float = 0.5

    use_hysteresis: bool = False
    threshold_on: float = 0.6
    threshold_off: float = 0.4

    min_event_duration_s: float = 0.0
    merge_gap_s: float = 0.0
    max_event_duration_s: Optional[float] = None

    clip_probs: bool = True
    eps: float = 1e-8

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PredictionPostProcessor:
    def __init__(self, config: PostProcessConfig):
        self.cfg = config
        self._validate_config()

    def _validate_config(self):
        if self.cfg.window_step_s <= 0:
            raise ValueError("window_step_s debe ser > 0.")

        if self.cfg.smoothing not in {"none", "moving_average", "majority_vote"}:
            raise ValueError("smoothing debe ser 'none', 'moving_average' o 'majority_vote'.")

        if self.cfg.smoothing_size < 1:
            raise ValueError("smoothing_size debe ser >= 1.")

        if self.cfg.use_hysteresis:
            if not (0.0 <= self.cfg.threshold_off <= 1.0):
                raise ValueError("threshold_off debe estar en [0,1].")
            if not (0.0 <= self.cfg.threshold_on <= 1.0):
                raise ValueError("threshold_on debe estar en [0,1].")
            if self.cfg.threshold_on < self.cfg.threshold_off:
                raise ValueError("threshold_on debe ser >= threshold_off.")

        if not (0.0 <= self.cfg.threshold <= 1.0):
            raise ValueError("threshold debe estar en [0,1].")

    def _safe_array(self, x) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        arr = np.squeeze(arr)
        arr = np.ravel(arr)
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)

        if self.cfg.clip_probs:
            arr = np.clip(arr, 0.0, 1.0)

        return arr.astype(np.float32)

    def _moving_average(self, x: np.ndarray) -> np.ndarray:
        k = max(1, int(self.cfg.smoothing_size))
        if k == 1:
            return x.copy()

        kernel = np.ones(k, dtype=np.float32) / float(k)
        return np.convolve(x, kernel, mode="same").astype(np.float32)

    def _majority_vote(self, x: np.ndarray) -> np.ndarray:
        k = max(1, int(self.cfg.smoothing_size))
        if k == 1:
            return (x >= self.cfg.threshold).astype(np.uint8).astype(np.float32)

        binary = (x >= self.cfg.threshold).astype(np.float32)
        kernel = np.ones(k, dtype=np.float32)
        votes = np.convolve(binary, kernel, mode="same")
        out = (votes >= (k / 2.0)).astype(np.float32)
        return out

    def _smooth(self, probs: np.ndarray) -> np.ndarray:
        if self.cfg.smoothing == "none":
            return probs.copy()

        if self.cfg.smoothing == "moving_average":
            return self._moving_average(probs)

        if self.cfg.smoothing == "majority_vote":
            return self._majority_vote(probs)

        raise RuntimeError("Configuración de smoothing no soportada.")

    def _binarize_standard(self, x: np.ndarray) -> np.ndarray:
        return (x >= float(self.cfg.threshold)).astype(np.uint8)

    def _binarize_hysteresis(self, x: np.ndarray) -> np.ndarray:
        on = float(self.cfg.threshold_on)
        off = float(self.cfg.threshold_off)

        out = np.zeros(len(x), dtype=np.uint8)
        active = False

        for i, val in enumerate(x):
            if not active and val >= on:
                active = True
            elif active and val < off:
                active = False
            out[i] = 1 if active else 0

        return out

    def _binarize(self, x: np.ndarray) -> np.ndarray:
        if self.cfg.use_hysteresis:
            return self._binarize_hysteresis(x)
        return self._binarize_standard(x)

    def _binary_to_events(self, binary: np.ndarray) -> List[Event]:
        events: List[Event] = []
        in_event = False
        start_idx = 0

        for i, val in enumerate(binary):
            if val == 1 and not in_event:
                in_event = True
                start_idx = i
            elif val == 0 and in_event:
                events.append((
                    start_idx * self.cfg.window_step_s,
                    i * self.cfg.window_step_s
                ))
                in_event = False

        if in_event:
            events.append((
                start_idx * self.cfg.window_step_s,
                len(binary) * self.cfg.window_step_s
            ))

        return events

    def _filter_short_events(self, events: List[Event]) -> List[Event]:
        min_dur = self.cfg.min_event_duration_s
        if min_dur is None or min_dur <= 0:
            return events

        return [(s, e) for s, e in events if (e - s) >= min_dur]

    def _merge_close_events(self, events: List[Event]) -> List[Event]:
        gap = self.cfg.merge_gap_s
        if not events:
            return []

        if gap is None or gap <= 0:
            return sorted(events, key=lambda x: x[0])

        events = sorted(events, key=lambda x: x[0])
        merged = [events[0]]

        for s, e in events[1:]:
            last_s, last_e = merged[-1]
            if (s - last_e) <= gap:
                merged[-1] = (last_s, max(last_e, e))
            else:
                merged.append((s, e))

        return merged

    def _split_long_events(self, events: List[Event]) -> List[Event]:
        max_dur = self.cfg.max_event_duration_s
        if max_dur is None or max_dur <= 0:
            return events

        out: List[Event] = []
        for s, e in events:
            dur = e - s
            if dur <= max_dur:
                out.append((s, e))
                continue

            cur = s
            while cur < e:
                nxt = min(cur + max_dur, e)
                out.append((cur, nxt))
                cur = nxt

        return out

    def process(self, y_pred_prob, return_intermediates: bool = False):
        probs = self._safe_array(y_pred_prob)
        smoothed = self._smooth(probs)
        binary = self._binarize(smoothed)

        events_raw = self._binary_to_events(binary)
        events_filtered = self._filter_short_events(events_raw)
        events_merged = self._merge_close_events(events_filtered)
        events_final = self._split_long_events(events_merged)

        if not return_intermediates:
            return events_final

        return {
            "probs": probs,
            "smoothed_probs": smoothed,
            "binary": binary,
            "events_raw": events_raw,
            "events_filtered": events_filtered,
            "events_merged": events_merged,
            "events_final": events_final,
            "config": self.cfg.to_dict(),
        }