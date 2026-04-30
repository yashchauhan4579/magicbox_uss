"""
Crowd analysis: CCN (Crowd Counting Network) model + CrowdAnalyticsState.

Density-based crowd counting using multi-scale feature fusion,
plus zone analysis, trend, risk, anomalies, hotspots, compression, and flow.
"""

import time
import os
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from vehicle import FullHeatmapRenderer


# ── CCN Neural Network ──

class FuseLayer(nn.Module):
    """Multi-scale dilated convolution fusion layer."""
    def __init__(self, in_channels=512):
        super().__init__()
        self.conv1x1_d1 = nn.Conv2d(in_channels, in_channels, 1)
        self.conv1x1_d2 = nn.Conv2d(in_channels, in_channels, 1)
        self.conv1x1_d3 = nn.Conv2d(in_channels, in_channels, 1)
        self.conv1x1_d4 = nn.Conv2d(in_channels, in_channels, 1)

        self.conv_d1 = nn.Conv2d(in_channels, in_channels, 3, padding=1, dilation=1)
        self.conv_d2 = nn.Conv2d(in_channels, in_channels, 3, padding=2, dilation=2)
        self.conv_d3 = nn.Conv2d(in_channels, in_channels, 3, padding=3, dilation=3)
        self.conv_d4 = nn.Conv2d(in_channels, in_channels, 3, padding=4, dilation=4)

    def forward(self, x):
        d1 = F.relu(self.conv_d1(F.relu(self.conv1x1_d1(x))))
        d2 = F.relu(self.conv_d2(F.relu(self.conv1x1_d2(x))))
        d3 = F.relu(self.conv_d3(F.relu(self.conv1x1_d3(x))))
        d4 = F.relu(self.conv_d4(F.relu(self.conv1x1_d4(x))))
        return d1 + d2 + d3 + d4


class CountLayer(nn.Module):
    """Counting layer with pooling."""
    def __init__(self, in_channels=512):
        super().__init__()
        self.avgpool_layer = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
        )
        self.maxpool_layer = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
        )
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1),
        )

    def forward(self, x):
        avg_out = self.avgpool_layer(F.adaptive_avg_pool2d(x, x.shape[-2:]))
        max_out = self.maxpool_layer(F.adaptive_max_pool2d(x, x.shape[-2:]))
        out = torch.cat([avg_out, max_out], dim=1)
        return F.relu(self.conv1x1(out))


class CCN(nn.Module):
    """Crowd Counting Network."""
    def __init__(self):
        super().__init__()

        # VGG-style feature extraction (layer3)
        self.layer3 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.layer4 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
        )

        self.layer5 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
        )

        self.fuse_layer3 = FuseLayer(512)
        self.fuse_layer4 = FuseLayer(512)
        self.fuse_layer5 = FuseLayer(512)

        self.count_layer3 = CountLayer(512)
        self.count_layer4 = CountLayer(512)
        self.count_layer5 = CountLayer(512)

        self.layer3_k = nn.Sequential(nn.Conv2d(512, 1, 1))
        self.layer4_k = nn.Sequential(nn.Conv2d(512, 1, 1))
        self.layer5_k = nn.Sequential(nn.Conv2d(512, 1, 1))

        self.layer3_i = nn.Sequential(nn.Conv2d(512, 3, 1))
        self.layer4_i = nn.Sequential(nn.Conv2d(512, 3, 1))
        self.layer5_i = nn.Sequential(nn.Conv2d(512, 3, 1))

        self.layer3_p = nn.Sequential(nn.Conv2d(512, 3, 1))
        self.layer4_p = nn.Sequential(nn.Conv2d(512, 3, 1))
        self.layer5_p = nn.Sequential(nn.Conv2d(512, 3, 1))

    def forward(self, x):
        f3 = self.layer3(x)
        f4 = self.layer4(f3)
        f5 = self.layer5(f4)

        fused3 = self.fuse_layer3(f3)
        fused4 = self.fuse_layer4(f4)
        fused5 = self.fuse_layer5(f5)

        c3 = self.count_layer3(fused3)
        c4 = self.count_layer4(fused4)
        c5 = self.count_layer5(fused5)

        density3 = F.relu(self.layer3_k(c3))
        density4 = F.relu(self.layer4_k(c4))
        density5 = F.relu(self.layer5_k(c5))

        h, w = density3.shape[-2:]
        density4_up = F.interpolate(density4, size=(h, w), mode='bilinear', align_corners=False)
        density5_up = F.interpolate(density5, size=(h, w), mode='bilinear', align_corners=False)

        density = (density3 + density4_up + density5_up) / 3.0
        return density


class CrowdCounter:
    """Wrapper for crowd counting inference."""

    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = device
        self.model = CCN()

        state_dict = torch.load(model_path, map_location=device, weights_only=False)

        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('CCN.'):
                new_state_dict[k[4:]] = v
            else:
                new_state_dict[k] = v

        self.model.load_state_dict(new_state_dict)
        self.model.to(device)
        self.model.eval()

        self.max_density_ema = None
        self.prev_gray = None

        # Tunables (inspired by UAV density-map cleanup workflows).
        self.target_size = int(os.getenv("IRIS_CROWD_TARGET_SIZE", "1024"))
        self.noise_floor_pct = float(os.getenv("IRIS_CROWD_NOISE_FLOOR_PCT", "70"))
        self.hotspot_hi_pct = float(os.getenv("IRIS_CROWD_HOTSPOT_HI_PCT", "99.5"))
        self.green_suppress = float(os.getenv("IRIS_CROWD_GREEN_SUPPRESS", "0.55"))
        self.motion_suppress = float(os.getenv("IRIS_CROWD_MOTION_SUPPRESS", "0.35"))
        self.min_component_ratio = float(os.getenv("IRIS_CROWD_MIN_COMPONENT_RATIO", "0.00008"))
        # Legacy crowdanalysis profile used a 1.8x calibration multiplier for reported count.
        self.count_multiplier = float(os.getenv("IRIS_CROWD_COUNT_MULTIPLIER", "1.8"))

        print(f"[CCN] Crowd counting model loaded on {device}")

    def preprocess(self, frame: np.ndarray, target_size: int | None = None) -> torch.Tensor:
        """Preprocess frame for inference."""
        if target_size is None:
            target_size = self.target_size
        h, w = frame.shape[:2]
        scale = target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)

        new_h = (new_h // 32) * 32
        new_w = (new_w // 32) * 32

        if new_h == 0:
            new_h = 32
        if new_w == 0:
            new_w = 32

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(frame_rgb, (new_w, new_h))

        img = resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std

        img = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()
        return img.to(self.device), (h, w), (new_h, new_w)

    def _suppress_green_regions(self, frame: np.ndarray, density_map: np.ndarray) -> np.ndarray:
        """Reduce vegetation false positives (trees/grass) using HSV green mask."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Broad green range for drone scenes.
        green_mask = cv2.inRange(hsv, (30, 30, 20), (95, 255, 255)).astype(np.float32) / 255.0
        if self.green_suppress <= 0:
            return density_map
        keep = 1.0 - np.clip(self.green_suppress, 0.0, 0.95) * green_mask
        return density_map * keep

    def _suppress_static_noise(self, frame: np.ndarray, density_map: np.ndarray) -> np.ndarray:
        """
        Reduce persistent/static artifacts.
        Motion is used as a soft prior, not a hard gate (crowds can stand still).
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if self.prev_gray is None:
            self.prev_gray = gray
            return density_map

        diff = cv2.absdiff(gray, self.prev_gray)
        self.prev_gray = gray

        _, motion = cv2.threshold(diff, 12, 1.0, cv2.THRESH_BINARY)
        motion = cv2.dilate(motion.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(np.float32)

        # Keep most dynamic response; retain baseline on static zones.
        static_floor = np.clip(self.motion_suppress, 0.0, 0.9)
        keep = static_floor + (1.0 - static_floor) * motion
        return density_map * keep

    def _remove_small_components(self, density_map: np.ndarray) -> np.ndarray:
        """Remove tiny isolated blobs that often come from textured background."""
        peak = float(density_map.max())
        if peak <= 1e-8:
            return density_map

        active = (density_map >= peak * 0.08).astype(np.uint8)
        nlabels, labels, stats, _ = cv2.connectedComponentsWithStats(active, connectivity=8)
        if nlabels <= 1:
            return density_map

        min_area = max(8, int(self.min_component_ratio * density_map.shape[0] * density_map.shape[1]))
        keep = np.zeros_like(active)
        for i in range(1, nlabels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                keep[labels == i] = 1
        return density_map * keep.astype(np.float32)

    @staticmethod
    def _warm_gradient(norm: np.ndarray) -> np.ndarray:
        """Warm gradient: transparent(0) → yellow(0.3) → orange(0.6) → red(1.0). Returns BGR uint8."""
        h, w = norm.shape[:2]
        out = np.zeros((h, w, 3), dtype=np.uint8)
        # Yellow (0,255,255 BGR) → Orange (0,165,255) → Red (0,0,255)
        r = np.clip(255 * np.ones_like(norm), 0, 255)
        g = np.clip(255 * (1.0 - norm), 0, 255)
        b = np.zeros_like(norm)
        out[..., 2] = r.astype(np.uint8)   # R
        out[..., 1] = g.astype(np.uint8)   # G
        out[..., 0] = b.astype(np.uint8)   # B
        return out

    def _build_heatmap(self, density_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        irisv3 crowd-worker style heatmap.
        Returns (heatmap_bgr, alpha) where alpha masks out zero-density areas
        so original frame shows through where there's no crowd.
        """
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as _plt

        d = np.maximum(density_map, 0).astype(np.float32)
        h, w = d.shape[:2]
        min_val = float(d.min())
        max_val = float(d.max())

        if max_val <= min_val or max_val == 0:
            return np.zeros((h, w, 3), dtype=np.uint8), np.zeros((h, w), dtype=np.float32)

        norm_obj = _plt.Normalize(vmin=min_val, vmax=max_val)
        cmap = _plt.get_cmap('jet')
        heatmap_rgba = cmap(norm_obj(d))
        heatmap_rgb = np.delete(heatmap_rgba, 3, 2)
        heatmap_bgr = cv2.cvtColor((heatmap_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

        # Alpha: only show heatmap where density is meaningful
        # This prevents JET's blue from washing over the entire frame
        norm_d = (d - min_val) / (max_val - min_val)
        alpha = np.clip(norm_d * 2.5, 0.0, 0.7).astype(np.float32)
        alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=8.0)
        alpha[norm_d < 0.08] = 0.0

        return heatmap_bgr, alpha

    @torch.no_grad()
    def count(self, frame: np.ndarray) -> dict:
        """Count people in frame with real-time heatmap generation."""
        img_tensor, orig_size, proc_size = self.preprocess(frame)

        density = self.model(img_tensor)

        density_np = density.squeeze().cpu().numpy()
        density_resized = cv2.resize(density_np, (orig_size[1], orig_size[0]), interpolation=cv2.INTER_CUBIC)
        density_resized = np.maximum(density_resized, 0).astype(np.float32)
        density_resized = cv2.GaussianBlur(density_resized, (0, 0), 1.2)

        # Adaptive floor removal.
        positive = density_resized[density_resized > 0]
        if positive.size > 8:
            floor = float(np.percentile(positive, self.noise_floor_pct))
            density_resized = np.maximum(0.0, density_resized - floor)
        else:
            density_resized[density_resized < 1e-4] = 0.0

        # UAV-specific cleanup: vegetation and static-texture suppression.
        density_resized = self._suppress_green_regions(frame, density_resized)
        density_resized = self._suppress_static_noise(frame, density_resized)
        density_resized = self._remove_small_components(density_resized)

        count = float(np.maximum(density_resized, 0).sum())
        count *= max(0.1, self.count_multiplier)
        heatmap, heat_alpha = self._build_heatmap(density_resized)

        return {
            'count': int(round(count)),
            'density_map': density_resized,
            'heatmap': heatmap,
            'heat_alpha': heat_alpha,
        }


# ── Crowd Analytics State ──

class CrowdAnalyticsState:
    """Track crowd analytics from CCN density maps for a single video source."""

    ZONE_ROWS = 4
    ZONE_COLS = 6

    THRESH_SPARSE = 15
    THRESH_GATHERING = 40
    THRESH_DENSE = 70

    def __init__(self, width: int, height: int, fps: float):
        self.width = width
        self.height = height
        self.fps = fps

        self.count_history = deque(maxlen=300)
        self.density_snapshots = deque(maxlen=30)
        self.prev_zone_densities = None

        self.zone_persistence_start = {}

        self.peak_count = 0
        self.peak_density = 0.0
        self.peak_window_start = time.time()

        self.fps_frame_count = 0
        self.fps_last_time = time.time()
        self.fps_value = 0.0

    def _classify_density(self, d):
        if d < self.THRESH_SPARSE:
            return "sparse"
        elif d < self.THRESH_GATHERING:
            return "gathering"
        elif d < self.THRESH_DENSE:
            return "dense"
        return "critical"

    def _compute_zones(self, density_map):
        """Partition density_map into 4x6 grid, sum per zone, normalize to 0-100."""
        h, w = density_map.shape[:2]
        cell_h = h / self.ZONE_ROWS
        cell_w = w / self.ZONE_COLS
        zones = []
        for r in range(self.ZONE_ROWS):
            for c in range(self.ZONE_COLS):
                y1 = int(r * cell_h)
                y2 = int((r + 1) * cell_h)
                x1 = int(c * cell_w)
                x2 = int((c + 1) * cell_w)
                cell = density_map[y1:y2, x1:x2]
                raw_sum = float(cell.sum())
                zones.append(raw_sum)
        max_val = max(zones) if zones else 1.0
        if max_val < 1e-6:
            max_val = 1.0
        normalized = [min(100.0, (z / max_val) * 100.0) for z in zones]
        return normalized

    def _compute_trend(self):
        """Linear regression on count_history over last 15s."""
        now = time.time()
        recent = [(t, c) for t, c in self.count_history if now - t <= 15.0]
        if len(recent) < 3:
            return "stable", 0.0

        ts = np.array([r[0] - recent[0][0] for r in recent])
        counts = np.array([r[1] for r in recent])

        if ts[-1] - ts[0] < 1.0:
            return "stable", 0.0

        n = len(ts)
        sx = ts.sum()
        sy = counts.sum()
        sxx = (ts * ts).sum()
        sxy = (ts * counts).sum()
        denom = n * sxx - sx * sx
        if abs(denom) < 1e-9:
            return "stable", 0.0
        slope = (n * sxy - sx * sy) / denom

        if slope > 0.5:
            return "increasing", round(float(slope), 2)
        elif slope < -0.5:
            return "decreasing", round(float(slope), 2)
        return "stable", round(float(slope), 2)

    def _merge_hotspots(self, zone_densities):
        """Flood-fill merge adjacent zones above gathering threshold."""
        ncells = self.ZONE_ROWS * self.ZONE_COLS
        hot_set = set()
        for i, d in enumerate(zone_densities):
            if d >= self.THRESH_GATHERING:
                hot_set.add(i)

        visited = set()
        regions = []
        for cell in hot_set:
            if cell in visited:
                continue
            region_cells = []
            queue = [cell]
            while queue:
                c = queue.pop(0)
                if c in visited:
                    continue
                visited.add(c)
                region_cells.append(c)
                r, co = divmod(c, self.ZONE_COLS)
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc_ = r + dr, co + dc
                    if 0 <= nr < self.ZONE_ROWS and 0 <= nc_ < self.ZONE_COLS:
                        ni = nr * self.ZONE_COLS + nc_
                        if ni in hot_set and ni not in visited:
                            queue.append(ni)
            if region_cells:
                regions.append(region_cells)

        now = time.time()
        hotspots = []
        for ri, cells in enumerate(regions):
            avg_d = sum(zone_densities[c] for c in cells) / len(cells)
            max_d = max(zone_densities[c] for c in cells)

            if max_d >= self.THRESH_DENSE:
                severity = "HIGH" if max_d >= self.THRESH_DENSE else "MODERATE"
            else:
                severity = "MODERATE" if avg_d >= self.THRESH_GATHERING else "LOW"
            if max_d >= self.THRESH_DENSE + 15:
                severity = "HIGH"

            size_pct = round(len(cells) / ncells * 100, 1)

            earliest = min((self.zone_persistence_start.get(c, now) for c in cells), default=now)
            persistence = round(now - earliest, 1)

            hotspots.append({
                "id": ri,
                "severity": severity,
                "avg_density": round(avg_d, 1),
                "size_pct": size_pct,
                "persistence": persistence,
                "zone_count": len(cells),
            })

        hotspots.sort(key=lambda x: (-x["avg_density"]))
        return hotspots

    def _compute_risk(self, max_density, trend_rate, zone_densities):
        """Weighted composite risk score 0-100."""
        density_score = min(100.0, max_density) * 0.35

        trend_score = min(100.0, max(0.0, trend_rate * 20.0 + 50.0)) * 0.25

        sat_count = sum(1 for d in zone_densities if d >= self.THRESH_GATHERING)
        sat_pct = sat_count / max(1, len(zone_densities)) * 100
        sat_score = sat_pct * 0.20

        now = time.time()
        max_persistence = 0.0
        for idx, start in self.zone_persistence_start.items():
            max_persistence = max(max_persistence, now - start)
        persist_score = min(100.0, max_persistence / 1.2) * 0.20

        return min(100, max(0, int(density_score + trend_score + sat_score + persist_score + 0.5)))

    def _detect_anomalies(self):
        """Detect flash gathering, panic dispersal, stationary buildup."""
        anomalies = []
        now = time.time()

        recent_5s = [(t, c) for t, c in self.count_history if now - t <= 5.0]
        recent_3s = [(t, c) for t, c in self.count_history if now - t <= 3.0]
        older_10s = [(t, c) for t, c in self.count_history if 5.0 < now - t <= 10.0]

        if len(recent_5s) >= 2 and len(older_10s) >= 2:
            recent_avg = sum(c for _, c in recent_5s) / len(recent_5s)
            older_avg = sum(c for _, c in older_10s) / len(older_10s)

            if older_avg > 0 and (recent_avg - older_avg) / older_avg > 0.5:
                anomalies.append({
                    "type": "flash_gathering",
                    "description": f"Rapid crowd increase: {int(older_avg)} -> {int(recent_avg)} in 5s",
                    "severity": "high",
                })

        if len(recent_3s) >= 2 and len(older_10s) >= 2:
            recent_avg = sum(c for _, c in recent_3s) / len(recent_3s)
            older_avg = sum(c for _, c in older_10s) / len(older_10s)

            if older_avg > 5 and (older_avg - recent_avg) / older_avg > 0.4:
                anomalies.append({
                    "type": "panic_dispersal",
                    "description": f"Rapid crowd decrease: {int(older_avg)} -> {int(recent_avg)} in 3s",
                    "severity": "critical",
                })

        for idx, start in self.zone_persistence_start.items():
            if now - start > 120.0:
                r, c = divmod(idx, self.ZONE_COLS)
                anomalies.append({
                    "type": "stationary_buildup",
                    "description": f"Zone ({r},{c}) crowded for {int(now - start)}s",
                    "severity": "medium",
                })
                break

        return anomalies

    def _compute_compression(self):
        """Track spatial std-dev of density over time."""
        if len(self.density_snapshots) < 3:
            return "stable"

        recent_stds = []
        for _, densities in list(self.density_snapshots)[-5:]:
            recent_stds.append(np.std(densities))

        older_stds = []
        for _, densities in list(self.density_snapshots)[:5]:
            older_stds.append(np.std(densities))

        if not recent_stds or not older_stds:
            return "stable"

        recent_avg = np.mean(recent_stds)
        older_avg = np.mean(older_stds)

        if recent_avg > older_avg * 1.3:
            return "compressing"
        elif recent_avg < older_avg * 0.7:
            return "expanding"
        return "stable"

    def _compute_flow(self, zone_densities):
        """Delta of zone densities between updates."""
        if self.prev_zone_densities is None:
            return {"total_inflow": 0, "total_outflow": 0, "net_flow": 0}

        inflow = 0.0
        outflow = 0.0
        for i in range(len(zone_densities)):
            delta = zone_densities[i] - self.prev_zone_densities[i]
            if delta > 0:
                inflow += delta
            else:
                outflow += abs(delta)

        return {
            "total_inflow": round(inflow, 1),
            "total_outflow": round(outflow, 1),
            "net_flow": round(inflow - outflow, 1),
        }

    def update(self, count, density_map) -> dict:
        """Main update: derive all crowd metrics from count + density_map."""
        now = time.time()

        self.fps_frame_count += 1
        elapsed = now - self.fps_last_time
        if elapsed >= 1.0:
            self.fps_value = self.fps_frame_count / elapsed
            self.fps_frame_count = 0
            self.fps_last_time = now

        self.count_history.append((now, count))

        if now - self.peak_window_start > 300:
            self.peak_count = count
            self.peak_density = 0.0
            self.peak_window_start = now
        self.peak_count = max(self.peak_count, count)

        zone_densities = self._compute_zones(density_map)

        for i, d in enumerate(zone_densities):
            if d >= self.THRESH_GATHERING:
                if i not in self.zone_persistence_start:
                    self.zone_persistence_start[i] = now
            else:
                self.zone_persistence_start.pop(i, None)

        self.density_snapshots.append((now, zone_densities[:]))

        max_density = max(zone_densities) if zone_densities else 0
        avg_density = sum(zone_densities) / len(zone_densities) if zone_densities else 0
        self.peak_density = max(self.peak_density, max_density)

        density_class = self._classify_density(max_density)

        trend, trend_rate = self._compute_trend()

        zones = []
        zone_dist = {"sparse": 0, "gathering": 0, "dense": 0, "critical": 0}
        for i, d in enumerate(zone_densities):
            r, c = divmod(i, self.ZONE_COLS)
            dc = self._classify_density(d)
            zone_dist[dc] += 1
            flow_delta = 0.0
            if self.prev_zone_densities is not None:
                flow_delta = round(d - self.prev_zone_densities[i], 1)
            persistence = round(now - self.zone_persistence_start.get(i, now), 1) if i in self.zone_persistence_start else 0.0
            zones.append({
                "index": i,
                "row": r,
                "col": c,
                "density": round(d, 1),
                "density_class": dc,
                "persistence": persistence,
                "flow_delta": flow_delta,
            })

        hotspots = self._merge_hotspots(zone_densities)
        risk_score = self._compute_risk(max_density, trend_rate, zone_densities)

        if risk_score < 30:
            operational_status = "MONITOR"
        elif risk_score < 60:
            operational_status = "ALERT"
        else:
            operational_status = "IMMEDIATE ACTION"

        time_to_critical = None
        if trend_rate > 0.1 and max_density < self.THRESH_DENSE:
            remaining = self.THRESH_DENSE - max_density
            time_to_critical = round(remaining / (trend_rate * 10 + 0.01), 1)
            if time_to_critical > 600:
                time_to_critical = None

        anomalies = self._detect_anomalies()
        compression_trend = self._compute_compression()
        flow_summary = self._compute_flow(zone_densities)

        self.prev_zone_densities = zone_densities[:]
        self._last_risk = risk_score
        self._last_status = operational_status

        return {
            "fps": round(self.fps_value, 1),
            "mode": "crowd",
            "crowd_count": count,
            "crowd_trend": trend,
            "crowd_trend_rate": trend_rate,
            "crowd_density": round(max_density, 1),
            "avg_density": round(avg_density, 1),
            "density_class": density_class,
            "zones": zones,
            "zone_distribution": zone_dist,
            "hotspots": hotspots,
            "risk_score": risk_score,
            "operational_status": operational_status,
            "time_to_critical": time_to_critical,
            "peak_count": self.peak_count,
            "peak_density": round(self.peak_density, 1),
            "anomalies": anomalies,
            "compression_trend": compression_trend,
            "flow_summary": flow_summary,
            "detection_count": count,
            "congestion_index": risk_score,
            "traffic_density": int(round(max_density)),
            "hot_regions": {
                "active_count": len(hotspots),
                "severity_counts": {
                    "HIGH": sum(1 for h in hotspots if h["severity"] == "HIGH"),
                    "MODERATE": sum(1 for h in hotspots if h["severity"] == "MODERATE"),
                    "LOW": sum(1 for h in hotspots if h["severity"] == "LOW"),
                },
                "regions": hotspots[:8],
            },
        }


UPLOAD_INFERENCE_SIZE = int(os.environ.get("IRIS_UPLOAD_INFERENCE_SIZE", "640"))
UPLOAD_MAX_DET = int(os.environ.get("IRIS_UPLOAD_MAX_DET", "35"))
INFERENCE_SIZE = int(os.environ.get("IRIS_INFERENCE_SIZE", "640"))
MAX_DET = int(os.environ.get("IRIS_MAX_DET", "35"))

_COLORMAP_BY_NAME = {
    "JET": cv2.COLORMAP_JET,
    "TURBO": cv2.COLORMAP_TURBO,
    "VIRIDIS": cv2.COLORMAP_VIRIDIS,
}


def _resolve_colormap(env_key: str, default_name: str):
    name = os.environ.get(env_key, default_name).strip().upper()
    return _COLORMAP_BY_NAME.get(name, _COLORMAP_BY_NAME[default_name])


class CrowdHeatmapRenderer:
    """Renders continuous crowd density heatmaps from head detections.

    Uses very wide Gaussian kernels so individual points merge into a
    smooth, continuous gradient field.  Colour is a warm ramp
    (transparent → yellow → orange → red) — only visible where people
    are detected.
    """

    def __init__(self, accumulate_frames=10):
        self.accumulate_frames = accumulate_frames
        self.density_accumulator = None
        self.frame_count = 0

    def update(self, tracked_detections):
        if tracked_detections is None or tracked_detections.tracker_id is None:
            return
        xyxys = tracked_detections.xyxy
        tids = tracked_detections.tracker_id
        if len(xyxys) == 0:
            return
        h, w = 480, 640
        if self.density_accumulator is None:
            self.density_accumulator = np.zeros((h, w), dtype=np.float32)
        for i in range(len(tids)):
            x1, y1, x2, y2 = xyxys[i]
            cx = int((x1 + x2) * 0.5)
            cy = int((y1 + y2) * 0.5)
            # Very wide sigma so detections merge into a continuous field
            sigma = max(50, int(min(x2 - x1, y2 - y1) * 2.0))
            y_grid, x_grid = np.ogrid[:h, :w]
            dist_sq = (x_grid - cx) ** 2 + (y_grid - cy) ** 2
            gaussian = np.exp(-dist_sq / (2 * (sigma ** 2)))
            self.density_accumulator += gaussian * 2.0
        self.frame_count += 1

    def render(self, frame):
        if self.density_accumulator is None:
            return frame
        h, w = frame.shape[:2]
        if self.density_accumulator.max() <= 0:
            return frame
        nonzero_vals = self.density_accumulator[self.density_accumulator > 0]
        if len(nonzero_vals) == 0:
            return frame
        p95 = np.percentile(nonzero_vals, 95)
        max_val = max(self.density_accumulator.max(), p95, 1e-6)
        density_norm = np.clip(self.density_accumulator / max_val, 0, 1)
        if density_norm.shape != (h, w):
            density_norm = cv2.resize(density_norm, (w, h), interpolation=cv2.INTER_LINEAR)
        # Heavy blur for smooth continuous gradient
        density_blurred = cv2.GaussianBlur(density_norm, (61, 61), 0)
        density_blurred = np.power(density_blurred, 0.70)

        # irisv3 style: JET colormap + simple addWeighted 50/50
        from matplotlib import pyplot as _plt
        _norm = _plt.Normalize(vmin=float(density_blurred.min()), vmax=float(density_blurred.max()) + 1e-8)
        _cmap = _plt.get_cmap('jet')
        _rgba = _cmap(_norm(density_blurred))
        _rgb = np.delete(_rgba, 3, 2)
        heatmap_colored = cv2.cvtColor((_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        result = cv2.addWeighted(frame, 0.5, heatmap_colored, 0.5, 0)

        self.density_accumulator *= 0.92
        if self.frame_count >= self.accumulate_frames:
            self.density_accumulator *= 0.5
            self.frame_count = 0
        return result


def inference_profile(is_upload: bool) -> tuple[int, int]:
    if is_upload:
        return UPLOAD_INFERENCE_SIZE, UPLOAD_MAX_DET
    return INFERENCE_SIZE, MAX_DET


def test_model():
    """Test the model loads correctly."""
    import os
    model_path = "models/crowd-model.pth"
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    counter = CrowdCounter(model_path, device)

    dummy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result = counter.count(dummy)
    print(f"Test count: {result['count']}")
    print(f"Density map shape: {result['density_map'].shape}")
    print(f"Heatmap shape: {result['heatmap'].shape}")


if __name__ == "__main__":
    test_model()
