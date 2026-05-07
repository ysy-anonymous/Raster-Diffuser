from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Optional

import copy
import numpy as np

from utils.dataset_utils import (
    random_rectangles,
    sample_start_goal,
    rectangles_to_grid,
    in_collision,
    validate_path_collision_free,
)
from data_generator_d.RRT_star_grid import RRTStarGrid


@dataclass
class DynamicSample2D:
    """One dynamic navigation instance."""

    maps: np.ndarray          # (T-1, 1, H, W)
    start_goal: np.ndarray    # (T-1, 4), [sx, sy, gx, gy]
    paths: np.ndarray         # (T-1, T, 2)
    rects_seq: np.ndarray     # optional: (T-1, num_rects, 4)
    bounds: np.ndarray
    cell_size: float

    def to_dict(self):
        return asdict(self)


    def __init__(
        self,
        bounds: List[Tuple[float, float]] | np.ndarray,
        num_samples: int,
        *,
        resolution: float = 1.0,
        max_rectangles: Tuple[int, int] = (2, 6),
        step_size: float = 0.5,
        max_iter_rrt: int = 2000,
        goal_tol: float = 0.3,
        horizon_length: int = 32,
        obstacle_speed: float = 0.5,
        direction_change_prob: float = 0.15,
        max_attempts_factor: int = 300,
        rng: Optional[int | np.random.Generator] = None,
    ) -> None:
        self.bounds = np.asarray(bounds, dtype=float)
        self.num_samples = int(num_samples)
        self.resolution = float(resolution)
        self.max_rectangles = max_rectangles
        self.step_size = step_size
        self.max_iter_rrt = max_iter_rrt
        self.goal_tol = goal_tol
        self.horizon_length = int(horizon_length)

        self.obstacle_speed = float(obstacle_speed)
        self.direction_change_prob = float(direction_change_prob)
        self.max_attempts_factor = int(max_attempts_factor)

        self.rng = np.random.default_rng(rng)
        self.origin = self.bounds[:, 0]

        self.nx, self.ny = (
            int((hi - lo) / self.resolution) for lo, hi in self.bounds
        )

        self.training_data_set = {
            "map": [],          # each: (T-1, 1, H, W)
            "start_goal": [],   # each: (T-1, 4)
            "paths": [],        # each: (T-1, T, 2)
            "rects_seq": [],    # optional debug/visualization
        }

    # ---------------------------------------------------------------------
    # Basic helpers
    # ---------------------------------------------------------------------

    def _point_free(self, point: np.ndarray, grid: np.ndarray) -> bool:
        return not in_collision(
            np.asarray(point, dtype=float),
            grid,
            self.resolution,
            self.origin,
        )

    def _resample_path_exact_T(self, path: np.ndarray) -> np.ndarray:
        """
        Force any RRT/B-spline path to exactly (T, 2).
        This protects your dataset shape even if the planner returns more/fewer points.
        """
        path = np.asarray(path, dtype=float)

        if path.shape[0] == self.horizon_length:
            return path

        if path.shape[0] < 2:
            raise ValueError("Path must contain at least two points.")

        seg_lens = np.linalg.norm(np.diff(path, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
        total = cum[-1]

        if total < 1e-8:
            return np.repeat(path[:1], self.horizon_length, axis=0)

        target = np.linspace(0.0, total, self.horizon_length)
        x = np.interp(target, cum, path[:, 0])
        y = np.interp(target, cum, path[:, 1])
        return np.stack([x, y], axis=1)

    def _sample_velocity(self) -> np.ndarray:
        """
        Sample one obstacle velocity.
        Includes 8-neighborhood directions; zero velocity is also allowed.
        """
        dirs = np.array(
            [
                [0, 0],
                [1, 0],
                [-1, 0],
                [0, 1],
                [0, -1],
                [1, 1],
                [1, -1],
                [-1, 1],
                [-1, -1],
            ],
            dtype=float,
        )

        d = dirs[self.rng.integers(0, len(dirs))]
        norm = np.linalg.norm(d)

        if norm > 0:
            d = d / norm

        return d * self.obstacle_speed

    def _advance_rectangles(
        self,
        rects: np.ndarray,
        velocities: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Move axis-aligned rectangular obstacles and bounce them at workspace bounds.

        rects:      (N, 4), each [x_min, y_min, w, h]
        velocities: (N, 2), each [vx, vy]
        """
        next_rects = rects.copy()
        next_vel = velocities.copy()

        x_lo, x_hi = self.bounds[0]
        y_lo, y_hi = self.bounds[1]

        for i in range(next_rects.shape[0]):
            # Randomly change direction over the planning horizon.
            if self.rng.random() < self.direction_change_prob:
                next_vel[i] = self._sample_velocity()

            x, y, w, h = next_rects[i]
            vx, vy = next_vel[i]

            x_new = x + vx
            y_new = y + vy

            # Bounce in x.
            if x_new < x_lo:
                x_new = x_lo
                next_vel[i, 0] *= -1.0
            elif x_new + w > x_hi:
                x_new = x_hi - w
                next_vel[i, 0] *= -1.0

            # Bounce in y.
            if y_new < y_lo:
                y_new = y_lo
                next_vel[i, 1] *= -1.0
            elif y_new + h > y_hi:
                y_new = y_hi - h
                next_vel[i, 1] *= -1.0

            next_rects[i, 0] = x_new
            next_rects[i, 1] = y_new

        return next_rects, next_vel

    def _generate_dynamic_maps(
        self,
        rects0: List[Tuple[float, float, float, float]],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate a dynamic obstacle-map sequence.

        Returns:
            maps:      (T-1, H, W), bool
            rects_seq: (T-1, N, 4)
        """
        num_steps = self.horizon_length - 1

        rects = np.asarray(rects0, dtype=float)
        velocities = np.stack(
            [self._sample_velocity() for _ in range(rects.shape[0])],
            axis=0,
        )

        maps = []
        rects_seq = []

        for _ in range(num_steps):
            grid = rectangles_to_grid(
                self.nx,
                self.ny,
                self.bounds,
                self.resolution,
                [tuple(r) for r in rects],
            )

            maps.append(grid.astype(bool))
            rects_seq.append(rects.copy())

            rects, velocities = self._advance_rectangles(rects, velocities)

        return np.stack(maps, axis=0), np.stack(rects_seq, axis=0)

    def _sample_goal_free_for_all_maps(self, maps: np.ndarray) -> np.ndarray | None:
        """
        Sample a fixed goal that is not occupied by any obstacle map.
        """
        for _ in range(1000):
            point = self.rng.uniform(self.bounds[:, 0], self.bounds[:, 1])
            if all(self._point_free(point, maps[t]) for t in range(maps.shape[0])):
                return point
        return None

    # ---------------------------------------------------------------------
    # Main generation
    # ---------------------------------------------------------------------

    def _try_generate_one(self, smooth: bool, interp: int) -> DynamicSample2D | None:
        T = self.horizon_length
        num_steps = T - 1

        # 1. Initial static obstacle layout.
        rects0 = random_rectangles(self.max_rectangles, self.bounds, self.rng)

        # 2. Roll out moving obstacles.
        maps_raw, rects_seq = self._generate_dynamic_maps(rects0)
        # maps_raw shape: (T-1, H, W)

        # 3. Sample start on first map.
        # You can use your existing sample_start_goal for the initial pair,
        # then replace goal with one that stays free across all maps.
        start, _ = sample_start_goal(
            self.bounds,
            maps_raw[0],
            self.resolution,
            self.origin,
            self.rng,
        )

        goal = self._sample_goal_free_for_all_maps(maps_raw)
        if goal is None:
            return None

        if not self._point_free(start, maps_raw[0]):
            return None

        maps_out = np.zeros((num_steps, 1, self.nx, self.ny), dtype=np.float32)
        start_goal_out = np.zeros((num_steps, 4), dtype=np.float32)
        paths_out = np.zeros((num_steps, T, 2), dtype=np.float32)

        current_start = np.asarray(start, dtype=float)

        # 4. Replan at each dynamic time step.
        for k in range(num_steps):
            grid_k = maps_raw[k]

            # Start and goal must not overlap with current obstacles.
            if not self._point_free(current_start, grid_k):
                return None
            if not self._point_free(goal, grid_k):
                return None

            planner = RRTStarGrid(
                self.bounds,
                grid_k,
                self.resolution,
                collect_training_data=False,
                max_iter=self.max_iter_rrt,
                step_size=self.step_size,
                goal_tol=self.goal_tol,
                rng=self.rng,
                min_points=T,
            )

            path = planner.plan(
                current_start,
                goal,
                prune=True,
                optimize=smooth,
                interp_points=interp,
            )

            if path is None:
                return None

            path = self._resample_path_exact_T(path)

            # Safety check for the current time slice.
            if not validate_path_collision_free(
                path,
                grid_k,
                self.resolution,
                self.origin,
            ):
                return None

            maps_out[k, 0] = grid_k.astype(np.float32)
            start_goal_out[k] = np.array(
                [current_start[0], current_start[1], goal[0], goal[1]],
                dtype=np.float32,
            )
            paths_out[k] = path.astype(np.float32)

            # Dataset/expert rollout update.
            # During model evaluation, replace this with pred_path[1].
            current_start = path[1].copy()

        return DynamicSample2D(
            maps=maps_out,
            start_goal=start_goal_out,
            paths=paths_out,
            rects_seq=rects_seq.astype(np.float32),
            bounds=self.bounds,
            cell_size=self.resolution,
        )

    def generate_dataset(self, smooth: bool = True, interp: int = 100):
        samples: List[DynamicSample2D] = []
        attempts = 0
        max_attempts = self.num_samples * self.max_attempts_factor

        while len(samples) < self.num_samples and attempts < max_attempts:
            attempts += 1
            print(
                f"Generating dynamic sample {len(samples) + 1}/{self.num_samples} "
                f"(attempts: {attempts})"
            )

            sample = self._try_generate_one(smooth=smooth, interp=interp)

            if sample is None:
                continue

            samples.append(sample)

            self.training_data_set["map"].append(copy.deepcopy(sample.maps))
            self.training_data_set["start_goal"].append(copy.deepcopy(sample.start_goal))
            self.training_data_set["paths"].append(copy.deepcopy(sample.paths))
            self.training_data_set["rects_seq"].append(copy.deepcopy(sample.rects_seq))

        if len(samples) < self.num_samples:
            raise RuntimeError(
                f"Could not create {self.num_samples} samples after {attempts} attempts."
            )

        # Convert list-of-arrays into batch arrays:
        # map:        (N, T-1, 1, H, W)
        # start_goal: (N, T-1, 4)
        # paths:      (N, T-1, T, 2)
        self.training_data_set["map"] = np.stack(self.training_data_set["map"], axis=0)
        self.training_data_set["start_goal"] = np.stack(
            self.training_data_set["start_goal"],
            axis=0,
        )
        self.training_data_set["paths"] = np.stack(self.training_data_set["paths"], axis=0)
        self.training_data_set["rects_seq"] = np.stack(
            self.training_data_set["rects_seq"],
            axis=0,
        )

        return samples, self.training_data_set

    def save_train_data(self, out_file: str | Path):
        np.save(out_file, self.training_data_set)

    def save_npz(self, samples: List[DynamicSample2D], outfile: str | Path) -> None:
        arr = {f"sample_{i}": s.to_dict() for i, s in enumerate(samples)}
        np.savez_compressed(outfile, **arr)