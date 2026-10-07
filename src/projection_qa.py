"""LiDAR-Camera Projection QA & Calibration Sensitivity Benchmarking Suite.

Topic A: LiDAR-camera projection QA
Bài lab Day 6 - Track 4 Computer Vision & Robotics
Tác giả: Đặng Quốc Hiệp (MSSV: 2A202602755)

Script này cung cấp:
1. Thí nghiệm sweep góc lệch Extrinsic (Yaw, Pitch, Roll) và Translation (tx, ty, tz).
2. Đo lường các metric:
   - Inside-FOV point retention ratio (%)
   - Object 2D Bounding Box point retention ratio (%) theo cự ly (Near <15m, Mid 15-30m, Far >=30m)
   - Mean reprojection pixel shift (Delta_u, Delta_v in pixels)
   - Projected point-cloud bounding box IoU so với 2D GT box
3. Đo latency chuẩn p50/p95 (30 lần lặp, bỏ lần đầu warm-up).
4. So sánh trên cả 2 dataset thật: KITTI (64-beam) vs nuScenes (32-beam, ảnh hưởng bù ego-motion).
5. Tự động kiểm tra phát hiện lỗi cài sẵn trong data/synthetic.
6. Xuất toàn bộ kết quả CSV vào results/ và đồ thị/ảnh minh hoạ vào results/figures/.
"""
from __future__ import annotations

import argparse
import copy
import os
import platform
import sys
import time
from pathlib import Path

# Đảm bảo in tiếng Việt trên console Windows không lỗi charmap
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from starter.datasets import dataset_type, load_frame
from starter.kitti_io import KittiCalib, KittiObject
from starter.projection import (
    cam_to_image,
    draw_box2d,
    overlay_points,
    perturb_extrinsic,
    project_velo_to_image,
    velo_to_cam,
)


def calculate_bbox_iou(box1: list[float] | np.ndarray, box2: list[float] | np.ndarray) -> float:
    """Tính Intersection-over-Union (IoU) giữa 2 bounding box [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def extract_object_lidar_indices(
    pts: np.ndarray, labels: list[KittiObject], calib: KittiCalib, image_shape: tuple[int, ...]
) -> dict[int, np.ndarray]:
    """Liên kết điểm LiDAR với từng object 2D/3D ground-truth trong điều kiện calibration chuẩn."""
    H, W = image_shape[:2]
    pts_cam = velo_to_cam(pts[:, :3], calib)
    uv, depth, mask = cam_to_image(pts_cam, calib.P2, (H, W))

    obj_indices: dict[int, np.ndarray] = {}
    valid_orig_indices = np.where(mask)[0]

    for i, obj in enumerate(labels):
        x1, y1, x2, y2 = obj.bbox
        # Chiều sâu vật thể (location z +- chiều dài/rộng/khoảng đệm an toàn)
        z_min = obj.location[2] - max(obj.dimensions[2], obj.dimensions[1]) / 2.0 - 0.8
        z_max = obj.location[2] + max(obj.dimensions[2], obj.dimensions[1]) / 2.0 + 0.8

        in_box = (uv[:, 0] >= x1) & (uv[:, 0] <= x2) & (uv[:, 1] >= y1) & (uv[:, 1] <= y2)
        in_depth = (depth >= z_min) & (depth <= z_max)

        matched = valid_orig_indices[in_box & in_depth]
        if len(matched) >= 5:  # Chỉ xét object có ít nhất 5 điểm LiDAR hợp lệ
            obj_indices[i] = matched

    return obj_indices


def evaluate_perturbation(
    pts: np.ndarray,
    labels: list[KittiObject],
    calib_gt: KittiCalib,
    calib_perturbed: KittiCalib,
    image_shape: tuple[int, ...],
    obj_indices: dict[int, np.ndarray],
) -> dict[str, float]:
    """Đo lường toàn diện ảnh hưởng của một cấu hình perturbed calib so với ground-truth."""
    H, W = image_shape[:2]

    # Chiếu unperturbed
    pts_cam_gt = velo_to_cam(pts[:, :3], calib_gt)
    uv_gt, depth_gt, mask_gt = cam_to_image(pts_cam_gt, calib_gt.P2, (H, W))

    # Chiếu perturbed
    pts_cam_p = velo_to_cam(pts[:, :3], calib_perturbed)
    uv_p, depth_p, mask_p = cam_to_image(pts_cam_p, calib_perturbed.P2, (H, W))

    # Metric 1: Inside image FOV ratio
    fov_ratio = float(mask_p.sum() / len(pts))

    # Metric 2: Reprojection pixel shift trên các điểm hợp lệ chung
    common_mask = mask_gt & mask_p
    if common_mask.sum() > 0:
        # Lấy toạ độ pixel tương ứng của các điểm chung
        cumsum_gt = np.cumsum(mask_gt) - 1
        cumsum_p = np.cumsum(mask_p) - 1
        common_idx = np.where(common_mask)[0]

        sub_uv_gt = uv_gt[cumsum_gt[common_idx]]
        sub_uv_p = uv_p[cumsum_p[common_idx]]

        delta_u = np.abs(sub_uv_p[:, 0] - sub_uv_gt[:, 0])
        delta_v = np.abs(sub_uv_p[:, 1] - sub_uv_gt[:, 1])
        delta_px = np.sqrt(delta_u**2 + delta_v**2)

        mean_delta_u = float(np.mean(delta_u))
        mean_delta_v = float(np.mean(delta_v))
        mean_delta_px = float(np.mean(delta_px))
    else:
        mean_delta_u = mean_delta_v = mean_delta_px = 0.0

    # Metric 3: Object 2D Bounding Box Point Retention Ratio theo khoảng cách
    retentions_near = []
    retentions_mid = []
    retentions_far = []
    retentions_all = []
    ious_all = []

    for obj_idx, orig_pt_ids in obj_indices.items():
        obj = labels[obj_idx]
        x1, y1, x2, y2 = obj.bbox
        dist_z = obj.location[2]

        is_valid = mask_p[orig_pt_ids]
        if is_valid.sum() == 0:
            ret = 0.0
            iou = 0.0
        else:
            cumsum_p = np.cumsum(mask_p) - 1
            valid_pt_ids = orig_pt_ids[is_valid]
            sub_uv = uv_p[cumsum_p[valid_pt_ids]]

            in_box = (
                (sub_uv[:, 0] >= x1)
                & (sub_uv[:, 0] <= x2)
                & (sub_uv[:, 1] >= y1)
                & (sub_uv[:, 1] <= y2)
            )
            ret = float(in_box.sum() / len(orig_pt_ids))

            p_box = [
                float(sub_uv[:, 0].min()),
                float(sub_uv[:, 1].min()),
                float(sub_uv[:, 0].max()),
                float(sub_uv[:, 1].max()),
            ]
            iou = calculate_bbox_iou(p_box, obj.bbox)

        retentions_all.append(ret)
        ious_all.append(iou)

        if dist_z < 15.0:
            retentions_near.append(ret)
        elif dist_z < 30.0:
            retentions_mid.append(ret)
        else:
            retentions_far.append(ret)

    return {
        "fov_ratio": fov_ratio,
        "mean_delta_u_px": mean_delta_u,
        "mean_delta_v_px": mean_delta_v,
        "mean_delta_px": mean_delta_px,
        "retention_all": float(np.mean(retentions_all)) if retentions_all else 0.0,
        "retention_near": float(np.mean(retentions_near)) if retentions_near else 0.0,
        "retention_mid": float(np.mean(retentions_mid)) if retentions_mid else 0.0,
        "retention_far": float(np.mean(retentions_far)) if retentions_far else 0.0,
        "mean_bbox_iou": float(np.mean(ious_all)) if ious_all else 0.0,
    }


def compute_edge_alignment_score(
    image: np.ndarray, uv: np.ndarray, mask: np.ndarray, obj_indices: dict[int, np.ndarray]
) -> float:
    """Tính alignment score giữa điểm LiDAR của đối tượng và contour Canny edge trên ảnh."""
    H, W = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    inv_edges = (edges == 0).astype(np.uint8)
    dist_map = cv2.distanceTransform(inv_edges, cv2.DIST_L2, 3)

    # Thu thập toạ độ các điểm đối tượng
    all_obj_pts = []
    for orig_ids in obj_indices.values():
        is_valid = mask[orig_ids]
        if is_valid.sum() > 0:
            cumsum = np.cumsum(mask) - 1
            all_obj_pts.append(uv[cumsum[orig_ids[is_valid]]])

    if not all_obj_pts:
        return 0.0

    pts_concat = np.concatenate(all_obj_pts, axis=0)
    u_coords = np.clip(pts_concat[:, 0].astype(int), 0, W - 1)
    v_coords = np.clip(pts_concat[:, 1].astype(int), 0, H - 1)

    dists = dist_map[v_coords, u_coords]
    # Hàm decay điểm chuẩn hoá với sigma = 4.0 pixel
    score = float(np.mean(np.exp(-((dists / 4.0) ** 2))))
    return score


def run_yaw_sweep(
    data_root: str,
    frame_id: str,
    out_dir: Path,
    yaw_levels: list[float] | None = None,
) -> pd.DataFrame:
    """Thí nghiệm 1: Sweep góc lệch Yaw từ -3.0 deg đến +3.0 deg."""
    if yaw_levels is None:
        yaw_levels = [-3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0]

    fr = load_frame(data_root, frame_id)
    pts = fr["points"]
    calib = fr["calib"]
    labels = fr["labels"]
    img = fr["image"]

    obj_indices = extract_object_lidar_indices(pts, labels, calib, img.shape)
    print(f"[Yaw Sweep] Khung hình {frame_id}: Tìm thấy {len(obj_indices)} object có đủ điểm LiDAR.")

    records = []
    for yaw in yaw_levels:
        cal_p = perturb_extrinsic(calib, yaw_deg=yaw)
        metrics = evaluate_perturbation(pts, labels, calib, cal_p, img.shape, obj_indices)

        # Tính edge alignment score
        pts_c = velo_to_cam(pts[:, :3], cal_p)
        uv_p, _, mask_p = cam_to_image(pts_c, cal_p.P2, img.shape)
        edge_score = compute_edge_alignment_score(img, uv_p, mask_p, obj_indices)

        rec = {
            "yaw_deg": yaw,
            "fov_inside_ratio": metrics["fov_ratio"],
            "mean_delta_u_px": metrics["mean_delta_u_px"],
            "mean_delta_px": metrics["mean_delta_px"],
            "retention_overall": metrics["retention_all"],
            "retention_near_lt15m": metrics["retention_near"],
            "retention_mid_15to30m": metrics["retention_mid"],
            "retention_far_gte30m": metrics["retention_far"],
            "point_bbox_iou": metrics["mean_bbox_iou"],
            "edge_alignment_score": edge_score,
        }
        records.append(rec)

    df = pd.DataFrame(records)
    csv_path = out_dir / "yaw_perturb_sweep.csv"
    df.to_csv(csv_path, index=False)
    print(f"[PASS] Đã lưu kết quả yaw sweep -> {csv_path}")

    # Vẽ biểu đồ trực quan hoá
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    # Subplot 1: Retention vs Yaw by Distance
    ax1.plot(df["yaw_deg"], df["retention_near_lt15m"] * 100, "o-", label="Gần (<15m)", color="#2ca02c", lw=2)
    ax1.plot(df["yaw_deg"], df["retention_mid_15to30m"] * 100, "s-", label="Trung bình (15-30m)", color="#ff7f0e", lw=2)
    ax1.plot(df["yaw_deg"], df["retention_far_gte30m"] * 100, "^-", label="Xa (>=30m)", color="#d62728", lw=2)
    ax1.plot(df["yaw_deg"], df["retention_overall"] * 100, "--", label="Toàn bộ vật thể", color="#1f77b4", lw=2)
    ax1.axvline(0, color="gray", linestyle=":", alpha=0.7)
    ax1.axhline(50, color="red", linestyle=":", alpha=0.5, label="Ngưỡng mất mát 50%")
    ax1.set_xlabel("Độ lệch góc Yaw (độ)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Tỷ lệ điểm rơi trong 2D Bbox (%)", fontsize=11, fontweight="bold")
    ax1.set_title("Độ nhạy lệch Yaw theo cự ly vật thể", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="lower center", fontsize=9)

    # Subplot 2: Edge alignment score & Reprojection Shift
    ax2.plot(df["yaw_deg"], df["point_bbox_iou"], "o-", label="Point-Cluster Bbox IoU", color="#9467bd", lw=2)
    ax2.plot(df["yaw_deg"], df["edge_alignment_score"], "s-", label="Canny Edge Alignment Score", color="#8c564b", lw=2)
    ax2_twin = ax2.twinx()
    ax2_twin.plot(df["yaw_deg"], df["mean_delta_px"], "x--", label="Độ dịch pixel tb (px)", color="#e377c2", lw=1.5)
    ax2_twin.set_ylabel("Độ dịch chuyển trung bình (pixel)", color="#e377c2", fontsize=11)

    ax2.set_xlabel("Độ lệch góc Yaw (độ)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Chỉ số tương đồng / IoU (0 - 1)", fontsize=11, fontweight="bold")
    ax2.set_title("Chỉ số phát hiện Calibration Drift", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="lower left", fontsize=9)

    fig.tight_layout()
    plot_path = out_dir / "figures" / "yaw_sweep_metric.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path)
    plt.close()
    print(f"[PASS] Đã lưu đồ thị yaw sweep -> {plot_path}")

    return df


def run_translation_sweep(
    data_root: str,
    frame_id: str,
    out_dir: Path,
    ty_levels: list[float] | None = None,
) -> pd.DataFrame:
    """Thí nghiệm 2: Sweep dịch chuyển ngang lateral translation ty từ -0.15m đến +0.15m."""
    if ty_levels is None:
        ty_levels = [-0.15, -0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10, 0.15]

    fr = load_frame(data_root, frame_id)
    pts = fr["points"]
    calib = fr["calib"]
    labels = fr["labels"]
    img = fr["image"]

    obj_indices = extract_object_lidar_indices(pts, labels, calib, img.shape)
    records = []
    for ty in ty_levels:
        cal_p = perturb_extrinsic(calib, t_xyz_m=(0.0, ty, 0.0))
        metrics = evaluate_perturbation(pts, labels, calib, cal_p, img.shape, obj_indices)
        rec = {
            "translation_y_m": ty,
            "fov_inside_ratio": metrics["fov_ratio"],
            "mean_delta_px": metrics["mean_delta_px"],
            "retention_overall": metrics["retention_all"],
            "retention_near": metrics["retention_near"],
            "retention_far": metrics["retention_far"],
            "point_bbox_iou": metrics["mean_bbox_iou"],
        }
        records.append(rec)

    df = pd.DataFrame(records)
    csv_path = out_dir / "translation_sweep.csv"
    df.to_csv(csv_path, index=False)
    print(f"[PASS] Đã lưu kết quả translation sweep -> {csv_path}")

    # Vẽ biểu đồ so sánh Translation
    plt.figure(figsize=(8, 5), dpi=150)
    plt.plot(df["translation_y_m"] * 100, df["retention_near"] * 100, "o-", label="Gần (<15m)", color="#2ca02c", lw=2)
    plt.plot(df["translation_y_m"] * 100, df["retention_far"] * 100, "^-", label="Xa (>=15m)", color="#d62728", lw=2)
    plt.plot(df["translation_y_m"] * 100, df["retention_overall"] * 100, "--", label="Toàn bộ", color="#1f77b4", lw=2)
    plt.xlabel("Độ dịch chuyển ngang ty (cm)", fontsize=11, fontweight="bold")
    plt.ylabel("Tỷ lệ điểm rơi trong 2D Bbox (%)", fontsize=11, fontweight="bold")
    plt.title("Ảnh hưởng của dịch chuyển tịnh tiến sensor (Translation drift)", fontsize=12, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower center", fontsize=9)
    plt.tight_layout()

    plot_path = out_dir / "figures" / "translation_sweep.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"[PASS] Đã lưu đồ thị translation sweep -> {plot_path}")

    return df


def run_latency_benchmark(data_root: str, frame_id: str, out_dir: Path, n_runs: int = 30) -> pd.DataFrame:
    """Bonus B3: Đo benchmark latency p50/p95 loại bỏ lần warm-up đầu."""
    fr = load_frame(data_root, frame_id)
    pts = fr["points"]
    calib = fr["calib"]
    shape = fr["image"].shape

    # Warmup 5 runs
    for _ in range(5):
        pts_cam = velo_to_cam(pts[:, :3], calib)
        cam_to_image(pts_cam, calib.P2, shape)

    records = []
    for i in range(n_runs):
        t0 = time.perf_counter()
        pts_cam = velo_to_cam(pts[:, :3], calib)
        t1 = time.perf_counter()
        uv, depth, mask = cam_to_image(pts_cam, calib.P2, shape)
        t2 = time.perf_counter()

        velo2cam_ms = (t1 - t0) * 1000.0
        cam2img_ms = (t2 - t1) * 1000.0
        total_ms = (t2 - t0) * 1000.0

        records.append({
            "run_id": i + 1,
            "points_count": len(pts),
            "velo_to_cam_ms": velo2cam_ms,
            "cam_to_image_ms": cam2img_ms,
            "total_projection_ms": total_ms,
        })

    df = pd.DataFrame(records)
    csv_path = out_dir / "latency_benchmark.csv"
    df.to_csv(csv_path, index=False)

    p50 = float(df["total_projection_ms"].quantile(0.50))
    p95 = float(df["total_projection_ms"].quantile(0.95))
    mean = float(df["total_projection_ms"].mean())
    print(f"[PASS] Benchmark Latency ({n_runs} runs): Mean={mean:.2f}ms, p50={p50:.2f}ms, p95={p95:.2f}ms -> {csv_path}")
    return df


def run_kitti_vs_nuscenes_comparison(
    kitti_root: str,
    kitti_frame: str,
    nuscenes_root: str,
    nuscenes_frame: str,
    out_dir: Path,
) -> pd.DataFrame:
    """Bonus B5: So sánh thí nghiệm trên cả 2 dataset thật KITTI (64 beams) và nuScenes (32 beams)."""
    fr_k = load_frame(kitti_root, kitti_frame)
    fr_n_ego = load_frame(nuscenes_root, nuscenes_frame, use_ego_motion=True)
    fr_n_noego = load_frame(nuscenes_root, nuscenes_frame, use_ego_motion=False)

    # KITTI stats
    uv_k, d_k, m_k = project_velo_to_image(fr_k["points"], fr_k["calib"], fr_k["image"].shape)
    # nuScenes stats with ego compensation
    uv_ne, d_ne, m_ne = project_velo_to_image(fr_n_ego["points"], fr_n_ego["calib"], fr_n_ego["image"].shape)
    # nuScenes stats without ego compensation
    uv_nne, d_nne, m_nne = project_velo_to_image(fr_n_noego["points"], fr_n_noego["calib"], fr_n_noego["image"].shape)

    # Đo độ lệch pixel do ego-motion gây ra trên nuScenes
    common_pts = m_ne & m_nne
    if common_pts.sum() > 0:
        c_ne = np.cumsum(m_ne) - 1
        c_nne = np.cumsum(m_nne) - 1
        common_ids = np.where(common_pts)[0]
        pts_ne = uv_ne[c_ne[common_ids]]
        pts_nne = uv_nne[c_nne[common_ids]]
        ego_motion_drift_px = float(np.mean(np.linalg.norm(pts_ne - pts_nne, axis=1)))
    else:
        ego_motion_drift_px = 0.0

    comparison = [
        {
            "dataset": "KITTI (64-beam HDL-64E)",
            "frame": kitti_frame,
            "total_lidar_points": len(fr_k["points"]),
            "image_resolution": f"{fr_k['image'].shape[1]}x{fr_k['image'].shape[0]}",
            "points_inside_image": int(m_k.sum()),
            "fov_coverage_ratio": float(m_k.mean()),
            "ego_motion_drift_px": 0.0,
            "sensor_sync_method": "Pre-rectified / Hardware trigger",
        },
        {
            "dataset": "nuScenes (32-beam)",
            "frame": nuscenes_frame,
            "total_lidar_points": len(fr_n_ego["points"]),
            "image_resolution": f"{fr_n_ego['image'].shape[1]}x{fr_n_ego['image'].shape[0]}",
            "points_inside_image": int(m_ne.sum()),
            "fov_coverage_ratio": float(m_ne.mean()),
            "ego_motion_drift_px": ego_motion_drift_px,
            "sensor_sync_method": "Asynchronous timestamps (Ego-motion deskew)",
        },
    ]

    df = pd.DataFrame(comparison)
    csv_path = out_dir / "kitti_vs_nuscenes.csv"
    df.to_csv(csv_path, index=False)
    print(f"[PASS] Đã lưu kết quả so sánh KITTI vs nuScenes -> {csv_path}")

    # Tạo ảnh demo chuẩn cho cả 2
    vis_k = overlay_points(fr_k["image"], uv_k, d_k)
    for obj in fr_k["labels"]:
        vis_k = draw_box2d(vis_k, obj.bbox, label=obj.type)
    kitti_demo_path = out_dir / "figures" / "demo_kitti_000011_overlay.png"
    cv2.imwrite(str(kitti_demo_path), vis_k)

    vis_n = overlay_points(fr_n_ego["image"], uv_ne, d_ne)
    for obj in fr_n_ego["labels"]:
        vis_n = draw_box2d(vis_n, obj.bbox, label=obj.type)
    nuscenes_demo_path = out_dir / "figures" / "demo_nuscenes_overlay.png"
    cv2.imwrite(str(nuscenes_demo_path), vis_n)

    print(f"[PASS] Đã tạo demo ảnh: {kitti_demo_path} và {nuscenes_demo_path}")
    return df


def run_synthetic_health_check(synthetic_root: str, out_dir: Path) -> pd.DataFrame:
    """Bonus B6: Phát hiện các lỗi cài sẵn trong data/synthetic."""
    frames = ["000000", "000001", "000002", "000003", "000004"]
    records = []

    ts_file = Path(synthetic_root) / "training" / "timestamps.txt"
    timestamps = [float(x.strip()) for x in ts_file.read_text().splitlines() if x.strip()] if ts_file.exists() else []

    prev_ts = None
    for idx, fid in enumerate(frames):
        fr = load_frame(synthetic_root, fid)
        pts = fr["points"]
        nan_cnt = int(np.isnan(pts).any(axis=1).sum())
        inf_cnt = int(np.isinf(pts).any(axis=1).sum())
        total = len(pts)

        ts = timestamps[idx] if idx < len(timestamps) else None
        dt = (ts - prev_ts) if (ts is not None and prev_ts is not None) else 0.1
        prev_ts = ts

        # Phát hiện lỗi
        defects = []
        if nan_cnt > 0 or inf_cnt > 0:
            defects.append(f"Chứa {nan_cnt + inf_cnt} điểm NaN/Inf ({((nan_cnt + inf_cnt)/total):.2%})")
        if abs(dt - 0.1) > 0.05:
            defects.append(f"Mất đồng bộ thời gian / Drop frame (dt = {dt:.2f}s thay vì 0.10s)")
        if total < 23000:
            defects.append(f"Suy giảm mật độ chùm quét đột ngột ({total} điểm so với ~23800 điểm)")

        records.append({
            "frame_id": fid,
            "timestamp_s": ts,
            "delta_t_s": dt,
            "total_points": total,
            "nan_inf_points": nan_cnt + inf_cnt,
            "defects_detected": "; ".join(defects) if defects else "Bình thường",
        })

    df = pd.DataFrame(records)
    csv_path = out_dir / "synthetic_defects.csv"
    df.to_csv(csv_path, index=False)
    print(f"[PASS] Đã lưu kết quả phân tích lỗi synthetic -> {csv_path}")
    return df


def generate_failure_cases(kitti_root: str, nuscenes_root: str, out_dir: Path) -> None:
    """Tạo ảnh failure case trực quan minh hoạ cho CP4."""
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Failure Case 1: Severe Yaw Drift (+2.0 deg) trên KITTI frame 000011
    fr_k = load_frame(kitti_root, "000011")
    cal_fail = perturb_extrinsic(fr_k["calib"], yaw_deg=2.0)
    uv_f, d_f, _ = project_velo_to_image(fr_k["points"], cal_fail, fr_k["image"].shape)
    vis_fail1 = overlay_points(fr_k["image"], uv_f, d_f)
    for obj in fr_k["labels"]:
        vis_fail1 = draw_box2d(vis_fail1, obj.bbox, label=f"{obj.type} (GT box)")

    # Thêm text cảnh báo lên ảnh
    cv2.putText(
        vis_fail1,
        "FAILURE: Extrinsic Yaw Drift +2.0 deg -> Geometry Debug Layer",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
    )
    cv2.putText(
        vis_fail1,
        "Points shifted laterally ~25px; Far pedestrian & car points spill out of 2D box",
        (30, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )
    fail_path1 = fig_dir / "fail_01_yaw_2deg_drift.png"
    cv2.imwrite(str(fail_path1), vis_fail1)
    print(f"[PASS] Đã tạo ảnh failure case 1 -> {fail_path1}")

    # 2. Failure Case 2: Ego-Motion Desynchronization trên nuScenes (Time Debug Layer)
    fr_n_noego = load_frame(nuscenes_root, "scene-0103_010", use_ego_motion=False)
    uv_ne, d_ne, _ = project_velo_to_image(fr_n_noego["points"], fr_n_noego["calib"], fr_n_noego["image"].shape)
    vis_fail2 = overlay_points(fr_n_noego["image"], uv_ne, d_ne)
    for obj in fr_n_noego["labels"]:
        vis_fail2 = draw_box2d(vis_fail2, obj.bbox, label=f"{obj.type}")

    cv2.putText(
        vis_fail2,
        "FAILURE: Ego-motion ignored (Asynchronous Timestamps) -> Time Debug Layer",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
    )
    fail_path2 = fig_dir / "fail_02_ego_motion_drift.png"
    cv2.imwrite(str(fail_path2), vis_fail2)
    print(f"[PASS] Đã tạo ảnh failure case 2 -> {fail_path2}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LiDAR-Camera Projection QA Benchmark Suite")
    parser.add_argument("--kitti-root", default="data/kitti_mini", help="Thư mục dữ liệu kitti_mini")
    parser.add_argument("--kitti-frame", default="000011", help="Frame id của KITTI")
    parser.add_argument("--nuscenes-root", default="data/nuscenes_mini_subset", help="Thư mục nuscenes_mini_subset")
    parser.add_argument("--nuscenes-frame", default="scene-0103_010", help="Scene id của nuScenes")
    parser.add_argument("--synthetic-root", default="data/synthetic", help="Thư mục dữ liệu synthetic")
    parser.add_argument("--out-dir", default="results", help="Thư mục lưu kết quả CSV và figures")
    parser.add_argument("--all", action="store_true", help="Chạy toàn bộ thí nghiệm và lưu đầy đủ kết quả")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    print("=================================================================")
    print(" BẮT ĐẦU CHẠY BENCHMARK LIDAR-CAMERA PROJECTION QA (TOPIC A)")
    print("=================================================================")

    # 1. Yaw Sweep Experiment
    print("\n--- 1. Chạy Extrinsic Yaw Sweep (-3.0 deg đến +3.0 deg) ---")
    run_yaw_sweep(args.kitti_root, args.kitti_frame, out_dir)

    # 2. Translation Sweep Experiment
    print("\n--- 2. Chạy Extrinsic Lateral Translation Sweep (-0.15m đến +0.15m) ---")
    run_translation_sweep(args.kitti_root, args.kitti_frame, out_dir)

    # 3. Latency Benchmark
    print("\n--- 3. Đo Latency Benchmark Chuẩn (30 runs, bỏ warm-up) ---")
    run_latency_benchmark(args.kitti_root, args.kitti_frame, out_dir, n_runs=30)

    # 4. KITTI vs nuScenes Comparison
    print("\n--- 4. So sánh Thực Nghiệm Trên 2 Dataset Thật (KITTI & nuScenes) ---")
    run_kitti_vs_nuscenes_comparison(
        args.kitti_root, args.kitti_frame, args.nuscenes_root, args.nuscenes_frame, out_dir
    )

    # 5. Synthetic Dataset Defects Analysis
    print("\n--- 5. Kiểm tra và Phát hiện Các Lỗi Cài Sẵn trong data/synthetic ---")
    run_synthetic_health_check(args.synthetic_root, out_dir)

    # 6. Failure Cases Generation
    print("\n--- 6. Sinh Ảnh Failure Cases Trực Quan (fail_*.png) ---")
    generate_failure_cases(args.kitti_root, args.nuscenes_root, out_dir)

    print("\n=================================================================")
    print(" HOÀN THÀNH TOÀN BỘ THÍ NGHIỆM VÀ XUẤT ĐẦY ĐỦ BẰNG CHỨNG")
    print("=================================================================")


if __name__ == "__main__":
    main()
