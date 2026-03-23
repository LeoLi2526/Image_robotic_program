from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class BandRegion:
    x: int
    y: int
    w: int
    h: int
    area: int
    cx: float
    cy: float


@dataclass
class TrajectoryResult:
    image_width: int
    image_height: int
    image_center_x: float
    target_x: float
    lateral_error: float
    heading_angle_deg: float
    centerline_points: List[Tuple[int, int]]
    upper_inner_edge_points: List[Tuple[int, int]]
    lower_inner_edge_points: List[Tuple[int, int]]
    upper_inner_line: dict
    lower_inner_line: dict
    fitted_centerline_points: List[Tuple[int, int]]
    warped_upper_inner_line: dict
    warped_lower_inner_line: dict
    warped_centerline_points: List[Tuple[int, int]]
    upper_band: dict
    lower_band: dict


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def preprocess(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    _, binary_inv = cv2.threshold(blur, 90, 255, cv2.THRESH_BINARY_INV)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 11))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 5))
    mask = cv2.morphologyEx(binary_inv, cv2.MORPH_CLOSE, kernel_close)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    return gray, mask


def _find_row_segments(row_profile: np.ndarray, threshold: float, min_height: int) -> List[tuple[int, int]]:
    segments: List[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(row_profile):
        if value >= threshold and start is None:
            start = idx
        elif value < threshold and start is not None:
            if idx - start >= min_height:
                segments.append((start, idx))
            start = None

    if start is not None and len(row_profile) - start >= min_height:
        segments.append((start, len(row_profile)))
    return segments


def detect_bands(mask: np.ndarray) -> List[BandRegion]:
    height, width = mask.shape
    x0 = int(width * 0.1)
    x1 = int(width * 0.9)
    roi = mask[:, x0:x1]
    row_profile = roi.mean(axis=1) / 255.0
    min_height = max(24, int(height * 0.02))

    candidate_segments = _find_row_segments(row_profile, threshold=0.30, min_height=min_height)
    if len(candidate_segments) < 2:
        candidate_segments = _find_row_segments(row_profile, threshold=0.22, min_height=min_height)
    if len(candidate_segments) < 2:
        raise RuntimeError("未能稳定检测到两条主要黑带，请调整阈值或拍摄条件。")

    scored_segments: List[tuple[float, tuple[int, int]]] = []
    for y0, y1 in candidate_segments:
        score = float(row_profile[y0:y1].mean() * (y1 - y0))
        scored_segments.append((score, (y0, y1)))

    scored_segments.sort(key=lambda item: item[0], reverse=True)
    top_segments = sorted([segment for _, segment in scored_segments[:2]], key=lambda seg: seg[0])

    bands: List[BandRegion] = []
    for y0, y1 in top_segments:
        band_mask = mask[y0:y1, :]
        coords = np.column_stack(np.where(band_mask > 0))
        if len(coords) == 0:
            continue
        ys = coords[:, 0]
        xs = coords[:, 1]
        x_min = int(xs.min())
        x_max = int(xs.max())
        y_min = int(y0 + ys.min())
        y_max = int(y0 + ys.max())
        area = int(len(coords))
        cx = float(xs.mean())
        cy = float(y0 + ys.mean())
        bands.append(
            BandRegion(
                x=x_min,
                y=y_min,
                w=x_max - x_min + 1,
                h=y_max - y_min + 1,
                area=area,
                cx=cx,
                cy=cy,
            )
        )

    if len(bands) != 2:
        raise RuntimeError("未能稳定检测到两条主要黑带，请调整阈值或拍摄条件。")

    bands.sort(key=lambda band: band.cy)
    return bands


def extract_edge_points(mask: np.ndarray, upper: BandRegion, lower: BandRegion) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    width = mask.shape[1]
    x_start = max(max(upper.x, lower.x), int(width * 0.1))
    x_end = min(min(upper.x + upper.w, lower.x + lower.w), int(width * 0.9))
    if x_end <= x_start:
        raise RuntimeError("上下黑带没有足够重叠区域，无法计算中心轨迹。")

    xs: List[int] = []
    upper_lower_edge: List[int] = []
    lower_upper_edge: List[int] = []

    for x in range(x_start, x_end):
        upper_col = np.where(mask[upper.y: upper.y + upper.h, x] > 0)[0]
        lower_col = np.where(mask[lower.y: lower.y + lower.h, x] > 0)[0]
        if len(upper_col) == 0 or len(lower_col) == 0:
            continue

        upper_edge_y = upper.y + int(upper_col.max())
        lower_edge_y = lower.y + int(lower_col.min())

        if lower_edge_y <= upper_edge_y:
            continue

        xs.append(x)
        upper_lower_edge.append(upper_edge_y)
        lower_upper_edge.append(lower_edge_y)

    if len(xs) < 20:
        raise RuntimeError("有效边界点过少，无法拟合中心轨迹。")

    return np.array(xs), np.array(upper_lower_edge), np.array(lower_upper_edge)


def smooth_series(values: np.ndarray, window: int = 31) -> np.ndarray:
    if len(values) < window:
        window = max(5, len(values) // 2 * 2 + 1)
    if window < 3:
        return values.astype(np.float32)
    kernel = np.ones(window, dtype=np.float32) / window
    padded = np.pad(values.astype(np.float32), (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def fit_line(xs: np.ndarray, ys: np.ndarray) -> dict:
    xs_f = xs.astype(np.float32)
    ys_s = smooth_series(ys, window=41).astype(np.float32)

    points = np.column_stack([xs_f, ys_s]).reshape(-1, 1, 2)
    vx, vy, x0, y0 = cv2.fitLine(points, cv2.DIST_HUBER, 0, 0.01, 0.01)
    vx = float(vx[0])
    vy = float(vy[0])
    x0 = float(x0[0])
    y0 = float(y0[0])

    if abs(vx) < 1e-6:
        vx = 1e-6

    slope = vy / vx
    intercept = y0 - slope * x0

    residuals = ys_s - (slope * xs_f + intercept)
    mad = float(np.median(np.abs(residuals - np.median(residuals))))
    if mad > 1e-6:
        mask = np.abs(residuals) <= 2.5 * 1.4826 * mad
        if int(mask.sum()) >= max(20, len(xs) // 3):
            xs_refined = xs_f[mask]
            ys_refined = ys_s[mask]
            coeffs = np.polyfit(xs_refined, ys_refined, deg=1)
            slope = float(coeffs[0])
            intercept = float(coeffs[1])

    return {
        "slope": slope,
        "intercept": intercept,
        "x_start": int(xs.min()),
        "x_end": int(xs.max()),
    }


def compute_centerline(xs: np.ndarray, upper_edge: np.ndarray, lower_edge: np.ndarray) -> np.ndarray:
    upper_s = smooth_series(upper_edge)
    lower_s = smooth_series(lower_edge)
    center_y = ((upper_s + lower_s) / 2.0).astype(np.int32)
    points = np.column_stack([xs.astype(np.int32), center_y])
    return points


def perspective_rectify(
    image: np.ndarray,
    mask: np.ndarray,
    upper_inner_line: dict,
    lower_inner_line: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    x0 = max(int(upper_inner_line["x_start"]), int(lower_inner_line["x_start"]))
    x1 = min(int(upper_inner_line["x_end"]), int(lower_inner_line["x_end"]))
    if x1 <= x0:
        raise RuntimeError("有效内侧边线范围不足，无法做透视校正。")

    src = np.array(
        [
            [x0, upper_inner_line["slope"] * x0 + upper_inner_line["intercept"]],
            [x1, upper_inner_line["slope"] * x1 + upper_inner_line["intercept"]],
            [x1, lower_inner_line["slope"] * x1 + lower_inner_line["intercept"]],
            [x0, lower_inner_line["slope"] * x0 + lower_inner_line["intercept"]],
        ],
        dtype=np.float32,
    )

    top_width = float(np.linalg.norm(src[1] - src[0]))
    bottom_width = float(np.linalg.norm(src[2] - src[3]))
    left_height = float(np.linalg.norm(src[3] - src[0]))
    right_height = float(np.linalg.norm(src[2] - src[1]))
    warp_w = max(200, int(max(top_width, bottom_width)))
    warp_h = max(120, int(max(left_height, right_height)))

    dst = np.array(
        [
            [0, 0],
            [warp_w - 1, 0],
            [warp_w - 1, warp_h - 1],
            [0, warp_h - 1],
        ],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(src, dst)
    warped_image = cv2.warpPerspective(image, matrix, (warp_w, warp_h))
    warped_mask = cv2.warpPerspective(mask, matrix, (warp_w, warp_h))
    return warped_image, warped_mask, matrix, (warp_w, warp_h)


def fit_warped_inner_edges(warped_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict, dict, np.ndarray]:
    height, width = warped_mask.shape
    xs: List[int] = []
    upper_edge: List[int] = []
    lower_edge: List[int] = []

    for x in range(width):
        col = np.where(warped_mask[:, x] > 0)[0]
        if len(col) < 2:
            continue
        xs.append(x)
        upper_edge.append(int(col.min()))
        lower_edge.append(int(col.max()))

    if len(xs) < 20:
        raise RuntimeError("透视校正后的有效边缘点过少。")

    xs_arr = np.array(xs, dtype=np.int32)
    upper_arr = np.array(upper_edge, dtype=np.int32)
    lower_arr = np.array(lower_edge, dtype=np.int32)

    upper_line = fit_line(xs_arr, upper_arr)
    lower_line = fit_line(xs_arr, lower_arr)
    centerline = compute_centerline(xs_arr, upper_arr, lower_arr)

    upper_points = np.column_stack([xs_arr, upper_arr])
    lower_points = np.column_stack([xs_arr, lower_arr])
    return upper_points, lower_points, upper_line, lower_line, centerline


def estimate_heading(points: np.ndarray, image_width: int) -> tuple[float, float, float]:
    bottom_slice = points[len(points) * 2 // 3 :]
    if len(bottom_slice) < 2:
        bottom_slice = points

    coeffs = np.polyfit(bottom_slice[:, 0], bottom_slice[:, 1], deg=1)
    slope = float(coeffs[0])
    heading_angle_deg = float(np.degrees(np.arctan2(1.0, slope if abs(slope) > 1e-6 else 1e-6)) - 90.0)

    target_x = float(np.median(bottom_slice[:, 0]))
    image_center_x = image_width / 2.0
    lateral_error = target_x - image_center_x
    return target_x, lateral_error, heading_angle_deg


def draw_result(
    image: np.ndarray,
    bands: List[BandRegion],
    upper_inner_edge: np.ndarray,
    lower_inner_edge: np.ndarray,
    upper_inner_line: dict,
    lower_inner_line: dict,
    centerline: np.ndarray,
    outdir: Path,
) -> None:
    bands_img = image.copy()
    colors = [(0, 255, 0), (255, 0, 0)]
    labels = ["upper_band", "lower_band"]
    for band, color, label in zip(bands, colors, labels):
        cv2.rectangle(bands_img, (band.x, band.y), (band.x + band.w, band.y + band.h), color, 3)
        cv2.putText(
            bands_img,
            label,
            (band.x, max(30, band.y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )

    center_img = bands_img.copy()
    for i in range(1, len(upper_inner_edge)):
        p1 = tuple(upper_inner_edge[i - 1])
        p2 = tuple(upper_inner_edge[i])
        cv2.line(center_img, p1, p2, (0, 255, 255), 1)

    for i in range(1, len(lower_inner_edge)):
        p1 = tuple(lower_inner_edge[i - 1])
        p2 = tuple(lower_inner_edge[i])
        cv2.line(center_img, p1, p2, (255, 0, 255), 1)

    for i in range(1, len(centerline)):
        p1 = tuple(centerline[i - 1])
        p2 = tuple(centerline[i])
        cv2.line(center_img, p1, p2, (0, 0, 255), 2)

    upper_x0 = int(upper_inner_line["x_start"])
    upper_x1 = int(upper_inner_line["x_end"])
    lower_x0 = int(lower_inner_line["x_start"])
    lower_x1 = int(lower_inner_line["x_end"])
    upper_line_p1 = (upper_x0, int(upper_inner_line["slope"] * upper_x0 + upper_inner_line["intercept"]))
    upper_line_p2 = (upper_x1, int(upper_inner_line["slope"] * upper_x1 + upper_inner_line["intercept"]))
    lower_line_p1 = (lower_x0, int(lower_inner_line["slope"] * lower_x0 + lower_inner_line["intercept"]))
    lower_line_p2 = (lower_x1, int(lower_inner_line["slope"] * lower_x1 + lower_inner_line["intercept"]))
    cv2.line(center_img, upper_line_p1, upper_line_p2, (0, 200, 255), 3)
    cv2.line(center_img, lower_line_p1, lower_line_p2, (255, 0, 200), 3)

    fitted_x0 = max(upper_x0, lower_x0)
    fitted_x1 = min(upper_x1, lower_x1)
    if fitted_x1 > fitted_x0:
        fitted_centerline = []
        for x in range(fitted_x0, fitted_x1 + 1):
            upper_y = upper_inner_line["slope"] * x + upper_inner_line["intercept"]
            lower_y = lower_inner_line["slope"] * x + lower_inner_line["intercept"]
            fitted_centerline.append((x, int((upper_y + lower_y) / 2.0)))
        for i in range(1, len(fitted_centerline)):
            cv2.line(center_img, fitted_centerline[i - 1], fitted_centerline[i], (0, 128, 255), 2)

    h, w = image.shape[:2]
    cv2.line(center_img, (w // 2, 0), (w // 2, h), (255, 255, 0), 1)

    cv2.imwrite(str(outdir / "03_bands_overlay.png"), bands_img)
    cv2.imwrite(str(outdir / "04_centerline_overlay.png"), center_img)


def draw_warped_result(
    warped_image: np.ndarray,
    upper_inner_edge: np.ndarray,
    lower_inner_edge: np.ndarray,
    upper_inner_line: dict,
    lower_inner_line: dict,
    centerline: np.ndarray,
    outdir: Path,
) -> None:
    canvas = warped_image.copy()
    for i in range(1, len(upper_inner_edge)):
        cv2.line(canvas, tuple(upper_inner_edge[i - 1]), tuple(upper_inner_edge[i]), (0, 255, 255), 1)
    for i in range(1, len(lower_inner_edge)):
        cv2.line(canvas, tuple(lower_inner_edge[i - 1]), tuple(lower_inner_edge[i]), (255, 0, 255), 1)
    for i in range(1, len(centerline)):
        cv2.line(canvas, tuple(centerline[i - 1]), tuple(centerline[i]), (0, 0, 255), 2)

    for line_info, color in ((upper_inner_line, (0, 200, 255)), (lower_inner_line, (255, 0, 200))):
        x0 = int(line_info["x_start"])
        x1 = int(line_info["x_end"])
        p1 = (x0, int(line_info["slope"] * x0 + line_info["intercept"]))
        p2 = (x1, int(line_info["slope"] * x1 + line_info["intercept"]))
        cv2.line(canvas, p1, p2, color, 2)

    cv2.imwrite(str(outdir / "05_warped_overlay.png"), canvas)


def save_result(
    outdir: Path,
    gray: np.ndarray,
    mask: np.ndarray,
    warped_image: np.ndarray,
    bands: List[BandRegion],
    upper_inner_edge: np.ndarray,
    lower_inner_edge: np.ndarray,
    upper_inner_line: dict,
    lower_inner_line: dict,
    centerline: np.ndarray,
    warped_upper_inner_edge: np.ndarray,
    warped_lower_inner_edge: np.ndarray,
    warped_upper_inner_line: dict,
    warped_lower_inner_line: dict,
    warped_centerline: np.ndarray,
    result: TrajectoryResult,
) -> None:
    cv2.imwrite(str(outdir / "01_gray.png"), gray)
    cv2.imwrite(str(outdir / "02_black_mask.png"), mask)
    draw_result(
        cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
        bands,
        upper_inner_edge,
        lower_inner_edge,
        upper_inner_line,
        lower_inner_line,
        centerline,
        outdir,
    )
    draw_warped_result(
        warped_image,
        warped_upper_inner_edge,
        warped_lower_inner_edge,
        warped_upper_inner_line,
        warped_lower_inner_line,
        warped_centerline,
        outdir,
    )

    with open(outdir / "result.json", "w", encoding="utf-8") as f:
        json.dump(asdict(result), f, ensure_ascii=False, indent=2)


def run(image_path: Path, outdir: Path) -> TrajectoryResult:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")

    ensure_dir(outdir)

    gray, mask = preprocess(image)
    bands = detect_bands(mask)
    upper, lower = bands
    xs, upper_edge, lower_edge = extract_edge_points(mask, upper, lower)
    upper_inner_line = fit_line(xs, upper_edge)
    lower_inner_line = fit_line(xs, lower_edge)
    upper_inner_edge_points = np.column_stack([xs.astype(np.int32), upper_edge.astype(np.int32)])
    lower_inner_edge_points = np.column_stack([xs.astype(np.int32), lower_edge.astype(np.int32)])
    centerline = compute_centerline(xs, upper_edge, lower_edge)
    warped_image, warped_mask, _, _ = perspective_rectify(image, mask, upper_inner_line, lower_inner_line)
    warped_upper_edge_points, warped_lower_edge_points, warped_upper_line, warped_lower_line, warped_centerline = fit_warped_inner_edges(
        warped_mask
    )

    fitted_centerline_points: List[Tuple[int, int]] = []
    fitted_x0 = max(int(upper_inner_line["x_start"]), int(lower_inner_line["x_start"]))
    fitted_x1 = min(int(upper_inner_line["x_end"]), int(lower_inner_line["x_end"]))
    if fitted_x1 > fitted_x0:
        for x in range(fitted_x0, fitted_x1 + 1):
            upper_y = upper_inner_line["slope"] * x + upper_inner_line["intercept"]
            lower_y = lower_inner_line["slope"] * x + lower_inner_line["intercept"]
            fitted_centerline_points.append((x, int((upper_y + lower_y) / 2.0)))

    target_x, lateral_error, heading_angle_deg = estimate_heading(centerline, image.shape[1])

    result = TrajectoryResult(
        image_width=image.shape[1],
        image_height=image.shape[0],
        image_center_x=image.shape[1] / 2.0,
        target_x=target_x,
        lateral_error=lateral_error,
        heading_angle_deg=heading_angle_deg,
        centerline_points=[(int(x), int(y)) for x, y in centerline],
        upper_inner_edge_points=[(int(x), int(y)) for x, y in upper_inner_edge_points],
        lower_inner_edge_points=[(int(x), int(y)) for x, y in lower_inner_edge_points],
        upper_inner_line=upper_inner_line,
        lower_inner_line=lower_inner_line,
        fitted_centerline_points=fitted_centerline_points,
        warped_upper_inner_line=warped_upper_line,
        warped_lower_inner_line=warped_lower_line,
        warped_centerline_points=[(int(x), int(y)) for x, y in warped_centerline],
        upper_band=asdict(upper),
        lower_band=asdict(lower),
    )

    save_result(
        outdir,
        gray,
        mask,
        warped_image,
        bands,
        upper_inner_edge_points,
        lower_inner_edge_points,
        upper_inner_line,
        lower_inner_line,
        centerline,
        warped_upper_edge_points,
        warped_lower_edge_points,
        warped_upper_line,
        warped_lower_line,
        warped_centerline,
        result,
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从双黑带图像中提取通道中心轨迹")
    parser.add_argument("--image", type=Path, required=True, help="输入图片路径")
    parser.add_argument("--outdir", type=Path, default=Path("output"), help="输出目录")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(args.image, args.outdir)
    summary = {
        "image_size": [result.image_width, result.image_height],
        "image_center_x": result.image_center_x,
        "target_x": result.target_x,
        "lateral_error": result.lateral_error,
        "heading_angle_deg": result.heading_angle_deg,
        "centerline_point_count": len(result.centerline_points),
        "upper_inner_line": result.upper_inner_line,
        "lower_inner_line": result.lower_inner_line,
        "warped_upper_inner_line": result.warped_upper_inner_line,
        "warped_lower_inner_line": result.warped_lower_inner_line,
        "result_file": str(args.outdir / "result.json"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
