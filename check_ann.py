import argparse
import glob
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Enhanced 3D->2D consistency diagnostics for KITTI-like labels.')
    parser.add_argument(
        '--data-root',
        default='/home/yxc/1_data/3DBox_Annotation_20260305154810',
        help='Dataset root containing training/image_2, training/label_2, training/calib.')
    parser.add_argument(
        '--split',
        default='training',
        choices=['training', 'testing'],
        help='Dataset split folder.')
    parser.add_argument(
        '--frame-id',
        default='1772689948923010048',
        help='Optional frame id for single-frame debug image output.')
    parser.add_argument(
        '--out-dir',
        default='debug_projection',
        help='Directory to save diagnostics.')
    parser.add_argument(
        '--min-visible-area',
        type=float,
        default=1.0,
        help='Minimum clipped 2D area to treat GT as visible.')
    parser.add_argument(
        '--low-iou-thr',
        type=float,
        default=0.5,
        help='Threshold for low-IoU samples.')
    parser.add_argument(
        '--save-low-iou-num',
        type=int,
        default=30,
        help='Number of worst samples to save as images.')
    return parser.parse_args()


def read_p2_from_calib(calib_path: str) -> np.ndarray:
    with open(calib_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if parts and parts[0] == 'P2:':
                values = np.array(list(map(float, parts[1:])), dtype=np.float64)
                return values.reshape(3, 4)
    raise ValueError(f'P2 not found in calib: {calib_path}')


def project_3d_corners(
        x: float,
        y: float,
        z: float,
        h: float,
        w: float,
        l: float,
        ry: float,
        p2: np.ndarray) -> Optional[np.ndarray]:
    # KITTI-like camera box: location is bottom center, dims are h,w,l.
    x_corners = np.array([l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2], dtype=np.float64)
    y_corners = np.array([0, 0, 0, 0, -h, -h, -h, -h], dtype=np.float64)
    z_corners = np.array([w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], dtype=np.float64)

    corners = np.vstack([x_corners, y_corners, z_corners])
    rot = np.array([
        [np.cos(ry), 0.0, np.sin(ry)],
        [0.0, 1.0, 0.0],
        [-np.sin(ry), 0.0, np.cos(ry)],
    ], dtype=np.float64)

    corners = rot @ corners
    corners[0, :] += x
    corners[1, :] += y
    corners[2, :] += z

    if np.any(corners[2, :] <= 1e-6):
        return None

    corners_h = np.vstack([corners, np.ones((1, 8), dtype=np.float64)])
    uv_h = p2 @ corners_h
    uv = uv_h[:2, :] / uv_h[2, :]
    return uv.T


def corners_to_bbox(corners_2d: np.ndarray) -> np.ndarray:
    xmin = float(np.min(corners_2d[:, 0]))
    ymin = float(np.min(corners_2d[:, 1]))
    xmax = float(np.max(corners_2d[:, 0]))
    ymax = float(np.max(corners_2d[:, 1]))
    return np.array([xmin, ymin, xmax, ymax], dtype=np.float64)


def clip_bbox(bbox: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    out = bbox.copy()
    out[0::2] = np.clip(out[0::2], 0, img_w - 1)
    out[1::2] = np.clip(out[1::2], 0, img_h - 1)
    return out


def bbox_area(bbox: np.ndarray) -> float:
    return max(0.0, float(bbox[2] - bbox[0])) * max(0.0, float(bbox[3] - bbox[1]))


def bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1 = max(float(a[0]), float(b[0]))
    iy1 = max(float(a[1]), float(b[1]))
    ix2 = min(float(a[2]), float(b[2]))
    iy2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = bbox_area(a) + bbox_area(b) - inter
    return inter / union if union > 1e-8 else 0.0


def is_on_border(bbox: np.ndarray, img_w: int, img_h: int, eps: int = 1) -> bool:
    return (
        bbox[0] <= eps or
        bbox[1] <= eps or
        bbox[2] >= (img_w - 1 - eps) or
        bbox[3] >= (img_h - 1 - eps)
    )


def draw_debug_overlay(
        image: np.ndarray,
        corners_2d: np.ndarray,
        gt_bbox: np.ndarray,
        pred_bbox: np.ndarray,
        title: str,
        footer: str) -> np.ndarray:
    vis = image.copy()

    corners_i = np.round(corners_2d).astype(np.int32)
    for i in range(4):
        cv2.line(vis, tuple(corners_i[i]), tuple(corners_i[(i + 1) % 4]), (0, 255, 0), 2)
        cv2.line(vis, tuple(corners_i[i + 4]), tuple(corners_i[((i + 1) % 4) + 4]), (0, 255, 0), 2)
        cv2.line(vis, tuple(corners_i[i]), tuple(corners_i[i + 4]), (0, 255, 0), 2)

    gx1, gy1, gx2, gy2 = np.round(gt_bbox).astype(np.int32)
    px1, py1, px2, py2 = np.round(pred_bbox).astype(np.int32)
    cv2.rectangle(vis, (gx1, gy1), (gx2, gy2), (255, 0, 0), 2)
    cv2.rectangle(vis, (px1, py1), (px2, py2), (0, 0, 255), 2)

    cv2.putText(vis, title, (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
    cv2.putText(vis, 'Blue: GT 2D  Red: Projected 2D  Green: 3D wireframe', (20, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)
    cv2.putText(vis, footer, (20, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
    return vis


def collect_samples(
        data_root: str,
        split: str,
        min_visible_area: float) -> Tuple[List[Dict], Dict[str, int]]:
    image_dir = os.path.join(data_root, split, 'image_2')
    label_dir = os.path.join(data_root, split, 'label_2')
    calib_dir = os.path.join(data_root, split, 'calib')

    label_files = sorted(glob.glob(os.path.join(label_dir, '*.txt')))
    samples: List[Dict] = []
    counters = {
        'total_lines': 0,
        'skipped_bad_line': 0,
        'skipped_missing_image': 0,
        'skipped_invalid_project': 0,
        'skipped_invisible_gt': 0,
        'kept_visible': 0,
    }

    for label_path in label_files:
        frame_id = os.path.splitext(os.path.basename(label_path))[0]
        calib_path = os.path.join(calib_dir, f'{frame_id}.txt')
        img_path = os.path.join(image_dir, f'{frame_id}.png')
        if not os.path.exists(img_path):
            counters['skipped_missing_image'] += 1
            continue
        if not os.path.exists(calib_path):
            counters['skipped_missing_image'] += 1
            continue

        p2 = read_p2_from_calib(calib_path)
        img = cv2.imread(img_path)
        if img is None:
            counters['skipped_missing_image'] += 1
            continue
        img_h, img_w = img.shape[:2]

        with open(label_path, 'r', encoding='utf-8') as f:
            for line_no, line in enumerate(f, 1):
                counters['total_lines'] += 1
                parts = line.strip().split()
                if len(parts) < 15:
                    counters['skipped_bad_line'] += 1
                    continue

                cls_name = parts[0]
                gt_bbox = np.array(list(map(float, parts[4:8])), dtype=np.float64)
                gt_bbox = clip_bbox(gt_bbox, img_w, img_h)
                if bbox_area(gt_bbox) <= min_visible_area:
                    counters['skipped_invisible_gt'] += 1
                    continue

                h, w, l = map(float, parts[8:11])
                x, y, z = map(float, parts[11:14])
                ry = float(parts[14])

                corners_2d = project_3d_corners(x, y, z, h, w, l, ry, p2)
                if corners_2d is None:
                    counters['skipped_invalid_project'] += 1
                    continue

                pred_bbox = corners_to_bbox(corners_2d)
                pred_bbox = clip_bbox(pred_bbox, img_w, img_h)
                if bbox_area(pred_bbox) <= min_visible_area:
                    counters['skipped_invalid_project'] += 1
                    continue

                iou = bbox_iou(gt_bbox, pred_bbox)
                gt_center = np.array([(gt_bbox[0] + gt_bbox[2]) * 0.5, (gt_bbox[1] + gt_bbox[3]) * 0.5])
                pred_center = np.array([(pred_bbox[0] + pred_bbox[2]) * 0.5, (pred_bbox[1] + pred_bbox[3]) * 0.5])
                delta = pred_center - gt_center

                samples.append({
                    'frame_id': frame_id,
                    'line_no': line_no,
                    'class': cls_name,
                    'img_path': img_path,
                    'img_h': img_h,
                    'img_w': img_w,
                    'gt_bbox': gt_bbox,
                    'pred_bbox': pred_bbox,
                    'corners_2d': corners_2d,
                    'iou': float(iou),
                    'dx': float(delta[0]),
                    'dy': float(delta[1]),
                    'gt_on_border': is_on_border(gt_bbox, img_w, img_h),
                    'pred_on_border': is_on_border(pred_bbox, img_w, img_h),
                    'h': h,
                    'w': w,
                    'l': l,
                    'x': x,
                    'y': y,
                    'z': z,
                    'ry': ry,
                })
                counters['kept_visible'] += 1

    return samples, counters


def print_stats(samples: List[Dict], low_iou_thr: float) -> None:
    if not samples:
        print('No visible/projectable samples to analyze.')
        return

    ious = np.array([s['iou'] for s in samples], dtype=np.float64)
    dxs = np.array([s['dx'] for s in samples], dtype=np.float64)
    dys = np.array([s['dy'] for s in samples], dtype=np.float64)
    lows = [s for s in samples if s['iou'] < low_iou_thr]

    print('\n=== Visible Target Consistency Stats ===')
    print(f'Samples: {len(samples)}')
    print(f'IoU mean/median: {ious.mean():.4f} / {np.median(ious):.4f}')
    print(f'IoU >= 0.7: {(ious >= 0.7).mean():.4f}')
    print(f'IoU >= 0.5: {(ious >= 0.5).mean():.4f}')
    print(f'IoU >= 0.3: {(ious >= 0.3).mean():.4f}')
    print(f'Center dx mean/median: {dxs.mean():.2f} / {np.median(dxs):.2f}')
    print(f'Center dy mean/median: {dys.mean():.2f} / {np.median(dys):.2f}')
    print(f'Low-IoU (<{low_iou_thr:.2f}) count: {len(lows)}  ratio: {len(lows) / len(samples):.4f}')

    if lows:
        low_gt_border = np.mean([s['gt_on_border'] for s in lows])
        low_pred_border = np.mean([s['pred_on_border'] for s in lows])
        all_gt_border = np.mean([s['gt_on_border'] for s in samples])
        all_pred_border = np.mean([s['pred_on_border'] for s in samples])
        print(f'Low-IoU gt_on_border ratio: {low_gt_border:.4f} (all: {all_gt_border:.4f})')
        print(f'Low-IoU pred_on_border ratio: {low_pred_border:.4f} (all: {all_pred_border:.4f})')

    by_class: Dict[str, List[float]] = {}
    for s in samples:
        by_class.setdefault(s['class'], []).append(s['iou'])

    print('\n=== Class-wise IoU ===')
    for cls_name, vals in sorted(by_class.items(), key=lambda x: x[0]):
        arr = np.array(vals, dtype=np.float64)
        print(
            f'{cls_name:20s} '
            f'N={len(vals):4d} '
            f'mean={arr.mean():.4f} '
            f'med={np.median(arr):.4f} '
            f'>=0.5={(arr >= 0.5).mean():.4f}')


def save_single_frame_debug(samples: List[Dict], frame_id: str, out_dir: str) -> None:
    frame_samples = [s for s in samples if s['frame_id'] == frame_id]
    if not frame_samples:
        print(f'No visible/projectable sample found for frame_id={frame_id}.')
        return

    os.makedirs(out_dir, exist_ok=True)
    # Pick the first instance in this frame for quick debugging.
    s = frame_samples[0]
    img = cv2.imread(s['img_path'])
    if img is None:
        print(f'Failed to read image: {s["img_path"]}')
        return

    title = f'Frame {s["frame_id"]}  cls={s["class"]}  line={s["line_no"]}'
    footer = f'IoU={s["iou"]:.3f}  dx={s["dx"]:.1f}  dy={s["dy"]:.1f}  z={s["z"]:.2f}  ry={s["ry"]:.3f}'
    vis = draw_debug_overlay(img, s['corners_2d'], s['gt_bbox'], s['pred_bbox'], title, footer)
    out_path = os.path.join(out_dir, f'frame_{frame_id}_first_instance.jpg')
    cv2.imwrite(out_path, vis)
    print(f'Saved single-frame debug image: {os.path.abspath(out_path)}')


def save_worst_samples(samples: List[Dict], out_dir: str, max_num: int) -> None:
    if not samples or max_num <= 0:
        return

    os.makedirs(out_dir, exist_ok=True)
    worst = sorted(samples, key=lambda x: x['iou'])[:max_num]
    saved = 0
    for idx, s in enumerate(worst, 1):
        img = cv2.imread(s['img_path'])
        if img is None:
            continue

        title = f'Worst#{idx} {s["frame_id"]}:{s["line_no"]} cls={s["class"]}'
        footer = (
            f'IoU={s["iou"]:.3f}  dx={s["dx"]:.1f}  dy={s["dy"]:.1f}  '
            f'gt_border={int(s["gt_on_border"])} pred_border={int(s["pred_on_border"])}  z={s["z"]:.2f}')
        vis = draw_debug_overlay(img, s['corners_2d'], s['gt_bbox'], s['pred_bbox'], title, footer)

        filename = f'worst_{idx:03d}_{s["frame_id"]}_line{s["line_no"]}_iou{s["iou"]:.3f}.jpg'
        out_path = os.path.join(out_dir, filename)
        cv2.imwrite(out_path, vis)
        saved += 1

    print(f'Saved worst-case overlays: {saved}/{len(worst)} to {os.path.abspath(out_dir)}')


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    samples, counters = collect_samples(
        data_root=args.data_root,
        split=args.split,
        min_visible_area=args.min_visible_area)

    print('=== Scan Summary ===')
    for k, v in counters.items():
        print(f'{k:>24s}: {v}')

    print_stats(samples, low_iou_thr=args.low_iou_thr)
    save_single_frame_debug(samples, frame_id=args.frame_id, out_dir=args.out_dir)
    save_worst_samples(samples, out_dir=args.out_dir, max_num=args.save_low_iou_num)


if __name__ == '__main__':
    main()
