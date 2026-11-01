import cv2
import numpy as np
from pathlib import Path
from lu_vp_detect import VPDetection
import numpy as np
from scipy.optimize import linear_sum_assignment
import csv


def load_gt_vps(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Ground-truth file not found: {path}")

    gt = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                values = [float(value) for value in line.split()]
            except ValueError:
                continue
            if len(values) < 3:
                continue

            v = np.asarray(values[:3], dtype=float)
            norm = np.linalg.norm(v)
            if norm > 1e-12:
                gt.append(v / norm)

    return np.asarray(gt, dtype=float).reshape(-1, 3)


def load_gt_lines(path):
    with open(path, "r") as f:
        lines = f.readlines()

    # Header only
    if len(lines) <= 1:
        return np.empty((0, 4), dtype=np.float32)

    data = np.loadtxt(path, skiprows=1)
    data = np.atleast_2d(data)

    return data[:, :4].astype(np.float32)


def normalize(v):
    v = np.asarray(v, dtype=float)
    norms = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(norms, 1e-12)


def angular_error_matrix(detected, gt):
    dots = np.abs(detected @ gt.T)
    dots = np.clip(dots, 0.0, 1.0)
    return np.degrees(np.arccos(dots))


def match_vps(errors, threshold):
    """Greedy one-to-one matching, lowest angular error first."""
    candidates = sorted(
        (errors[i, j], i, j)
        for i in range(errors.shape[0])
        for j in range(errors.shape[1])
    )

    used_detected = set()
    used_gt = set()
    matched_errors = []

    for error, i, j in candidates:
        if error > threshold:
            break
        if i in used_detected or j in used_gt:
            continue

        used_detected.add(i)
        used_gt.add(j)
        matched_errors.append(error)

    return len(matched_errors), matched_errors


def angular_error_matrix(pred_vps, gt_vps):
    """
    Pairwise angular errors in degrees.
    pred_vps: (N, 3) predicted 3D directions
    gt_vps:   (M, 3) ground-truth 3D directions
    """
    pred = np.asarray(pred_vps, dtype=float).reshape(-1, 3)
    gt = np.asarray(gt_vps, dtype=float).reshape(-1, 3)

    if len(pred) == 0 or len(gt) == 0:
        return np.empty((len(pred), len(gt)))

    # Normalize vectors
    pred /= np.maximum(np.linalg.norm(pred, axis=1, keepdims=True), 1e-12)
    gt /= np.maximum(np.linalg.norm(gt, axis=1, keepdims=True), 1e-12)

    # Vanishing-point directions are equivalent up to sign
    cosines = np.abs(pred @ gt.T)
    cosines = np.clip(cosines, 0.0, 1.0)

    return np.degrees(np.arccos(cosines))


def hungarian_match(pred_vps, gt_vps, threshold_deg=5.0):
    """
    Globally optimal one-to-one matching, minimizing total
    angular error. Matches above threshold are rejected.
    """
    errors = angular_error_matrix(pred_vps, gt_vps)

    n_pred, n_gt = errors.shape

    if n_pred == 0 or n_gt == 0:
        return [], n_pred, n_gt

    # Add dummy assignments so the optimizer can leave points unmatched.
    # A valid match costs its angular error; an unmatched point costs
    # threshold_deg. Thus, matches over the threshold are not beneficial.
    size = n_pred + n_gt
    unmatched_cost = threshold_deg
    invalid_cost = 1e6

    cost = np.full((size, size), invalid_cost, dtype=float)

    # Real prediction-to-ground-truth assignments
    cost[:n_pred, :n_gt] = np.where(
        errors <= threshold_deg,
        errors,
        invalid_cost
    )

    # Each prediction can be left unmatched
    cost[:n_pred, n_gt:] = unmatched_cost

    # Each ground-truth point can be left unmatched
    cost[n_pred:, :n_gt] = unmatched_cost

    # Dummy-to-dummy assignments
    cost[n_pred:, n_gt:] = 0.0

    row_ind, col_ind = linear_sum_assignment(cost)

    matches = []
    for r, c in zip(row_ind, col_ind):
        if r < n_pred and c < n_gt and errors[r, c] <= threshold_deg:
            matches.append((r, c, float(errors[r, c])))

    return matches, n_pred, n_gt





def evaluate_dataset1(image_root, gt_root, cx, cy, focal_px, focal_length, configuration):
    image_root = Path(image_root)
    gt_root = Path(gt_root)

    image_paths = sorted(p for p in image_root.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})

    if not image_paths:
        raise RuntimeError(f"No images found under {image_root}")

    THRESHOLDS = [5.0]
    total_detected = 0
    total_gt = 0
    processed = 0
    skipped = 0

    matched_counts = {t: 0 for t in THRESHOLDS}
    matched_errors = {t: [] for t in THRESHOLDS}

    detector = VPDetection(length_thresh=3, principal_point=(cx, cy), focal_length=focal_px, seed=42)
    for image_path in image_paths:
        gt_path = gt_root / f"{image_path.stem}.txt"
        if not gt_path.exists():
            #print(f"SKIP {image_path.name}: GT not found ({gt_path})")
            skipped += 1
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            #print(f"SKIP {image_path.name}: could not read image")
            skipped += 1
            continue

        try:
            gt_vps = load_gt_vps(gt_path)
            if len(gt_vps) == 0:
                #print(f"SKIP {image_path.name}: no valid GT directions")
                skipped += 1
                continue

            detected_vps = detector.find_vps(image)
            detected_vps = np.asarray(detected_vps, dtype=float).reshape(-1, 3)
            detected_vps = normalize(detected_vps)

            if not np.isfinite(detected_vps).all():
                detected_vps = detected_vps[np.isfinite(detected_vps).all(axis=1)]

            total_detected += len(detected_vps)
            total_gt += len(gt_vps)
            processed += 1

            image_matches = {}
            for threshold in THRESHOLDS:
                matches, _, _ = hungarian_match(detected_vps, gt_vps, threshold_deg=threshold)

                count = len(matches)
                image_precision = (count / len(detected_vps) if len(detected_vps) else 0.0)
                image_recall = (count / len(gt_vps) if len(gt_vps) else 0.0)
                image_mean_error = (np.mean([e for _, _, e in matches]) if matches else float("nan"))

                errors_at_threshold = [error for _, _, error in matches]

                matched_counts[threshold] += count
                matched_errors[threshold].extend(errors_at_threshold)
                image_matches[threshold] = count

                save_image_result(
                    focal_length=focal_length,
                    configuration=configuration,
                    image_name=image_path.name,
                    detected_vps=len(detected_vps),
                    gt_vps=len(gt_vps),
                    matched_vps=count,
                    precision=image_precision,
                    recall=image_recall,
                    mean_error=image_mean_error
                )

            # Show per-image precision and recall at 5 degrees.
            correct = image_matches[5.0]

        except Exception as exc:
            print(f"SKIP {image_path.name}: detector/evaluation error: {exc}")
            skipped += 1

    
    print("\n========== DATASET RESULTS ==========")
    print(f"Images found: {len(image_paths)}")
    print(f"Images evaluated: {processed}")
    print(f"Images skipped: {skipped}")
    print(f"Total detected VPs: {total_detected}")
    print(f"Total ground-truth VPs: {total_gt}")

    for threshold in THRESHOLDS:
        correct = matched_counts[threshold]

        precision = (correct / total_detected if total_detected else 0.0)
        recall = correct / total_gt if total_gt else 0.0
        aa = 100.0 * recall

        errors = matched_errors[threshold]
        mean_error = np.mean(errors) if errors else float("nan")

        print(f"\n--- Threshold: {threshold:.0f}° ---")
        print(f"Matched VPs: {correct}")
        print(f"False positives: {total_detected - correct}")
        print(f"False negatives: {total_gt - correct}")
        print(f"Precision: {precision:.2%}")
        print(f"Recall: {recall:.2%}")
        print(f"AA@{threshold:.0f}° (GT-normalized): {aa:.2f}%")
        print(f"Mean matched angular error: {mean_error:.3f}°")

        save_results(
            focal_length=focal_length,
            configuration=configuration,
            processed=processed,
            total_detected=total_detected,
            total_gt=total_gt,
            matched=correct,
            precision=precision,
            recall=recall,
            aa=aa,
            mean_error=mean_error
        )

def evaluate_dataset2(image_root, gt_root, line_root, file_ending, cx, cy, focal_px,  focal_length, configuration) :
    image_root = Path(image_root)
    gt_root = Path(gt_root)
    line_root = Path(line_root)

    image_paths = sorted(p for p in image_root.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    if not image_paths:
        raise RuntimeError(f"No images found under {image_root}")
    
    THRESHOLDS = [5.0]
    total_detected = 0
    total_gt = 0
    processed = 0
    skipped = 0

    matched_counts = {t: 0 for t in THRESHOLDS}
    matched_errors = {t: [] for t in THRESHOLDS}

    detector = VPDetection(length_thresh=3, principal_point=(cx, cy), focal_length=focal_px, seed=42)

    for image_path in image_paths:
        gt_path = gt_root / f"{image_path.stem}.txt"
        lines_path = line_root / f"{image_path.stem}{file_ending}"

        if not gt_path.exists() or not lines_path.exists():
            skipped += 1
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            skipped += 1
            continue

        try:
            gt_vps = load_gt_vps(gt_path)
            lines = load_gt_lines(lines_path)
            if len(gt_vps) == 0 or len(lines) < 2:
                skipped += 1
                continue

            detected_vps = detector.find_vps_from_lines(image, lines)
            detected_vps = np.asarray(detected_vps, dtype=float).reshape(-1, 3)
            detected_vps = normalize(detected_vps)
            detected_vps = detected_vps[np.isfinite(detected_vps).all(axis=1)]
            total_detected += len(detected_vps)
            total_gt += len(gt_vps)
            processed += 1

            for threshold in THRESHOLDS:
                matches, _, _ = hungarian_match(detected_vps, gt_vps, threshold_deg=threshold)

                count = len(matches)
                image_precision = (count / len(detected_vps) if len(detected_vps) else 0.0)
                image_recall = (count / len(gt_vps) if len(gt_vps) else 0.0)
                image_mean_error = (np.mean([e for _, _, e in matches]) if matches else float("nan"))

                matched_counts[threshold] += count
                matched_errors[threshold].extend(error for _, _, error in matches)

                save_image_result(
                    focal_length=focal_length,
                    configuration=configuration,
                    image_name=image_path.name,
                    detected_vps=len(detected_vps),
                    gt_vps=len(gt_vps),
                    matched_vps=count,
                    precision=image_precision,
                    recall=image_recall,
                    mean_error=image_mean_error
                )

        except Exception as exc:
            print(
                f"SKIP {image_path.name}: "
                f"detector/evaluation error: {exc}"
            )
            skipped += 1

    print("\n========== DATASET RESULTS ==========")
    print(f"Images found: {len(image_paths)}")
    print(f"Images evaluated: {processed}")
    print(f"Images skipped: {skipped}")
    print(f"Total detected VPs: {total_detected}")
    print(f"Total ground-truth VPs: {total_gt}")

    for threshold in THRESHOLDS:
        correct = matched_counts[threshold]

        precision = correct / total_detected if total_detected else 0.0
        recall = correct / total_gt if total_gt else 0.0
        aa = 100.0 * recall

        errors = matched_errors[threshold]
        mean_error = np.mean(errors) if errors else float("nan")

        print(f"\n--- Threshold: {threshold:.0f}° ---")
        print(f"Matched VPs: {correct}")
        print(f"False positives: {total_detected - correct}")
        print(f"False negatives: {total_gt - correct}")
        print(f"Precision: {precision:.2%}")
        print(f"Recall: {recall:.2%}")
        print(f"AA@{threshold:.0f}° (GT-normalized): {aa:.2f}%")
        print(f"Mean matched angular error: {mean_error:.3f}°")

        save_results(
            focal_length=focal_length,
            configuration=configuration,
            processed=processed,
            total_detected=total_detected,
            total_gt=total_gt,
            matched=correct,
            precision=precision,
            recall=recall,
            aa=aa,
            mean_error=mean_error
        )




def save_results(focal_length, configuration, processed, total_detected, total_gt, matched, precision, recall, aa, mean_error):
    file_exists = RESULTS_FILE.exists()
    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        if not file_exists:
            writer.writerow([
                "focal_length",
                "configuration",
                "images_evaluated",
                "detected_vps",
                "ground_truth_vps",
                "matched_vps",
                "precision",
                "recall",
                "AA_5deg",
                "mean_angular_error"
            ])

        writer.writerow([
            focal_length,
            configuration,
            processed,
            total_detected,
            total_gt,
            matched,
            precision,
            recall,
            aa,
            mean_error
        ])

def save_image_result(
    focal_length,
    configuration,
    image_name,
    detected_vps,
    gt_vps,
    matched_vps,
    precision,
    recall,
    mean_error
):
    file_exists = IMAGE_RESULTS_FILE.exists()

    with open(IMAGE_RESULTS_FILE, "a", newline="") as f:
        writer = csv.writer(f, delimiter="\t")

        if not file_exists:
            writer.writerow([
                "focal_length",
                "configuration",
                "image",
                "detected_vps",
                "ground_truth_vps",
                "matched_vps",
                "precision",
                "recall",
                "mean_angular_error"
            ])

        writer.writerow([
            focal_length,
            configuration,
            image_name,
            detected_vps,
            gt_vps,
            matched_vps,
            precision,
            recall,
            mean_error
        ])



if __name__ == "__main__":
    focals = []
    for f in range(675, 99, -25):
        focals.append(f)

    RESULTS_FILE = Path("vp_detection_results.txt")
    if RESULTS_FILE.exists():
        RESULTS_FILE.unlink()

    IMAGE_RESULTS_FILE = Path("vp_detection_per_image.txt")
    if IMAGE_RESULTS_FILE.exists():
        IMAGE_RESULTS_FILE.unlink()

    for focal in focals:
        print(f"\n=== focal length {focal} ===")
        imgs_path = f"../data/fisheye/{focal}/images"
        lines_path = f"../data/fisheye/{focal}/"

        print("\n Pure Image ==========================================================")
        evaluate_dataset1(
            image_root= imgs_path,
            gt_root="../data/vps3d",
            cx=307.55130528,
            cy=251.45424496,
            focal_px=focal,
            focal_length=focal,
            configuration="image"
        )
        
        print("\n Ground Truth Lines ==================================================")
        evaluate_dataset2(
            image_root=imgs_path,
            gt_root="../data/vps3d",
            line_root= f"{lines_path}lines",
            file_ending =".txt",
            cx=307.55130528,
            cy=251.45424496,
            focal_px=focal,
            focal_length=focal,
            configuration="all_lines"
        )
        
        print("\n Lines Toward VP ======================================================")
        evaluate_dataset2(
            image_root=imgs_path,
            gt_root="../data/vps3d",
            line_root=f"{lines_path}lines_toward_vp",
            file_ending=".txt",
            cx=307.55130528,
            cy=251.45424496,
            focal_px=focal,
            focal_length=focal,
            configuration="toward"
        )

        print("\n Random Lines =========================================================")
        evaluate_dataset2(
                image_root=imgs_path,
                gt_root="../data/vps3d",
                line_root=f"{lines_path}lines_random_vp",
                file_ending=".txt",
                cx=307.55130528,
                cy=251.45424496,
                focal_px=focal,
                focal_length=focal,
                configuration="random"
            )
       



"""
i have a theory that one of the reason the vanishing detection fails more when the image has curved lins, is because in normal perspective in manhattan world, each direction will have just one vanishing point, so all the lines pointing in that direction can help support the vanishing, while fisheye causes the lines to become curve, which makes it so each direction has two vanishing points. Therefore, there will be a point, whore going right means the lines point towards one vanishing point, while going left means the lines pount towards the other vanishing point, and this might cause some problems.
Therefore, i would like to test, if we cut off the lines going in the wrong direction of the true vanishing points, if that might improve the vanishing point detection.

how can i test this?
"""
    