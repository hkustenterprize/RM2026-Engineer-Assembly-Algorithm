"""
Unit tests for RandomCropPose in random_crop_aug.py.

Run:
    conda run -n mujoco-sim python test_random_crop_aug.py
"""

import sys
import os
import random
import numpy as np

# ── 路径设置 ──────────────────────────────────────────────────────────────────
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

from random_crop_aug import RandomCropPose

try:
    from ultralytics.utils.instance import Instances

    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False
    print("[WARN] ultralytics not found — will use minimal Instances stub")


# ── Minimal stub for Instances (used when ultralytics is unavailable) ─────────
class _InstancesStub:
    """Minimal stub that mimics the ultralytics.utils.instance.Instances API."""

    def __init__(self, bboxes, keypoints=None):
        self.bboxes = bboxes.copy()
        self._bboxes = self.bboxes
        self.keypoints = keypoints.copy() if keypoints is not None else None
        self._segments = None

    def __len__(self):
        return len(self.bboxes)


def make_instances(bboxes, keypoints=None):
    if HAS_ULTRALYTICS:
        inst = Instances(
            bboxes=bboxes.astype(np.float32),
            keypoints=keypoints.astype(np.float32) if keypoints is not None else None,
            bbox_format="xyxy",
            normalized=False,
        )
        return inst
    else:
        return _InstancesStub(bboxes, keypoints)


# ── Helper ────────────────────────────────────────────────────────────────────


def make_labels(img_h=480, img_w=640, n_instances=2, n_kp=12, seed=0):
    """Create a minimal labels dict that RandomCropPose expects."""
    rng = np.random.default_rng(seed)

    img = (rng.random((img_h, img_w, 3)) * 255).astype(np.uint8)

    # Two bboxes that cover separate regions in the image (xyxy absolute)
    bboxes = np.array(
        [
            [50, 50, 200, 200],  # fully within most crops
            [400, 300, 600, 450],  # near the right/bottom edge
        ],
        dtype=np.float32,
    )[:n_instances]

    # Keypoints: (N, K, 3) — evenly spread across the image
    kpts = np.zeros((n_instances, n_kp, 3), dtype=np.float32)
    for i in range(n_instances):
        for k in range(n_kp):
            x = rng.integers(bboxes[i, 0], bboxes[i, 2] + 1)
            y = rng.integers(bboxes[i, 1], bboxes[i, 3] + 1)
            kpts[i, k] = [x, y, 2]  # vis=2 (visible)

    instances = make_instances(bboxes, kpts)
    labels = {
        "img": img,
        "cls": np.array([0] * n_instances, dtype=np.int64),
        "instances": instances,
    }
    return labels, bboxes.copy(), kpts.copy()


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_image_shape():
    """Cropped image should have shape proportional to crop_scale."""
    labels, _, _ = make_labels(img_h=480, img_w=640)
    tf = RandomCropPose(crop_scale=(0.6, 0.6), p=1.0)
    result = tf(labels)
    h, w = result["img"].shape[:2]
    assert h == int(480 * 0.6), f"Expected h={int(480*0.6)}, got {h}"
    assert w == int(640 * 0.6), f"Expected w={int(640*0.6)}, got {w}"
    print("[PASS] test_image_shape")


def test_no_crop_when_p0():
    """p=0.0 should leave the image unchanged."""
    labels, _, _ = make_labels()
    orig_h, orig_w = labels["img"].shape[:2]
    tf = RandomCropPose(crop_scale=(0.5, 0.8), p=0.0)
    result = tf(labels)
    h, w = result["img"].shape[:2]
    assert h == orig_h and w == orig_w, "Image shape should be unchanged when p=0"
    print("[PASS] test_no_crop_when_p0")


def test_keypoints_inside_crop_unchanged():
    """
    When we force a crop that covers the full image, ALL keypoints should keep
    their original visibility and shifted coordinates should be identical to original.
    """
    labels, _, kpts_orig = make_labels(
        img_h=480, img_w=640, n_instances=1, n_kp=5, seed=1
    )
    tf = RandomCropPose(crop_scale=(1.0, 1.0), p=1.0)  # crop_scale=1.0 → no actual crop
    result = tf(labels)
    kpts_out = result["instances"].keypoints  # (N, K, 3)

    # All vis flags should be preserved
    assert np.all(
        kpts_out[:, :, 2] == kpts_orig[:1, :5, 2]
    ), "Visibility flags changed unexpectedly for full-image crop"
    # Coordinates should equal original (offset=0)
    assert np.allclose(
        kpts_out[:, :, :2], kpts_orig[:1, :5, :2]
    ), "Keypoint coords changed for full-image crop"
    print("[PASS] test_keypoints_inside_crop_unchanged")


def test_keypoints_outside_crop_set_to_oov():
    """
    Keypoints that fall outside the crop region must have vis=0 after transform.
    We craft a scenario where we know exactly which keypoints will be outside.
    """
    img_h, img_w = 200, 200

    # One instance with two keypoints:
    #   kp0 at (50, 50)  — will be INSIDE the crop [0:100, 0:100]
    #   kp1 at (150, 150) — will be OUTSIDE the crop [0:100, 0:100]
    bboxes = np.array([[0, 0, 200, 200]], dtype=np.float32)
    kpts = np.array([[[50, 50, 2], [150, 150, 2]]], dtype=np.float32)  # (1, 2, 3)
    instances = make_instances(bboxes, kpts)
    labels = {
        "img": np.zeros((img_h, img_w, 3), dtype=np.uint8),
        "cls": np.array([0]),
        "instances": instances,
    }

    # Force crop to exactly [0:100, 0:100]
    tf = RandomCropPose(crop_scale=(0.5, 0.5), p=1.0)

    # Monkey-patch to fix the random crop to top-left corner
    original_randint = random.randint
    original_uniform = random.uniform
    random.randint = lambda a, b: 0  # y1=0, x1=0
    random.uniform = lambda a, b: 0.5  # scale exactly 0.5
    try:
        result = tf(labels)
    finally:
        random.randint = original_randint
        random.uniform = original_uniform

    kpts_out = result["instances"].keypoints  # (N, K, 3)
    # kp0 (50-0=50) should be inside (50 < 100) → vis preserved (2)
    assert kpts_out[0, 0, 2] == 2, f"kp0 vis expected 2, got {kpts_out[0, 0, 2]}"
    assert kpts_out[0, 0, 0] == 50, f"kp0 x expected 50, got {kpts_out[0, 0, 0]}"
    # kp1 (150-0=150) should be outside (150 >= 100) → vis=0
    assert kpts_out[0, 1, 2] == 0, f"kp1 vis expected 0 (OOV), got {kpts_out[0, 1, 2]}"
    assert (
        kpts_out[0, 1, 0] == 0
    ), f"kp1 x should be zeroed out, got {kpts_out[0, 1, 0]}"
    print("[PASS] test_keypoints_outside_crop_set_to_oov")


def test_keypoint_coords_shifted():
    """Keypoints inside crop should have coordinates relative to crop origin."""
    img_h, img_w = 200, 200
    # kp at (120, 80), crop origin will be (50, 30)
    bboxes = np.array([[50, 30, 170, 150]], dtype=np.float32)
    kpts = np.array([[[120, 80, 2]]], dtype=np.float32)  # (1, 1, 3)
    instances = make_instances(bboxes, kpts)
    labels = {
        "img": np.zeros((img_h, img_w, 3), dtype=np.uint8),
        "cls": np.array([0]),
        "instances": instances,
    }

    tf = RandomCropPose(crop_scale=(0.7, 0.7), p=1.0)

    call_count = [0]

    def fixed_randint(a, b):
        # First call: y1=30, second: x1=50
        call_count[0] += 1
        return 30 if call_count[0] == 1 else 50

    original_randint = random.randint
    original_uniform = random.uniform
    random.randint = fixed_randint
    random.uniform = lambda a, b: 0.7
    try:
        result = tf(labels)
    finally:
        random.randint = original_randint
        random.uniform = original_uniform

    kpts_out = result["instances"].keypoints
    # crop_h = int(200 * 0.7) = 140, crop_w = 140
    # y1=30, x1=50
    # kp inside: x=120-50=70, y=80-30=50, vis=2
    assert kpts_out[0, 0, 2] == 2, f"vis expected 2, got {kpts_out[0, 0, 2]}"
    assert kpts_out[0, 0, 0] == 70, f"shifted x expected 70, got {kpts_out[0, 0, 0]}"
    assert kpts_out[0, 0, 1] == 50, f"shifted y expected 50, got {kpts_out[0, 0, 1]}"
    print("[PASS] test_keypoint_coords_shifted")


def test_instance_removed_when_bbox_too_small():
    """
    Instances where the cropped bbox area < min_bbox_area_ratio * original area
    should be removed.
    """
    img_h, img_w = 200, 200
    # Instance: bbox at (160, 0, 200, 40) — almost entirely outside a [0:100, 0:100] crop
    bboxes = np.array([[160, 0, 200, 40]], dtype=np.float32)
    kpts = np.array([[[180, 20, 2]]], dtype=np.float32)
    instances = make_instances(bboxes, kpts)
    labels = {
        "img": np.zeros((img_h, img_w, 3), dtype=np.uint8),
        "cls": np.array([0]),
        "instances": instances,
    }

    tf = RandomCropPose(crop_scale=(0.5, 0.5), p=1.0, min_bbox_area_ratio=0.1)

    orig_randint = random.randint
    orig_uniform = random.uniform
    random.randint = lambda a, b: 0
    random.uniform = lambda a, b: 0.5  # crop = [0:100, 0:100]
    try:
        result = tf(labels)
    finally:
        random.randint = orig_randint
        random.uniform = orig_uniform

    # Clipped bbox: x1_clipped=100-0=100, ..., area=0 → should be removed
    n_remaining = len(result["instances"])
    assert n_remaining == 0, f"Expected 0 remaining instances, got {n_remaining}"
    print("[PASS] test_instance_removed_when_bbox_too_small")


def test_oov_keypoint_already_zero():
    """Keypoints with vis=0 before crop should remain vis=0 after crop."""
    img_h, img_w = 200, 200
    bboxes = np.array([[0, 0, 200, 200]], dtype=np.float32)
    # kp0: vis=0 (already OOV), kp1: vis=2 inside
    kpts = np.array([[[50, 50, 0], [80, 80, 2]]], dtype=np.float32)
    instances = make_instances(bboxes, kpts)
    labels = {
        "img": np.zeros((img_h, img_w, 3), dtype=np.uint8),
        "cls": np.array([0]),
        "instances": instances,
    }

    tf = RandomCropPose(crop_scale=(1.0, 1.0), p=1.0)  # full-image crop
    result = tf(labels)
    kpts_out = result["instances"].keypoints
    assert kpts_out[0, 0, 2] == 0, f"vis=0 kp should stay 0, got {kpts_out[0, 0, 2]}"
    assert kpts_out[0, 1, 2] == 2, f"vis=2 kp should stay 2, got {kpts_out[0, 1, 2]}"
    print("[PASS] test_oov_keypoint_already_zero")


def test_multiple_runs_deterministic():
    """Same seed → same result; different seed → may differ."""
    labels1, _, _ = make_labels(seed=42)
    labels2, _, _ = make_labels(seed=42)

    random.seed(0)
    tf = RandomCropPose(crop_scale=(0.5, 0.9), p=0.8)
    random.seed(7)
    r1 = tf(labels1)
    random.seed(7)
    r2 = tf(labels2)

    assert r1["img"].shape == r2["img"].shape, "Same seed should yield same crop shape"
    print("[PASS] test_multiple_runs_deterministic")


# ── Run all ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        test_image_shape,
        test_no_crop_when_p0,
        test_keypoints_inside_crop_unchanged,
        test_keypoints_outside_crop_set_to_oov,
        test_keypoint_coords_shifted,
        test_instance_removed_when_bbox_too_small,
        test_oov_keypoint_already_zero,
        test_multiple_runs_deterministic,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {t.__name__}: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
