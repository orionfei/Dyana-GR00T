import numpy as np
import pytest

from gr00t.data.transform.concat import ConcatTransform
from gr00t.data.transform.dyana import (
    DyanaMotionTokenTransform,
    DyanaMotionViewTransform,
    DyanaTargetCropTransform,
)
from gr00t.experiment.data_config import load_data_config


LANGUAGE_KEY = "annotation.human.action.task_description"


def make_moving_square_clip(
    num_frames: int = 11,
    height: int = 64,
    width: int = 64,
    square_size: int = 6,
) -> np.ndarray:
    clip = np.zeros((num_frames, height, width, 3), dtype=np.uint8)
    for idx in range(num_frames):
        x0 = 4 + idx * 3
        y0 = 8 + idx * 2
        clip[idx, y0 : y0 + square_size, x0 : x0 + square_size] = 255
    return clip


@pytest.mark.parametrize(
    ("config_name", "expected_video_keys", "expect_motion_view", "expect_target_crop", "expect_token"),
    [
        ("dyana_lora_11f_18d", ["video.ego_view"], False, False, False),
        ("dyana_motion_token_11f_18d", ["video.ego_view"], False, False, True),
        ("dyana_motion_view_11f_18d", ["video.ego_view", "video.motion_view"], True, False, False),
        (
            "dyana_motion_view_token_11f_18d",
            ["video.ego_view", "video.motion_view"],
            True,
            False,
            True,
        ),
        ("dyana_target_crop_11f_18d", ["video.ego_view", "video.target_crop"], False, True, False),
        (
            "dyana_target_crop_token_11f_18d",
            ["video.ego_view", "video.target_crop"],
            False,
            True,
            True,
        ),
        (
            "dyana_motion_crop_11f_18d",
            ["video.ego_view", "video.motion_view", "video.target_crop"],
            True,
            True,
            False,
        ),
        (
            "dyana_motion_crop_token_11f_18d",
            ["video.ego_view", "video.motion_view", "video.target_crop"],
            True,
            True,
            True,
        ),
    ],
)
def test_dyana_data_configs_are_modular(
    config_name: str,
    expected_video_keys: list[str],
    expect_motion_view: bool,
    expect_target_crop: bool,
    expect_token: bool,
):
    data_config = load_data_config(config_name)
    modality_config = data_config.modality_config()
    transforms = data_config.transform()

    assert modality_config["video"].modality_keys == ["video.ego_view"]
    concat_transform = next(
        transform for transform in transforms.transforms if isinstance(transform, ConcatTransform)
    )
    assert concat_transform.video_concat_order == expected_video_keys
    assert any(isinstance(transform, DyanaMotionViewTransform) for transform in transforms.transforms) is expect_motion_view
    assert any(isinstance(transform, DyanaTargetCropTransform) for transform in transforms.transforms) is expect_target_crop
    assert any(isinstance(transform, DyanaMotionTokenTransform) for transform in transforms.transforms) is expect_token


def test_motion_token_transform_canonicalizes_known_motion_families():
    transform = DyanaMotionTokenTransform(apply_to=[LANGUAGE_KEY])
    data = {
        LANGUAGE_KEY: np.array(
            [
                "Grab the object in the video that is making a circular motion",
                "Grab the object in the video that is making a straight motion",
                "Grab the object in the video that is doing simple harmonic motion",
                "Grab the object in the video that is drifting randomly",
            ]
        )
    }

    processed = transform(data)

    assert processed[LANGUAGE_KEY] == [
        "motion=circular",
        "motion=linear",
        "motion=harmonic",
        "motion=unknown",
    ]


def test_motion_view_transform_detects_synthetic_motion():
    clip = make_moving_square_clip()
    transform = DyanaMotionViewTransform(diff_threshold=5, gaussian_blur_kernel=3)

    processed = transform({"video.ego_view": clip.copy()})
    motion_view = processed["video.motion_view"]

    assert motion_view.shape == clip.shape
    assert motion_view.dtype == np.uint8
    assert motion_view[0].sum() == 0
    assert motion_view[1:].sum() > 0
    assert transform.last_empty_sequences == 0
    assert transform.last_sequence_count == 1


def test_target_crop_transform_centers_the_moving_target():
    clip = make_moving_square_clip()
    motion_transform = DyanaMotionViewTransform(diff_threshold=5, gaussian_blur_kernel=3)
    crop_transform = DyanaTargetCropTransform(min_motion_pixels=4, crop_size=20, expand_ratio=1.2)

    processed = motion_transform({"video.ego_view": clip.copy()})
    processed = crop_transform(processed)
    cropped = processed["video.target_crop"]

    bright_pixels = np.argwhere(cropped[-1, ..., 0] > 64)
    assert bright_pixels.size > 0

    center_y = cropped.shape[1] / 2
    center_x = cropped.shape[2] / 2
    bright_center_y = float(bright_pixels[:, 0].mean())
    bright_center_x = float(bright_pixels[:, 1].mean())

    assert abs(bright_center_y - center_y) < 12
    assert abs(bright_center_x - center_x) < 12
    assert crop_transform.last_fallback_sequences == 0
    assert crop_transform.last_sequence_count == 1


def test_target_crop_transform_reports_fallback_on_static_clip():
    static_clip = np.zeros((11, 64, 64, 3), dtype=np.uint8)
    static_clip[:, 24:30, 10:16] = 255
    transform = DyanaTargetCropTransform(min_motion_pixels=4, crop_size=20, expand_ratio=1.2)

    processed = transform({"video.ego_view": static_clip.copy()})

    assert processed["video.target_crop"].shape == static_clip.shape
    assert transform.last_fallback_sequences == 1
    assert transform.last_sequence_count == 1
