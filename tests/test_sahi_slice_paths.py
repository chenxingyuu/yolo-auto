from __future__ import annotations

from pathlib import Path

from yolo_auto.control_api import _sahi_output_subdir_and_stem


def test_preserve_nested_path_under_split_directory(tmp_path: Path) -> None:
    dataset_root = tmp_path / "jieyangquanjing"
    train_dir = dataset_root / "images" / "train" / "jieyangquanjing" / "cam_2026-04-04_12-09-27"
    train_dir.mkdir(parents=True)
    img = train_dir / "frame_001774.jpg"
    img.write_bytes(b"")

    sub, stem = _sahi_output_subdir_and_stem(
        str(img),
        split="train",
        split_ref="images/train",
        dataset_root=str(dataset_root),
    )
    assert sub == "jieyangquanjing/cam_2026-04-04_12-09-27"
    assert stem == "frame_001774"


def test_flat_image_under_split_directory(tmp_path: Path) -> None:
    dataset_root = tmp_path / "ds"
    tdir = dataset_root / "images" / "train"
    tdir.mkdir(parents=True)
    img = tdir / "a.jpg"
    img.write_bytes(b"")

    sub, stem = _sahi_output_subdir_and_stem(
        str(img),
        split="train",
        split_ref="images/train",
        dataset_root=str(dataset_root),
    )
    assert sub == ""
    assert stem == "a"


def test_train_txt_paths_relative_to_dataset_root(tmp_path: Path) -> None:
    dataset_root = tmp_path / "ds"
    (dataset_root / "train.txt").write_text("")
    img = dataset_root / "images" / "train" / "proj" / "cam" / "frame.jpg"
    img.parent.mkdir(parents=True)
    img.write_bytes(b"")

    sub, stem = _sahi_output_subdir_and_stem(
        str(img),
        split="train",
        split_ref="train.txt",
        dataset_root=str(dataset_root),
    )
    assert sub == "proj/cam"
    assert stem == "frame"
