from pathlib import Path

import pytest

from region_matcher.utils import list_images


def test_list_images_recursively(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    (tmp_path / "a.jpg").touch()
    (nested / "b.png").touch()
    (nested / "ignore.txt").touch()

    assert [path.name for path in list_images(tmp_path)] == ["a.jpg", "b.png"]


def test_list_images_rejects_unsupported_file(tmp_path: Path) -> None:
    file_path = tmp_path / "notes.txt"
    file_path.touch()
    with pytest.raises(ValueError):
        list_images(file_path)
