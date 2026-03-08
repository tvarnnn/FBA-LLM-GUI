from __future__ import annotations

from pathlib import Path


def build_png_facts_block(image_path: Path) -> str:
    if image_path.suffix.lower() != ".png":
        raise ValueError(f"build_png_facts_block only supports .png, got: {image_path.suffix}")

    if not image_path.exists():
        raise FileNotFoundError(f"PNG not found: {image_path}")

    return (
        "FILE_TYPE: PNG\n"
        f"FILE_NAME: {image_path.name}\n"
        "IMAGE_SUPPORT_STATUS: STUB\n"
        "IMAGE_FINDINGS:\n"
        "- PNG uploaded successfully.\n"
        "- Visual analysis not yet implemented in this build.\n"
        "- Do not infer chart values or visual trends from this section yet.\n"
    )