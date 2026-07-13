"""OpenCV post-processing: sharpen, smart-crop, webp export."""
from __future__ import annotations

from pathlib import Path


def postprocess(
    img_path: Path,
    sharpen: bool = True,
    auto_crop: bool = False,
    web_sizes: list[tuple[int, int]] | None = None,
    webp: bool = True,
) -> list[Path]:
    """Sharpen, crop, resize, and export a generated image. Returns output paths."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return [img_path]

    img = cv2.imread(str(img_path))
    if img is None:
        return [img_path]

    if sharpen:
        blur = cv2.GaussianBlur(img, (0, 0), 3.0)
        img  = cv2.addWeighted(img, 1.5, blur, -0.5, 0)

    if auto_crop:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
        coords = cv2.findNonZero(thresh)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            pad = 10
            x = max(0, x - pad); y = max(0, y - pad)
            w = min(img.shape[1] - x, w + 2 * pad)
            h = min(img.shape[0] - y, h + 2 * pad)
            img = img[y:y + h, x:x + w]

    outputs = []
    stem    = img_path.stem
    out_dir = img_path.parent

    if webp:
        out_path = out_dir / f"{stem}.webp"
        cv2.imwrite(str(out_path), img, [cv2.IMWRITE_WEBP_QUALITY, 92])
        outputs.append(out_path)
    else:
        cv2.imwrite(str(img_path), img)
        outputs.append(img_path)

    if web_sizes:
        for w, h in web_sizes:
            resized   = cv2.resize(img, (w, h), interpolation=cv2.INTER_LANCZOS4)
            suffix    = ".webp" if webp else ".png"
            out_path  = out_dir / f"{stem}_{w}x{h}{suffix}"
            if webp:
                cv2.imwrite(str(out_path), resized, [cv2.IMWRITE_WEBP_QUALITY, 88])
            else:
                cv2.imwrite(str(out_path), resized)
            outputs.append(out_path)

    return outputs
