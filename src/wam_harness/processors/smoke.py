from __future__ import annotations


def rgb_image(
    color: list[int],
    *,
    width: int = 8,
    height: int = 8,
) -> list[list[list[int]]]:
    return [[list(color) for _ in range(width)] for _ in range(height)]


def zero_vector(dim: object, *, default: int = 1) -> list[float]:
    size = default if dim is None else int(dim)
    return [0.0] * size
