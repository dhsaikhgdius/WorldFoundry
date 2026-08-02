"""Deterministic optimizer partitions shared by post-training learners."""

from __future__ import annotations


def balanced_contiguous_partitions(
    item_count: int,
    partition_count: int,
) -> tuple[tuple[int, int], ...]:
    """Split ``item_count`` ordered items into balanced, non-empty intervals."""

    if isinstance(item_count, bool) or not isinstance(item_count, int):
        raise TypeError("item_count must be an integer")
    if isinstance(partition_count, bool) or not isinstance(partition_count, int):
        raise TypeError("partition_count must be an integer")
    if item_count <= 0:
        raise ValueError("item_count must be positive")
    if partition_count <= 0:
        raise ValueError("partition_count must be positive")
    if partition_count > item_count:
        raise ValueError("partition_count cannot exceed item_count")

    base_size, larger_partitions = divmod(item_count, partition_count)
    start = 0
    partitions: list[tuple[int, int]] = []
    for index in range(partition_count):
        size = base_size + int(index < larger_partitions)
        end = start + size
        partitions.append((start, end))
        start = end
    return tuple(partitions)


__all__ = ["balanced_contiguous_partitions"]
