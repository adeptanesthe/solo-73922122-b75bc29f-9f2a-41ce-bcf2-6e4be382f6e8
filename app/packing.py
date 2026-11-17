"""拼接批次的文档重排：工作量度量与最优排列搜索。

背景
----
分块注意力把多篇文档拼接成一条序列后按固定块长切块处理。算子实际执行的
计算量是“含有效注意力位置的 (query 块, key 块) 对的尺寸乘积之和”——
整块被掩码（两块间没有同文档 token 对）的块对会被算子跳过。调整文档顺序
不改变篇内 token 顺序与注意力语义，但会改变块边界落在哪些文档上，从而
改变前后向实际执行的块计算量。

本模块提供：

1. ``block_work_for_order``：按定义直接枚举块对，计算给定文档顺序下算子
   实际执行的工作量。与算子共用同一套分块（``_split``）与块掩码
   （``_block_valid``）函数，保证度量与算子实际跳过的块对完全一致。
2. ``optimal_permutation``：在保持篇内 token 顺序的前提下，搜索使工作量
   最小的文档排列。文档数 M ≤ 12，用子集动态规划（O(2^M · M · 块数)）
   求精确最优；并列最优时取字典序最小的排列，保证结果确定。

DP 增量代价基于以下观察：对某个 query 块 qb，设其最左 token 属于文档 d
（d 的起始位置为 p），则与 qb 含有共同文档的 key 块恰好是从 p 所在块到
qb 的连续区间（文档是连续段，介于其间的每个块都被某个出现在 qb 中的
文档覆盖），故 qb 的工作量为 |qb| · (q1 − (p // C) · C)。每个块的贡献在
放置覆盖其左端点的文档时一次性计入，总代价即精确的总工作量。
"""

from __future__ import annotations

from .attention import _block_valid, _split
from .verification import make_doc_ids


def block_work_for_order(doc_lengths: list[int], chunk_size: int) -> int:
    """给定文档顺序，按定义计算算子实际执行的块工作量。

    工作量 = 所有“含有效注意力位置的 (query 块, key 块) 对”的尺寸乘积
    |qb| · |kb| 之和。k0 >= q1 的块对不可能含有效位置（key 全在未来），
    其余块对用与算子相同的掩码函数判定是否被跳过。
    """
    n = sum(doc_lengths)
    blocks = _split(n, chunk_size)
    doc_ids = make_doc_ids(doc_lengths)
    total = 0
    for q0, q1 in blocks:
        for k0, k1 in blocks:
            if k0 >= q1:
                # key 块整体在 query 块之后：块内不存在 k <= q 的位置。
                continue
            if k0 == q0 or bool(_block_valid(doc_ids, q0, q1, k0, k1).any()):
                total += (q1 - q0) * (k1 - k0)
    return total


def _edge_cost(
    pos: int, length: int, chunk_size: int, blocks: list[tuple[int, int]]
) -> int:
    """把长度为 length 的文档放到位置 pos 时产生的工作量增量。

    该文档覆盖左端点 q0 ∈ [pos, pos+length) 的每个 query 块 qb，贡献
    |qb| · (q1 − (pos // C) · C)；其余块的贡献由其他文档放置时计入。
    """
    base = (pos // chunk_size) * chunk_size
    end = pos + length
    cost = 0
    for q0, q1 in blocks:
        if q0 >= end:
            break  # 块按起点升序，之后的块左端点不在本文档内
        if q0 >= pos:
            cost += (q1 - q0) * (q1 - base)
    return cost


def optimal_permutation(
    doc_lengths: list[int], chunk_size: int
) -> tuple[list[int], int]:
    """搜索使块工作量最小的文档排列（保持篇内 token 顺序）。

    返回 ``(permutation, work)``：``permutation[i]`` 为放在新顺序第 i 位的
    原始文档索引，``work`` 为该排列的工作量（精确最优）。并列最优时取
    字典序最小的排列，保证同一输入永远得到同一结果。

    子集 DP：状态为已放置文档集合 S（其总长度即下一个放置位置），
    dp[S] = 放置剩余文档的最小附加工作量，按位Mask从大到小递推。
    """
    m = len(doc_lengths)
    n = sum(doc_lengths)
    blocks = _split(n, chunk_size)
    full = (1 << m) - 1

    # dp[S]：已放置集合为 S 时，放完剩余文档所需的最小附加工作量。
    dp = [0] * (1 << m)
    for state in range(full - 1, -1, -1):
        pos = 0
        for d in range(m):
            if state >> d & 1:
                pos += doc_lengths[d]
        best = None
        for d in range(m):
            if state >> d & 1:
                continue
            total = _edge_cost(pos, doc_lengths[d], chunk_size, blocks) + dp[
                state | (1 << d)
            ]
            if best is None or total < best:
                best = total
        dp[state] = best if best is not None else 0

    # 依 dp 值重建排列：每步取能达到最优的最小文档索引 → 字典序最小。
    permutation: list[int] = []
    state = 0
    pos = 0
    while state != full:
        for d in range(m):
            if state >> d & 1:
                continue
            total = _edge_cost(pos, doc_lengths[d], chunk_size, blocks) + dp[
                state | (1 << d)
            ]
            if total == dp[state]:
                permutation.append(d)
                state |= 1 << d
                pos += doc_lengths[d]
                break
    return permutation, dp[0]


__all__ = ["block_work_for_order", "optimal_permutation"]
