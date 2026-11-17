"""文档重排（/pack-plan）测试：工作量度量、最优排列搜索与 API 行为。"""

from __future__ import annotations

import itertools
import random

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.packing import block_work_for_order, optimal_permutation

client = TestClient(app)


def py_work(lengths: list[int], chunk: int) -> int:
    """纯 Python 独立枚举：逐 token 判定有效位置（不依赖 torch 掩码）。"""
    n = sum(lengths)
    doc = []
    for i, length in enumerate(lengths):
        doc += [i] * length
    blocks = [(b, min(b + chunk, n)) for b in range(0, n, chunk)]
    total = 0
    for q0, q1 in blocks:
        for k0, k1 in blocks:
            if k0 >= q1:
                continue
            hit = any(
                doc[q] == doc[k] and k <= q
                for q in range(q0, q1)
                for k in range(k0, k1)
            )
            if hit:
                total += (q1 - q0) * (k1 - k0)
    return total


def brute_force_best(lengths: list[int], chunk: int) -> tuple[list[int], int]:
    """暴力枚举全部排列：返回（字典序最小的最优排列, 最小工作量）。"""
    best_work, best_perm = None, None
    for p in itertools.permutations(range(len(lengths))):
        work = py_work([lengths[i] for i in p], chunk)
        if best_work is None or work < best_work or (
            work == best_work and list(p) < best_perm
        ):
            best_work, best_perm = work, list(p)
    return best_perm, best_work


# ---------------------------------------------------------------- 度量与搜索


@pytest.mark.parametrize(
    "doc_lengths,chunk_size",
    [
        ([8], 1),
        ([8], 4),
        ([7, 11, 5], 8),
        ([16], 4),
        ([33, 31], 32),
        ([1, 1], 32),
        ([3, 1, 9, 2], 5),
        ([10, 1, 10, 1], 4),
        ([5, 5, 5, 5, 5], 7),
        ([96], 32),
        ([1] * 12, 3),
    ],
)
def test_work_metric_matches_independent_enumeration(doc_lengths, chunk_size):
    """torch 掩码枚举的工作量必须与纯 Python 逐 token 枚举一致。"""
    assert block_work_for_order(doc_lengths, chunk_size) == py_work(
        doc_lengths, chunk_size
    )


def test_dp_reported_work_matches_enumeration():
    """DP 报告的工作量必须等于其排列的实际枚举工作量。"""
    rng = random.Random(2026)
    for _ in range(40):
        m = rng.randint(2, 9)
        lengths = [rng.randint(1, 24) for _ in range(m)]
        while sum(lengths) > 96:
            lengths = [rng.randint(1, 24) for _ in range(m)]
        chunk = rng.randint(1, 32)
        perm, work = optimal_permutation(lengths, chunk)
        assert sorted(perm) == list(range(m))
        assert work == py_work([lengths[i] for i in perm], chunk)
        assert work == block_work_for_order([lengths[i] for i in perm], chunk)


def test_dp_matches_brute_force_optimum():
    """小规模上与暴力枚举对比：工作量最优且并列时字典序最小。"""
    rng = random.Random(7)
    for _ in range(25):
        m = rng.randint(2, 7)
        lengths = [rng.randint(1, 20) for _ in range(m)]
        while sum(lengths) > 96:
            lengths = [rng.randint(1, 20) for _ in range(m)]
        chunk = rng.randint(1, 32)
        perm, work = optimal_permutation(lengths, chunk)
        best_perm, best_work = brute_force_best(lengths, chunk)
        assert work == best_work
        assert perm == best_perm


def test_packing_actually_reduces_work():
    """交错长短文档时，重排后工作量严格下降（204 -> 164）。"""
    lengths = [10, 1, 10, 1]
    perm, work = optimal_permutation(lengths, 4)
    assert work == 164
    assert block_work_for_order(lengths, 4) == 204
    assert work < block_work_for_order(lengths, 4)


def test_optimal_permutation_deterministic():
    for _ in range(3):
        assert optimal_permutation([5, 9, 3, 7], 4) == optimal_permutation(
            [5, 9, 3, 7], 4
        )


# ---------------------------------------------------------------- API 行为


def test_pack_plan_in_openapi_and_docs():
    assert client.get("/docs").status_code == 200
    openapi = client.get("/openapi.json").json()
    assert "/pack-plan" in openapi["paths"]
    assert "/verify" in openapi["paths"]


def test_pack_plan_success_and_reduction():
    resp = client.post(
        "/pack-plan",
        json={"seed": 42, "doc_lengths": [10, 1, 10, 1], "chunk_size": 4, "s": 1.0},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["passed"] is True
    assert body["failure_reasons"] == []
    assert body["finite_ok"] is True
    assert body["permutation"] == [0, 1, 3, 2]
    assert body["permuted_doc_lengths"] == [10, 1, 1, 10]
    assert body["work_original"] == 204
    assert body["work_packed"] == 164
    assert body["work_reduction"] == 40
    for key in (
        "output_max_abs_err",
        "grad_q_max_abs_err",
        "grad_k_max_abs_err",
        "grad_v_max_abs_err",
    ):
        assert body["errors"][key] <= 1e-8
    # 映回原序的张量形状：N=22，d=16
    assert len(body["output"]) == 22
    assert len(body["output"][0]) == 16
    for name in ("grad_q", "grad_k", "grad_v"):
        assert len(body[name]) == 22


def test_pack_plan_output_matches_verify_original_order():
    """/pack-plan 映回原序的结果必须与 /verify 原序结果一致（≤1e-8）。"""
    payload = {"seed": 7, "doc_lengths": [6, 3, 9, 2, 4], "chunk_size": 5, "s": 3.0}
    plan = client.post("/pack-plan", json=payload).json()
    ref = client.post("/verify", json=payload).json()
    assert plan["passed"] and ref["passed"]
    for name in ("output", "grad_q", "grad_k", "grad_v"):
        for row_p, row_r in zip(plan[name], ref[name]):
            for a, b in zip(row_p, row_r):
                assert abs(a - b) <= 1e-8


def test_pack_plan_permutation_is_minimum_across_cases():
    """多组用例：返回排列的工作量不超过原序，且等于暴力最优。"""
    rng = random.Random(11)
    for _ in range(6):
        m = rng.randint(2, 6)
        lengths = [rng.randint(1, 18) for _ in range(m)]
        while sum(lengths) > 96:
            lengths = [rng.randint(1, 18) for _ in range(m)]
        chunk = rng.randint(1, 16)
        resp = client.post(
            "/pack-plan?include_tensors=false",
            json={
                "seed": rng.randint(0, 10_000),
                "doc_lengths": lengths,
                "chunk_size": chunk,
                "s": 1.5,
            },
        )
        body = resp.json()
        assert body["passed"] is True
        _, best_work = brute_force_best(lengths, chunk)
        assert body["work_packed"] == best_work
        assert body["work_packed"] <= body["work_original"]
        assert body["work_original"] == py_work(lengths, chunk)
        assert body["output"] is None  # include_tensors=false


def test_pack_plan_determinism():
    payload = {"seed": 3, "doc_lengths": [4, 8, 2, 6], "chunk_size": 3, "s": 2.0}
    a = client.post("/pack-plan?include_tensors=false", json=payload).json()
    b = client.post("/pack-plan?include_tensors=false", json=payload).json()
    assert a["permutation"] == b["permutation"]
    assert a["work_packed"] == b["work_packed"]


def test_pack_plan_max_size_case():
    """12 篇、总长 96 的上限情形也能快速给出通过结果。"""
    resp = client.post(
        "/pack-plan?include_tensors=false",
        json={
            "seed": 1,
            "doc_lengths": [8] * 12,
            "chunk_size": 7,
            "s": 1000.0,
        },
    )
    body = resp.json()
    assert body["passed"] is True
    assert body["num_docs"] == 12
    assert body["total_tokens"] == 96


def test_pack_plan_failure_returns_no_permutation(monkeypatch):
    """核对失败时必须明确失败且不返回可用排列。"""
    import app.packplan as packplan

    # 强制容差为负：任何误差都判定失败，模拟核对失败路径。
    monkeypatch.setattr(packplan, "TOLERANCE", -1.0)
    resp = client.post(
        "/pack-plan",
        json={"seed": 5, "doc_lengths": [4, 6, 3], "chunk_size": 2, "s": 1.0},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["passed"] is False
    assert body["failure_reasons"] != []
    assert body["permutation"] is None
    assert body["permuted_doc_lengths"] is None
    assert body["output"] is None
    assert body["grad_q"] is None


def test_pack_plan_validation_errors():
    base = {"seed": 0, "chunk_size": 8, "s": 1.0}
    # 文档数越界：1 篇与 13 篇
    for bad_lengths in ([8], [4] * 13):
        resp = client.post("/pack-plan", json={**base, "doc_lengths": bad_lengths})
        assert resp.status_code == 422
    # 总长超过 96
    resp = client.post("/pack-plan", json={**base, "doc_lengths": [50, 47]})
    assert resp.status_code == 422
    # 非正长度
    resp = client.post("/pack-plan", json={**base, "doc_lengths": [4, 0]})
    assert resp.status_code == 422
    # 块长越界
    for bad_chunk in (0, 33):
        resp = client.post(
            "/pack-plan",
            json={**base, "doc_lengths": [4, 4], "chunk_size": bad_chunk},
        )
        assert resp.status_code == 422
    # s 越界
    for bad_s in (0.0, 1000.1):
        resp = client.post(
            "/pack-plan", json={**base, "doc_lengths": [4, 4], "s": bad_s}
        )
        assert resp.status_code == 422


def test_verify_still_works_after_pack_plan_addition():
    resp = client.post(
        "/verify?include_tensors=false",
        json={"seed": 42, "doc_lengths": [7, 11, 5], "chunk_size": 8, "s": 1.0},
    )
    assert resp.status_code == 200
    assert resp.json()["passed"] is True
