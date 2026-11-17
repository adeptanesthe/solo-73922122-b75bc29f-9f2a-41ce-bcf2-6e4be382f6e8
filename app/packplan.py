"""POST /pack-plan 的编排逻辑：最优排列搜索、重排运行与映回核对。

流程
----
1. 用子集 DP 求使块工作量最小的文档排列（保持篇内 token 顺序）。
2. 按种子生成与原序完全相同的 Q/K/V 与上游梯度（与 /verify 同一生成逻辑）。
3. 原序运行现有分块算子，得到对照输出与 Q/K/V 梯度。
4. 按新顺序重排 Q/K/V 与上游梯度，运行同一算子，再把输出与梯度映回原序。
5. 核对映回结果与原序结果：四项 float64 最大绝对误差均须 ≤ 1e-8，
   且均为有限值；核对失败时 ``passed=false`` 且不返回可用排列。
"""

from __future__ import annotations

import torch

from .attention import chunked_causal_attention
from .packing import block_work_for_order, optimal_permutation
from .schemas import PackPlanRequest, PackPlanResponse, VerifyRequest
from .verification import TOLERANCE, generate_inputs, make_doc_ids


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.max(torch.abs(a - b)).item())


def _token_permutation(doc_lengths: list[int], permutation: list[int]) -> torch.Tensor:
    """新顺序下每个位置对应的原始 token 索引（篇内顺序保持）。"""
    starts: list[int] = []
    pos = 0
    for length in doc_lengths:
        starts.append(pos)
        pos += length
    order: list[int] = []
    for doc in permutation:
        order.extend(range(starts[doc], starts[doc] + doc_lengths[doc]))
    return torch.tensor(order, dtype=torch.int64)


def _run_operator(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    doc_ids: torch.Tensor,
    grad_upstream: torch.Tensor,
    scale: float,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """运行现有分块算子（前向 + 反向），返回输出与 Q/K/V 梯度。"""
    q_ = q.detach().clone().requires_grad_(True)
    k_ = k.detach().clone().requires_grad_(True)
    v_ = v.detach().clone().requires_grad_(True)
    out = chunked_causal_attention(q_, k_, v_, doc_ids, scale, chunk_size)
    gq, gk, gv = torch.autograd.grad(out, (q_, k_, v_), grad_outputs=grad_upstream)
    return out.detach(), gq, gk, gv


def run_pack_plan(req: PackPlanRequest, include_tensors: bool = True) -> PackPlanResponse:
    """搜索最优文档排列，重排运行现有算子并映回原序核对。"""
    failure_reasons: list[str] = []

    # --- 1) 最优排列与原序/新序工作量 ---
    permutation, work_packed_dp = optimal_permutation(req.doc_lengths, req.chunk_size)
    permuted_lengths = [req.doc_lengths[i] for i in permutation]
    work_original = block_work_for_order(req.doc_lengths, req.chunk_size)
    work_packed = block_work_for_order(permuted_lengths, req.chunk_size)
    # 防御性自检：DP 报告的工作量必须与实际枚举一致且不劣于原序。
    if work_packed != work_packed_dp or work_packed > work_original:
        failure_reasons.append(
            "内部一致性检查失败：最优排列工作量与枚举结果不一致"
        )

    # --- 2) 与原序完全相同的数据（与 /verify 同一生成逻辑） ---
    gen_req = VerifyRequest(
        seed=req.seed,
        doc_lengths=req.doc_lengths,
        chunk_size=req.chunk_size,
        s=req.s,
    )
    data = generate_inputs(gen_req)
    scale = float(req.s) / 4.0  # 题目规定 softmax(s QK^T / 4) V
    n = sum(req.doc_lengths)

    # --- 3) 原序运行（对照） ---
    out_o, gq_o, gk_o, gv_o = _run_operator(
        data.q, data.k, data.v, data.doc_ids, data.grad_upstream, scale, req.chunk_size
    )

    # --- 4) 重排运行：同批 Q/K/V 与上游梯度按新顺序重排 ---
    idx = _token_permutation(req.doc_lengths, permutation)
    inverse = torch.empty_like(idx)
    inverse[idx] = torch.arange(n, dtype=torch.int64)
    doc_ids_packed = make_doc_ids(permuted_lengths)
    out_p, gq_p, gk_p, gv_p = _run_operator(
        data.q[idx],
        data.k[idx],
        data.v[idx],
        doc_ids_packed,
        data.grad_upstream[idx],
        scale,
        req.chunk_size,
    )

    # --- 5) 映回原序 ---
    out_back, gq_back, gk_back, gv_back = (
        out_p[inverse],
        gq_p[inverse],
        gk_p[inverse],
        gv_p[inverse],
    )

    # --- 6) 与原序计算核对 ---
    errors = {
        "output_max_abs_err": _max_abs(out_back, out_o),
        "grad_q_max_abs_err": _max_abs(gq_back, gq_o),
        "grad_k_max_abs_err": _max_abs(gk_back, gk_o),
        "grad_v_max_abs_err": _max_abs(gv_back, gv_o),
    }
    packed_tensors = [out_back, gq_back, gk_back, gv_back]
    finite_ok = all(torch.isfinite(t).all().item() for t in packed_tensors)

    for name, value in errors.items():
        if value != value:  # NaN
            failure_reasons.append(f"{name} 为 NaN")
        elif value > TOLERANCE:
            failure_reasons.append(f"{name}={value:.3e} 超过容差 {TOLERANCE:.0e}")
    if not finite_ok:
        failure_reasons.append("重排运行的输出或梯度中存在 NaN/Inf")

    passed = not failure_reasons

    def to_list(t: torch.Tensor) -> list[list[float]]:
        return t.detach().tolist()

    # 核对失败时不返回可用排列：排列与张量字段一律置空。
    report_tensors = passed and include_tensors
    return PackPlanResponse(
        passed=passed,
        tolerance=TOLERANCE,
        permutation=permutation if passed else None,
        permuted_doc_lengths=permuted_lengths if passed else None,
        work_original=work_original,
        work_packed=work_packed,
        work_reduction=work_original - work_packed,
        errors=errors,
        scale_used=scale,
        total_tokens=n,
        num_docs=len(req.doc_lengths),
        chunk_size=req.chunk_size,
        s=req.s,
        seed=req.seed,
        finite_ok=finite_ok,
        failure_reasons=failure_reasons,
        output=to_list(out_back) if report_tensors else None,
        grad_q=to_list(gq_back) if report_tensors else None,
        grad_k=to_list(gk_back) if report_tensors else None,
        grad_v=to_list(gv_back) if report_tensors else None,
    )
