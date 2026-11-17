"""请求/响应模型与输入校验。"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

MAX_TOTAL_TOKENS = 512
MAX_CHUNK_SIZE = 32
MIN_CHUNK_SIZE = 1
MAX_S = 1000.0
HEAD_DIM = 16

# /pack-plan 专属约束：2~12 篇文档，总长不超过 96。
PACK_MIN_DOCS = 2
PACK_MAX_DOCS = 12
PACK_MAX_TOTAL_TOKENS = 96


def _check_positive_doc_lengths(value: list[int], max_total: int) -> list[int]:
    if any(not isinstance(x, int) or x <= 0 for x in value):
        raise ValueError("doc_lengths 必须为正整数列表")
    total = sum(value)
    if total == 0 or total > max_total:
        raise ValueError(f"文档总长度必须在 1..{max_total} 之间，当前为 {total}")
    return value


def _check_s_finite_value(value: float) -> float:
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError("s 必须为有限正数")
    return value


class VerifyRequest(BaseModel):
    """/verify 请求体。"""

    seed: int = Field(..., ge=0, description="随机种子（非负整数）")
    doc_lengths: list[int] = Field(
        ..., min_length=1, description="各文档长度（正整数），总长不超过 512"
    )
    chunk_size: int = Field(..., ge=MIN_CHUNK_SIZE, le=MAX_CHUNK_SIZE)
    s: float = Field(..., gt=0.0, le=MAX_S, description="注意力缩放系数")

    @field_validator("doc_lengths")
    @classmethod
    def _check_doc_lengths(cls, value: list[int]) -> list[int]:
        return _check_positive_doc_lengths(value, MAX_TOTAL_TOKENS)

    @field_validator("s")
    @classmethod
    def _check_s_finite(cls, value: float) -> float:
        return _check_s_finite_value(value)


class ErrorMetrics(BaseModel):
    output_max_abs_err: float
    grad_q_max_abs_err: float
    grad_k_max_abs_err: float
    grad_v_max_abs_err: float


class VerifyResponse(BaseModel):
    passed: bool
    tolerance: float
    errors: ErrorMetrics
    scale_used: float = Field(..., description="实际进入 softmax 的系数 s/4")
    total_tokens: int
    num_docs: int
    chunk_size: int
    s: float
    seed: int
    finite_ok: bool = Field(..., description="所有输出与梯度是否均为有限值")
    failure_reasons: list[str]
    output: list[list[float]] | None = None
    grad_q: list[list[float]] | None = None
    grad_k: list[list[float]] | None = None
    grad_v: list[list[float]] | None = None


class ScenarioSummary(BaseModel):
    name: str
    description: str
    request: VerifyRequest


class PackPlanRequest(BaseModel):
    """/pack-plan 请求体：在保持篇内 token 顺序的前提下重排文档。"""

    seed: int = Field(..., ge=0, description="随机种子（非负整数）")
    doc_lengths: list[int] = Field(
        ...,
        min_length=PACK_MIN_DOCS,
        max_length=PACK_MAX_DOCS,
        description="2~12 篇文档的长度（正整数），总长不超过 96",
    )
    chunk_size: int = Field(..., ge=MIN_CHUNK_SIZE, le=MAX_CHUNK_SIZE)
    s: float = Field(..., gt=0.0, le=MAX_S, description="注意力缩放系数")

    @field_validator("doc_lengths")
    @classmethod
    def _check_doc_lengths(cls, value: list[int]) -> list[int]:
        return _check_positive_doc_lengths(value, PACK_MAX_TOTAL_TOKENS)

    @field_validator("s")
    @classmethod
    def _check_s_finite(cls, value: float) -> float:
        return _check_s_finite_value(value)


class PackPlanResponse(BaseModel):
    """/pack-plan 响应体。

    核对失败（``passed=false``）时不返回可用排列：``permutation`` 与
    ``permuted_doc_lengths`` 为 null，张量字段也为 null。
    """

    passed: bool
    tolerance: float
    permutation: list[int] | None = Field(
        ..., description="工作量最小的新顺序（元素为原始文档索引）；核对失败为 null"
    )
    permuted_doc_lengths: list[int] | None = Field(
        ..., description="新顺序对应的文档长度列表；核对失败为 null"
    )
    work_original: int = Field(..., description="原序的块工作量")
    work_packed: int | None = Field(..., description="最优排列的块工作量")
    work_reduction: int | None = Field(..., description="工作量下降量（原序 − 新序）")
    errors: ErrorMetrics
    scale_used: float = Field(..., description="实际进入 softmax 的系数 s/4")
    total_tokens: int
    num_docs: int
    chunk_size: int
    s: float
    seed: int
    finite_ok: bool = Field(..., description="重排运行的输出与梯度是否均为有限值")
    failure_reasons: list[str]
    output: list[list[float]] | None = None
    grad_q: list[list[float]] | None = None
    grad_k: list[list[float]] | None = None
    grad_v: list[list[float]] | None = None
