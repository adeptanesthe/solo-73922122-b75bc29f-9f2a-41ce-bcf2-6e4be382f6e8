"""FastAPI 入口：分块注意力核对服务。

端点
----
* ``GET  /health``           健康检查
* ``GET  /scenarios``        内置边界情形样例列表
* ``POST /verify``           单次核对（自定义分块算子 vs 独立稠密参考）
* ``POST /pack-plan``        文档重排规划：最小化块工作量并核对重排结果
* ``POST /scenarios/run``    运行全部或单个内置样例
* ``GET  /docs``             Swagger UI（自动生成）
"""

from __future__ import annotations

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

from .packplan import run_pack_plan
from .scenarios import SCENARIOS
from .schemas import PackPlanRequest, PackPlanResponse, VerifyRequest, VerifyResponse
from .verification import TOLERANCE, run_verification

app = FastAPI(
    title="分块文档内因果注意力核对 API",
    version="1.1.0",
    description=(
        "CPU + float64 环境下，对自定义分块（文档内因果）注意力 autograd 算子"
        "与独立稠密参考实现进行前向输出和 Q/K/V 梯度核对，容差为 1e-8。"
        "工作张量与跨前后向保存量规模为 O(N + 块长²)。"
        "另提供 /pack-plan：在保持篇内 token 顺序的前提下搜索使分块注意力"
        "实际块计算量最小的文档排列，并用同一算子重排运行、映回原序核对。"
    ),
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/scenarios")
def list_scenarios() -> dict[str, object]:
    return {
        "tolerance": TOLERANCE,
        "scenarios": [
            {"name": name, "description": s.description, "request": s.request.model_dump()}
            for name, s in SCENARIOS.items()
        ],
    }


@app.post("/verify", response_model=VerifyResponse)
def verify(
    req: VerifyRequest,
    include_tensors: bool = Query(
        default=True, description="是否在响应中回传完整输出/梯度张量"
    ),
) -> VerifyResponse:
    result = run_verification(req, include_tensors=include_tensors)
    # 核对失败属于业务结果而非服务错误：返回 200 并在体内明确 passed=false。
    return result


@app.post("/pack-plan", response_model=PackPlanResponse)
def pack_plan(
    req: PackPlanRequest,
    include_tensors: bool = Query(
        default=True, description="是否在响应中回传完整输出/梯度张量"
    ),
) -> PackPlanResponse:
    result = run_pack_plan(req, include_tensors=include_tensors)
    # 与 /verify 一致：核对失败是业务结果，返回 200 且 passed=false，
    # 此时不返回可用排列（permutation 为 null）。
    return result


@app.post("/scenarios/run")
def run_scenarios(
    name: str | None = Query(default=None, description="只运行指定样例；缺省运行全部"),
    include_tensors: bool = Query(default=False),
) -> JSONResponse:
    if name is not None and name not in SCENARIOS:
        return JSONResponse(
            status_code=404,
            content={
                "error": f"未知样例 {name!r}",
                "available": list(SCENARIOS.keys()),
            },
        )
    chosen = [name] if name else list(SCENARIOS.keys())
    results = {}
    all_passed = True
    for key in chosen:
        resp = run_verification(SCENARIOS[key].request, include_tensors=include_tensors)
        all_passed = all_passed and resp.passed
        results[key] = resp.model_dump()
    return JSONResponse(
        content={"all_passed": all_passed, "tolerance": TOLERANCE, "results": results}
    )
