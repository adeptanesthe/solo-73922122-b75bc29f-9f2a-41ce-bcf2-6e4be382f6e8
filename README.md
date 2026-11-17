# 分块文档内因果注意力核对 API

把多篇文档拼入同一批次（总长 N ≤ 512），在 **CPU + float64** 下核对一个
自定义分块注意力 autograd 算子与独立稠密参考实现的前向输出及 Q/K/V 梯度。

- 每个 token 只能读取**同一篇文档中位置不晚于自己**的 token
- 输出：`O = softmax(s · Q Kᵀ / 4) · V`
- Q/K/V 与上游梯度均为 16 维，float64
- 自定义算子前向、后向**均分块**计算，块长 `chunk_size ∈ [1, 32]`
- 不调用任何框架内置注意力算子，也不依赖内置 softmax 原语（online softmax 手写）
- 工作张量与跨前后向保存量规模 **O(N + C²)**，C 为块长；独立稠密参考不受此限制
- 容差：输出与 Q/K/V 梯度的最大绝对误差均 **≤ 1e-8**，否则 `passed=false` 并给出原因
- `POST /pack-plan`：调整拼接批次的文档顺序（保持篇内 token 顺序），
  最小化分块注意力前后向**实际执行的块计算量**，并用同一算子重排运行、
  映回原序后与原序计算核对（四项误差均 ≤ 1e-8 才返回可用排列）

## 目录结构

```
app/
  attention.py    # 自定义分块 autograd 算子（online softmax 前向 + 分块反向）
  reference.py    # 独立稠密参考（完整 N×N 掩码与注意力矩阵，不受内存约束）
  verification.py # 数据生成、双路计算与误差核对
  packing.py      # 块工作量度量与最优文档排列搜索（子集 DP，精确最优）
  packplan.py     # /pack-plan 编排：重排运行现有算子并映回原序核对
  schemas.py      # 请求/响应模型与输入校验
  scenarios.py    # 内置边界情形样例
  main.py         # FastAPI 入口
tests/            # pytest：正确性、gradcheck、边界情形、内存与源码审计、API
Dockerfile
docker-compose.yml
```

## 启动（Docker Compose）

```bash
docker compose up --build
# 服务监听 http://localhost:8000
# Swagger 文档：http://localhost:8000/docs
# OpenAPI：     http://localhost:8000/openapi.json
```

## 本地（无 Docker）运行

```bash
pip install -r requirements.txt   # torch 使用 CPU wheel 源
uvicorn app.main:app --host 0.0.0.0 --port 8000
pytest -q                         # 运行自测
```

## 接口

### `POST /verify`

请求体：

| 字段 | 类型 | 约束 |
| --- | --- | --- |
| `seed` | int | ≥ 0，决定 Q/K/V 与上游梯度的确定性生成 |
| `doc_lengths` | int[] | 正整数列表，总长 1..512 |
| `chunk_size` | int | 1..32 |
| `s` | float | 0 < s ≤ 1000 |

查询参数 `include_tensors`（默认 `true`）：是否回传完整输出/梯度张量。

curl 示例：

```bash
curl -X POST "http://localhost:8000/verify?include_tensors=false" \
  -H "Content-Type: application/json" \
  -d '{"seed":42,"doc_lengths":[7,11,5],"chunk_size":8,"s":1.0}'
```

响应关键字段：`passed`、`tolerance=1e-8`、`errors`（四项最大绝对误差）、
`finite_ok`、`failure_reasons`。核对失败是业务结果而非 HTTP 错误，仍返回 200，
通过 `passed=false` 与 `failure_reasons` 明确表达；参数非法返回 422。

### `POST /pack-plan`

在保持篇内 token 顺序的前提下重排文档，降低分块注意力前后向实际执行的
块计算量，并核对重排不改变计算结果。

请求体：

| 字段 | 类型 | 约束 |
| --- | --- | --- |
| `seed` | int | ≥ 0，决定 Q/K/V 与上游梯度的确定性生成（与 /verify 同一逻辑） |
| `doc_lengths` | int[] | 2~12 篇，正整数列表，总长 ≤ 96 |
| `chunk_size` | int | 1..32 |
| `s` | float | 0 < s ≤ 1000 |

行为：

1. **工作量度量**：算子实际执行的计算量 = 含有效注意力位置的
   (query 块, key 块) 对的尺寸乘积 `|qb|·|kb|` 之和（整块被掩码的块对
   会被算子跳过，不计入）。度量与算子共用同一套分块与掩码函数。
2. **最优排列**：子集动态规划（O(2^M·M·块数)，M ≤ 12）求精确最优；
   并列最优取字典序最小排列，结果确定。
3. **重排运行**：用同批 Q/K/V 与上游梯度按新顺序重排，运行现有分块
   算子（前向 + 反向），再把输出与 Q/K/V 梯度**映回原序**返回。
4. **核对**：映回结果与原序计算的四项 float64 最大绝对误差均须
   ≤ 1e-8 且均为有限值；**核对失败时 `passed=false`，且不返回可用
   排列**（`permutation`、`permuted_doc_lengths` 与张量字段均为 null）。

响应关键字段：`permutation`（新顺序，元素为原始文档索引）、
`work_original` / `work_packed` / `work_reduction`（原序/新序/下降的工作量）、
`errors`、`finite_ok`、`failure_reasons`、`output`/`grad_q`/`grad_k`/`grad_v`
（原序，可用 `include_tensors=false` 关闭）。

curl 示例：

```bash
curl -X POST "http://localhost:8000/pack-plan?include_tensors=false" \
  -H "Content-Type: application/json" \
  -d '{"seed":42,"doc_lengths":[10,1,10,1],"chunk_size":4,"s":1.0}'
# work_original=204, work_packed=164, permutation=[0,1,3,2], passed=true
```

### `GET /scenarios` / `POST /scenarios/run`

列出或运行内置边界情形；`POST /scenarios/run?name=<名称>` 只跑单个样例。

## 内置边界情形（题目要求）

| 名称 | 配置 | 覆盖点 |
| --- | --- | --- |
| `doc_cross_chunk` | 文档 7/11/5，块长 8，s=1 | 文档边界落在块内部，切块不改变归一化 |
| `fully_masked_block` | 单文档 16，块长 4，s=2.5 | 整块被因果屏蔽时安全跳过、无 NaN |
| `large_score` | 文档 33/31，块长 32，s=1000 | 大分数下 online softmax 前后向无溢出 |
| `combined_stress` | 文档 100×4+112（N=512），块长 7，s=999 | 三类情形叠加 + 块长不整除 |
| `minimal_single_token` | 两篇各 1 token，块长 32 | 最小批次与退化分块循环 |

```bash
curl -X POST "http://localhost:8000/scenarios/run"
```

## 实现要点

1. **掩码**：`valid(i,j) = (doc_i == doc_j) ∧ (pos_j ≤ pos_i)`，按块即时构造 C×C 布尔掩码。
2. **前向 online softmax**：逐 key 块维护每行的运行最大值 `m`、归一化分母 `d`
   与归一化输出 `O`。遇到新块时以 `exp(m_old − m_new)` 修正旧累积量，
   全程只做最大值平移，不保存 N×N 矩阵。
3. **整块屏蔽**：块内无有效位置时直接跳过；行级 `-inf` 通过 `isfinite` 判定，
   以 0 参与指数运算，杜绝 `−inf − −inf`、`0/0` 产生的 NaN。
4. **后向**：用保存的行最大值/分母在每个块上**重算**块内概率 P（仅 C×C），
   利用 `dP = P ⊙ (dO·Vᵀ − Σ_j P_ij dO_iᵀV_j)` 与 `dS = scale·dP` 分块累加
   `dQ/dK/dV`；保存量仅为 Q/K/V、doc_ids、行最大值、行分母、输出，均为 O(N)。
5. **数值稳定**：float64 全程；s=1000 时原始分数很大，最大值平移保证
   `exp` 输入 ≤ 0，前后向均为有限值。
6. **确定性**：同一 `seed` 生成完全相同的输入与结果。
7. **重排工作量**：对 query 块 qb，设其最左 token 属于起始位置为 p 的文档，
   则与 qb 含共同文档的 key 块恰好是从 p 所在块到 qb 的连续区间，qb 的
   工作量为 `|qb|·(q1 − ⌊p/C⌋·C)`。每个块的贡献在放置覆盖其左端点的文档
   时一次性计入，因此可用子集 DP 精确求出工作量最小的文档排列；
   重排只改变求和顺序，float64 下与原序结果的偏差远低于 1e-8。

## 自测结果

`pytest -q` 共 51 项全部通过，包含：11 组参数化前向+梯度核对（误差均 ≤ 1e-8，
实测在 1e-12 ~ 1e-16 量级）、`torch.autograd.gradcheck` 有限差分核对、
块长不变性、确定性、掩码语义、大分数有限性、保存张量线性规模审计、
禁用算子源码审计、工作量度量与独立枚举交叉验证、最优排列与暴力枚举对比、
/pack-plan 成功/失败/校验/确定性行为，以及全部 API 端点测试。
