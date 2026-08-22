# 自建 Graphiti 后端（替代 Zep Cloud）

MiroFish 默认使用 Zep Cloud 作为知识图谱与长期记忆后端。本文档说明如何切换到
**本地自建的 Graphiti + FalkorDB**，用于规避 Zep Cloud 的免费额度限制。

上游行为完全保留：不设置 `ZEP_BACKEND` 或设为 `cloud` 时，一切与原版一致。

---

## 架构

```
Flask 后端 ──► utils/zep.get_zep_client()
                    │
      ZEP_BACKEND=cloud ├─► zep_cloud.Zep（Zep Cloud）
   ZEP_BACKEND=graphiti └─► GraphitiAdapter
                                 │
                                 ├─ graphiti-core ──► FalkorDB（图存储）
                                 ├─ LLM failover ──► OpenAI 兼容端点（实体/关系抽取）
                                 └─ Embedder failover ──► Ollama 等（向量）
```

`GraphitiAdapter` 暴露与 Zep SDK **完全相同**的 `.graph` / `.batch` 命名空间，
因此 `graph_builder`、`zep_tools`、`zep_entity_reader`、`zep_graph_memory_updater`、
`oasis_profile_generator`、`utils/zep_paging` 全部零改动。

代码位置：`backend/app/services/graphiti_backend/`

| 文件 | 职责 |
|---|---|
| `adapter.py` | Zep 形状的门面 + 从 `Config` 构建单例 |
| `engine.py` | 同步封装 graphiti-core 的全部操作 |
| `namespaces.py` | `graph` / `node` / `edge` / `episode` / `batch` 命名空间 |
| `batch_store.py` | 本地 Batch API（落盘状态机，支持重启对账） |
| `ontology_store.py` | Zep 本体 → graphiti 类型，并持久化 |
| `graph_registry.py` | graph_id 簿记（graphiti 本身没有「图」这个对象） |
| `failover.py` | LLM / Embedder 的多端点 failover |
| `runtime.py` | 后台事件循环（Flask 是同步框架） |

---

## 前置条件

1. **FalkorDB 可达。** 若你使用 graphiti-mcp 的 compose，需要把 FalkorDB 端口映射出来：

   ```yaml
   ports:
     - "127.0.0.1:6380:6379"   # 6380 避开本机其他 redis
   ```

2. **一个 OpenAI 兼容的 LLM 端点**用于实体/关系抽取（可直接复用 `LLM_*` 配置）。

3. **一个 OpenAI 兼容的 embedding 端点**（例如 Ollama 的 `nomic-embed-text`）。

4. **后端依赖已安装**：`graphiti-core[falkordb]`（已写入 `pyproject.toml` /
   `requirements.txt`）。版本需与 graphiti-mcp 容器内一致，避免 FalkorDB
   schema / 索引不兼容：

   ```bash
   docker exec graphiti-mcp pip show graphiti-core
   ```

---

## 配置

在项目根目录 `.env` 中：

```bash
ZEP_BACKEND=graphiti

GRAPHITI_FALKORDB_URI=redis://127.0.0.1:6380
GRAPHITI_FALKORDB_DATABASE=mirofish

# 抽取用 LLM（留空则回落到 LLM_* 配置）
GRAPHITI_LLM_BASE_URL=http://127.0.0.1:8317/v1
GRAPHITI_LLM_API_KEY=<your-key>
GRAPHITI_LLM_MODEL_NAME=Kimi-k2.6
GRAPHITI_LLM_MODEL_FALLBACKS=claude-sonnet-4-6,gpt-5.4-mini

# Embedder：主端点优先，失败时回落备用端点
GRAPHITI_EMBEDDER_BASE_URL=http://127.0.0.1:11434/v1
GRAPHITI_EMBEDDER_FALLBACK_BASE_URL=
GRAPHITI_EMBEDDER_MODEL=nomic-embed-text
GRAPHITI_EMBEDDER_DIM=768
```

`Config.validate()` 会在 graphiti 模式下检查 `GRAPHITI_FALKORDB_URI`、
`GRAPHITI_EMBEDDER_BASE_URL` 与 LLM key；cloud 模式下才强制 `ZEP_API_KEY`。

> **Embedder 维度不可随意更换。** FalkorDB 中已有的向量是按当前模型的维度存的，
> 换模型（例如 `nomic-embed-text` 768 维 → `qwen3-embedding`）会让旧数据搜不到。

### Docker 部署

容器内不能用 `127.0.0.1`。把 URI 换成对应的容器网络地址，例如
`redis://falkordb:6379`、`http://cli-proxy-api:8317/v1`，并确保 MiroFish 容器
加入了同一个 Docker network。

---

## 数据隔离

两层隔离：

1. **FalkorDB 图键**：`GRAPHITI_FALKORDB_DATABASE`（默认 `mirofish`）与
   graphiti-mcp 使用的 `main` 是不同的图键，索引与数据完全分开。
2. **group_id**：MiroFish 的每个图谱 ID（`mirofish_<hex>`）直接作为 `group_id`，
   项目之间互不干扰。

因此可以安全共用同一个 FalkorDB 实例。

图谱状态（本体、批次、graph_id 簿记）落盘在
`backend/uploads/graph_state/`。

---

## 与 Zep Cloud 的行为差异

| 方面 | Zep Cloud | 自建 Graphiti |
|---|---|---|
| 本体 | 服务端按图谱强制 | 每次 `add_episode` 携带类型；适配器持久化本体后自动带上 |
| 摄取 | 异步，需轮询 `episode.processed` | 同步完成；已入库的 episode 直接返回 `processed=True`，尚在批次队列中的返回 `False` |
| Batch API | 服务端对象，带服务端重试 | 本地状态机，落盘；失败条目以 `partial` 暴露，无服务端重试 |
| `batch.add` 返回的 episode uuid | 服务端真实 uuid | 适配器发的轮询句柄（入库前就要给出）；真实 uuid 记在批次状态里 |
| 分页游标 | 不透明游标 | 行偏移量游标（`ORDER BY uuid` + `SKIP/LIMIT`），`utils/zep_paging` 无需改动 |
| reranker | `cross_encoder` / `rrf` / `mmr` / … | 仅本地重排：`rrf`（默认）、`mmr`、`episode_mentions`；`cross_encoder` 会降级为 `rrf`（它对每条候选都要一次 LLM 调用，成本不划算） |
| episode metadata | 服务端字段 | graphiti 无对应字段，折叠进 `source_description` |
| 非文本 episode | 支持 `json` / `message` | 仅 `text`，其它类型会报错 |

---

## 已绕开的 graphiti-core / FalkorDB 坑

实测（graphiti-core 0.28.2 + FalkorDB）过程中发现并在适配器里绕开的问题，
升级 graphiti-core 时值得回头确认：

1. **`group_id` 就是 FalkorDB 的图键。** `add_episode` 会把 driver clone 到
   以 `group_id` 命名的图键上，并替换 `Graphiti.driver`。适配器因此为每个
   graph 维护一个 driver，并在所有读操作显式传入，不依赖「上一次写到哪」。
   索引也要按图键分别建。
2. **`get_by_group_ids(uuid_cursor=...)` 在 FalkorDB 上不生效**，游标条件没有
   进到查询里，每页都返回同样的行。适配器改用 `ORDER BY uuid` + `SKIP/LIMIT`。
3. **`add_episode(uuid=...)` 是「更新已存在的 episode」**，传新 uuid 会
   `NodeNotFound`。所以批次的 episode uuid 是适配器自己发的句柄。
4. **FalkorDB 拒绝非基本类型的属性值。** 模型偶尔会回 JSON Schema 外壳
   （`{"properties": {...}}`）或嵌套对象，graphiti 会原样写库并报
   `Property values can only be of primitive types`。
   `json_llm_client.normalize_to_schema()` 会拆掉外壳、并把声明为字符串的
   字段里的嵌套值序列化成 JSON 文本。
5. **代理常忽略 `response_format`。** `JsonTolerantOpenAIClient` 把 schema 也
   写进 prompt，并容忍 Markdown 代码围栏与前后文字。
6. **falkordb 1.6.2 与 redis 8.x 不兼容**（`himport_registry`）。依赖锁到
   `falkordb==1.6.0` + `redis<8`，与 graphiti-mcp 容器一致。

---

## 验证

```bash
cd backend

# 1. 单元测试（不需要任何外部服务）
uv run pytest tests/test_graphiti_adapter.py tests/test_llm_failover.py

# 2. 全量测试
uv run pytest tests

# 3. 对真实 FalkorDB / LLM / embedder 的实测
#    建图 → 设本体 → 批量导入 → 轮询 → 分页 → 搜索 → 节点详情 → 删图
uv run python scripts/validate_graphiti_local_integration.py
```

实测脚本默认在结束时删除临时图谱；加 `--keep-graph` 可保留以便人工检查，
加 `--page-size N` 可调整分页大小（默认 2，用来确实走到游标翻页）。

端到端验证：`ZEP_BACKEND=graphiti` 启动后端 → 上传文档建图 → 确认节点/边落入
FalkorDB → 跑一轮短模拟 → 确认 memory updater 写入与 report 检索可用。
