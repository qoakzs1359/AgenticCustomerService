# 电商客服智能体 (customer_service_agent)

一个面向国内电商平台（淘宝 / 京东 / 拼多多 / 抖音）的售前 / 售后 / 物流一站式 AI 客服。
采用 **ReAct（Reason + Act）** 架构，支持 **Tool Calling** 查询订单/物流/商品，内置 **RAG** 向量检索知识库，处理不了自动转人工。

> MVP 阶段已完成：本地 Web 聊天界面 + Mock 平台 API + DeepSeek LLM + Qdrant RAG + MySQL/Redis 持久化 + 转人工 WebSocket 接管。
> 真实平台对接（淘宝/京东/拼多多/抖音）留作后续扩展。

---

## 技术架构

```
┌───────────────────────────────────────────────────────────────────┐
│                        Web 前端层                                  │
│  ┌──────────────────────┐   ┌──────────────────────────────────┐  │
│  │  用户聊天界面 (千牛风) │   │  人工客服后台（队列 + 实时对话）    │  │
│  │  Vue3 + Element Plus  │   │  Vue3 + Element Plus             │  │
│  └──────────┬───────────┘   └──────────────┬───────────────────┘  │
│             │  WebSocket                     │  WebSocket            │
└─────────────┼───────────────────────────────┼──────────────────────┘
              │                               │
              ▼                               ▼
┌───────────────────────────────────────────────────────────────────┐
│                     FastAPI 后端                                    │
│                                                                     │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────────┐   │
│  │ WebSocket    │   │ REST Mock    │   │ 会话管理 / 状态机      │   │
│  │ 网关         │   │ 平台接口     │   │ (Session + Transfer)  │   │
│  └──────┬───────┘   └──────┬───────┘   └──────────┬───────────┘   │
│         │                  │                      │                │
│         ▼                  ▼                      ▼                │
│  ┌─────────────────────────────────────────────────────────┐      │
│  │              Agent Core (ReAct Loop)                     │      │
│  │                                                           │      │
│  │  用户消息                                                  │      │
│  │    │                                                       │      │
│  │    ▼                                                       │      │
│  │  ┌──────────┐   ┌───────────────┐   ┌────────────────┐   │      │
│  │  │ RAG 检索  │──▶│ DeepSeek LLM  │──▶│ Tool Calling   │   │      │
│  │  │ Qdrant   │   │ deepseek-chat │   │ 订单/物流/商品   │   │      │
│  │  │ bge-zh   │   │ (OpenAI 协议) │   │ (Mock REST API) │   │      │
│  │  └──────────┘   └───────┬───────┘   └────────┬───────┘   │      │
│  │                          │                    │            │      │
│  │                          ▼                    ▼            │      │
│  │                   组装回复消息 ◀── Tool 返回结果             │      │
│  │                          │                                  │      │
│  │                          ▼                                  │      │
│  │              转人工关键词（人工 / 转人工）？                  │      │
│  │               ├── No → 回复用户                              │      │
│  │               └── Yes → Redis 队列通知 → 客服接管           │      │
│  └─────────────────────────────────────────────────────────┘      │
│                                                                     │
│  ┌──────────┐   ┌──────────┐   ┌──────────────────────────┐      │
│  │  MySQL   │   │  Redis   │   │  Qdrant                   │      │
│  │ 会话持久化 │   │ 中期记忆  │   │  product_faqs (15 条)   │      │
│  │ SQLAlchemy│   │ ACL 用户  │   │  bge-base-zh-v1.5 (768d)│      │
│  └──────────┘   └──────────┘   └──────────────────────────┘      │
└───────────────────────────────────────────────────────────────────┘
```

### 核心依赖

| 分类 | 组件 | 版本 | 说明 |
|------|------|------|------|
| Web 框架 | FastAPI | ≥ 0.110 | 异步 Web 框架 |
| Web 服务器 | uvicorn | ≥ 0.29 | ASGI 服务器 |
| ORM | SQLAlchemy | ≥ 2.0 | 异步 ORM |
| MySQL 驱动 | aiomysql | ≥ 0.2 | 异步 MySQL 连接 |
| Redis | redis-py | ≥ 5.0 | 支持 ACL 用户认证 |
| 向量库 | qdrant-client | ≥ 1.10 | Qdrant 官方 Python SDK |
| Embedding | sentence-transformers | ≥ 2.2 | `BAAI/bge-base-zh-v1.5`（768 维，中文） |
| LLM | openai SDK | ≥ 1.45 | 调用 DeepSeek（兼容 OpenAI 协议）`deepseek-chat` 模型 |
| 配置 | pydantic-settings | ≥ 2.4 | 类型安全的 `.env` 加载 |
| 日志 | loguru | ≥ 0.7 | 零配置日志 |
| 前端 | Vue 3 + Vite | Vue 3.5 / Vite 8 | SPA 框架 |
| UI | Element Plus | ≥ 2.14 | 后台管理风格 UI 组件 |
| 路由 | Vue Router 4 | ≥ 4.6 | `/` 用户聊天 / `/agent` 客服后台 |
| 状态 | Pinia | ≥ 4.1 | 轻量状态管理 |

### 工具函数（Tool Calling）

| Tool | 输入 | 来源 | 说明 |
|------|------|------|------|
| `query_order` | `order_id: str` | `GET /api/mock/orders/{order_id}` | 查订单状态、金额、商品 |
| `query_logistics` | `order_id: str` | `GET /api/mock/logistics/{order_id}` | 查物流轨迹、快递公司 |
| `query_product` | `product_id: str` | `GET /api/mock/products/{product_id}` | 查商品规格、库存、发货时效 |

### RAG 向量库

- **Collection 名称**：`product_faqs`
- **向量维度**：768
- **距离度量**：COSINE
- **Embedding 模型**：`BAAI/bge-base-zh-v1.5`（HuggingFace 自动下载，首次约 400MB）
- **原始数据**：`data/faqs.csv`（可自行扩展商品、FAQ、政策条目）

---

## 目录结构

```
customer_service_agent/
├── .env.example              # 环境变量模板（复制为 .env 使用）
├── .env                      # 你的实际配置（不入库）
├── requirements.txt          # Python 依赖
├── main.py                   # FastAPI 入口
├── data/
│   └── faqs.csv              # RAG 知识库原始数据，初始化时灌入 Qdrant
├── logs/                     # 运行时日志（loguru 自动生成）
├── src/
│   ├── api/
│   │   ├── websocket.py      # WebSocket 网关（用户 + 客服双通道）
│   │   └── mock.py           # 平台 Mock REST API（订单/物流/商品）
│   ├── agent/
│   │   ├── customer_service_agent.py  # ReAct 智能体核心
│   │   └── prompt.py         # System Prompt 模板
│   ├── core/
│   │   ├── config.py         # pydantic-settings 类型安全配置
│   │   └── logging.py        # loguru 日志初始化
│   ├── rag/
│   │   ├── embedding.py      # bge-base-zh-v1.5 封装（带 Mock 降级）
│   │   ├── vector_store.py   # Qdrant 封装（upsert / search）
│   │   ├── retriever.py      # 检索逻辑
│   │   └── init_faqs.py      # FAQ 入库脚本
│   ├── services/
│   │   ├── mysql.py          # SQLAlchemy ORM + 会话持久化 CRUD
│   │   └── memory.py         # Redis 中期记忆（ACL 支持）
│   └── tools/
│       └── mock_tools.py     # 三个 Tool 实现 + 注册表
├── web/                      # 前端静态产物
│   ├── dist/                 # Vite 构建产物（FastAPI 直接挂载）
│   ├── user.html             # 原生 HTML 兜底
│   └── agent.html
└── web-vue/                  # Vue3 前端工程
    ├── vite.config.js        # 开发代理 /api → FastAPI 8000，输出到 web/dist
    ├── package.json
    └── src/
        ├── main.js           # Element Plus + Vue Router + Pinia
        ├── App.vue
        ├── router/index.js
        ├── composables/useSocket.js
        └── views/
            ├── UserChat.vue      # 用户聊天界面（千牛风格）
            └── AgentConsole.vue  # 客服后台
```

---

## 环境要求

| 组件 | 版本要求 | 说明 |
|------|---------|------|
| Python | ≥ 3.10 | 推荐 3.12 |
| Node.js | ≥ 18 | 仅前端开发时需要 |
| MySQL | ≥ 8.0 | 需提前创建数据库（见下方步骤） |
| Redis | ≥ 5.0 | 6.0+ ACL 用户需填写 `REDIS_USER` |
| Qdrant | ≥ 1.7 | 默认端口 6333 |
| DeepSeek API Key | — | [https://platform.deepseek.com](https://platform.deepseek.com) 申请 |

---

## 快速开始

### 1. 克隆项目

```bash
git clone <repo-url>
cd customer_service_agent
```

### 2. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

> ⚠️ **Windows 用户注意**：`sentence-transformers` 依赖 `torch`，需用 PyTorch 官方源安装：
> ```bash
> # 方案 A：装到系统 site-packages
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> pip install sentence-transformers
> 
> # 方案 B（推荐）：装到工作区 .pylibs，避免沙箱/权限问题
> pip install torch --index-url https://download.pytorch.org/whl/cpu --target .pylibs
> pip install sentence-transformers -i https://pypi.org/simple/ --target .pylibs --no-compile --upgrade
> ```
> 如果用 `--target .pylibs`，`main.py` 已自动把 `.pylibs` 加进 `sys.path`。

### 3. 配置环境变量

```bash
cp .env.example .env
# 然后编辑 .env，填入你的真实连接信息
```

关键配置项：

```ini
# DeepSeek（必填，否则智能体走 Mock 模式）
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxx

# MySQL（库需提前创建，见下方步骤）
MYSQL_HOST=127.0.0.1
MYSQL_USER=root
MYSQL_PASSWORD=你的密码
MYSQL_DATABASE=customer_service

# Redis（ACL 用户请填 REDIS_USER，留空=default）
REDIS_HOST=127.0.0.1
REDIS_USER=ai_study
REDIS_PASSWORD=你的密码

# Qdrant
QDRANT_HOST=127.0.0.1
```

### 4. 创建 MySQL 数据库

```bash
mysql -u root -p -e "CREATE DATABASE IF NOT EXISTS customer_service DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
```

表由 FastAPI 启动时自动通过 SQLAlchemy `metadata.create_all` 建立。

### 5. 初始化 Qdrant 知识库

首次运行会从 HuggingFace 下载 `BAAI/bge-base-zh-v1.5` 模型（约 400MB），之后本地缓存。

```bash
python -m src.rag.init_faqs
```

成功输出：`✅ 完成！Qdrant 中已有 15 条向量`

### 6. 启动 FastAPI 后端

```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

启动成功后自动挂载 Vue3 前端产物：

| 页面 | 地址 |
|------|------|
| 用户聊天界面 | http://localhost:8000/ |
| 人工客服后台 | http://localhost:8000/#/agent |
| 健康检查 | http://localhost:8000/health |
| Mock API 示例 | http://localhost:8000/api/mock/orders/DD20261022001 |

### 7.（可选）前端开发模式

如果需要改 Vue 前端：

```bash
cd web-vue
npm install
npm run dev      # http://localhost:5173（Vite 热更新 + 自动代理 /api → FastAPI 8000）
npm run build    # 构建生产产物到 ../web/dist
```

---

## 测试流程

1. 打开浏览器，同时访问 **`/`**（用户聊天）和 **`/#/agent`**（客服后台）
2. 在用户界面点快捷按钮或输入问题：
   - `📦 查订单` → 智能体 Tool Calling 查订单（Mock）
   - `这个羽绒服有黑色吗` → RAG 检索命中商品 FAQ
   - `支持七天无理由吗` → RAG 检索命中政策 FAQ
3. 输入 **"人工"** 或 **"转人工"** → 用户端等待，客服后台左侧队列出现待接入
4. 客服后台点 **"▶ 立即接入"** → 接管对话，实时 WebSocket 双向通讯
5. 数据库验证：
   - MySQL：`SELECT * FROM session;` 和 `SELECT * FROM message ORDER BY id DESC;`
   - Redis：`keys cs:*:history`
   - Qdrant：通过 Dashboard（默认 http://localhost:6333/dashboard ）查看 `product_faqs` collection

---

## 降级 & 容错

| 组件 | 不可用时表现 | 恢复方式 |
|------|-------------|---------|
| DeepSeek API Key 未配置 | 智能体退回 **Mock 关键词匹配模式**，但对话链路保持可用 | 在 `.env` 填写 `DEEPSEEK_API_KEY` 后重启 |
| MySQL 连不上 | 启动时跳过建表，运行时降级为内存会话 | 检查 MySQL 服务 / `.env` 配置 |
| Redis 连不上 | 运行时异步超时降级，会话历史不写入 Redis | 检查 Redis 服务 / ACL 用户 / SSH 隧道 |
| Qdrant 连不上 | RAG 检索返回空列表，智能体失去知识库能力但对话仍可继续 | 检查 Qdrant 服务 |
| `sentence-transformers` 未装 | Embedding 降级为**全零 Mock 向量**，RAG 检索完全失效 | 按"快速开始 → 安装 Python 依赖"步骤安装 |

---

## 后续扩展方向

- [ ] 真实平台对接（淘宝千牛开放平台 / 京东开放平台 / 拼多多开放平台 / 抖店开放平台）
- [ ] 意图识别与转人工规则细化（置信度阈值 / 连续 Tool Calling 失败 / 关键词 + 语义双匹配）
- [ ] RAG 混合检索（向量 + BM25 关键词）提升准确率
- [ ] 多模型路由（简单问题走便宜模型，复杂问题走强模型）
- [ ] 会话质量分析 & 智能体自我评估
- [ ] Docker Compose 一键部署

---

## 许可证

MIT
