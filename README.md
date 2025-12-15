# 基于强化学习的 Python 豆瓣电影推荐系统

这是一个“无服务器部署、本地单机运行”的电影推荐项目：前台继续使用 `Flask + SQLite`，后台补齐本地强化学习重排、语义增强推荐、实验可视化和更规范的 `repository / service / schema` 分层。

## 项目定位

- 数据链路：“可扩容、可断点续爬、可生成行为样本”的本地数据链路。
- 算法链路：“传统推荐展示”到“baseline 对比 + 轻量语义召回 + 本地 PPO 重排 + 离线评估 + 在线经验池”的完整闭环。
- 工程链路：“配置、日志、版本化 API、后台管理台、任务状态追踪、repository/service/schema 分层”的可维护结构。

## 核心能力

- 混合推荐召回：内容推荐 + 协同过滤，与语义增强链路隔离对比。
- 语义增强推荐：补充本地轻量 `Word2Vec-style / GloVe-style` 实现，分别支持内容召回、协同过滤语义化和 `semantic_hybrid`。
- 本地 RL 重排：PPO 重排层，加入 `ε-greedy` 探索、覆盖率奖励、多样性加权。
- 轻量语义扩展：状态编码支持轻量语义标签；若配置 OpenAI-compatible 接口，可进一步生成更细粒度语义特征。
- 本地缓存：用户特征、电影特征、状态向量默认走内存缓存，也支持本机 Redis。
- 批量更新：在线反馈写入 `rl_experience`，达到阈值后再批量训练，避免逐条更新。
- 模型版本管理：模型权重本地落盘，SQLite 注册版本，支持回滚和自动清理旧版本。
- 离线评估：支持 `precision@k / recall@k / ndcg@k / coverage / diversity`。
- 后台可视化：新增管理台页面和 `api/v1/admin/*` 接口，实验页支持折线图/柱状图查看算法趋势。
- 数据采集增强：多页爬取、随机延迟、UA 池、代理文件、断点续爬、去重清洗、状态文件落盘。

## 目录结构

```text
app.py
config.py
myutils/
  admin_api.py               # 后台页面和版本化管理 API
  app_logging.py             # 结构化日志初始化
  evaluation.py              # 离线评估指标
  behaviorData.py            # 行为写入与在线经验入池
  recommend.py               # 混合召回 + PPO 重排入口
  crawler/
    core.py                  # 爬虫公共能力：UA/延迟/代理/断点/清洗
    jobs.py                  # 电影、评论、行为样本任务
  recommender/
    semantic_embeddings.py   # 本地 Word2Vec-style / GloVe-style 语义召回
  rl/
    cache.py                 # 本地内存/Redis 缓存
    features.py              # 用户/电影/状态向量编码
    semantic.py              # 轻量语义标签与可选远程语义增强
    local_ppo.py             # 本地 PPO 重排、训练、版本管理
repositories/
  behavior_repository.py
  experiment_repository.py
  model_repository.py
  movie_repository.py
  system_repository.py
services/
  behavior_service.py
  catalog_service.py
  experiment_service.py
  model_service.py
  recommendation_service.py
  system_service.py
  ui_audit_service.py
schemas/
  common.py
templates/
  admin_dashboard.html
  admin_models.html
  admin_crawler.html
  admin_experiments.html
runtime/
  crawler/
  logs/
artifacts/
  rl_models/ppo/
```

## 快速开始

1. 安装依赖

```bash
pip install -r requirements.txt
```

2. 启动项目

```bash
python app.py
```

3. 浏览器访问

```text
http://127.0.0.1:5001
```

默认管理员账号：

```text
alice@example.com / 123456
```

## 语义推荐取舍

这次落地的是“本地轻量 `Word2Vec-style / GloVe-style` 实现”，目标是兼顾算法深度和单机可运行性，不是直接引入重型外部 embedding 训练框架。

这样做的价值：

- 面试里能把 trade-off 讲清楚：为什么不是一上来就上重型分布式训练，而是在当前数据规模、单机约束和项目目标下，先落地可解释、可跑通、可对比的语义 embedding。
- 算法深度比纯 `TF-IDF` 更强：可以讲共现建模、局部上下文、词向量空间、语义近邻这些点。
- 工程落地成本可控：不依赖额外服务，不破坏当前 `Flask + SQLite` 结构，方便直接接入离线评估和后台实验页。

当前语义链路和 baseline 链路是隔离的：

- baseline：`baseline_content / baseline_cf / baseline_hybrid`
- semantic：`word2vec_content / glove_content / word2vec_cf / glove_cf / semantic_hybrid`
- rerank：`ppo_rerank`

这能保证面试时清楚展示“基础方案 -> 语义增强 -> RL 重排”的渐进优化路径。

## 本地 RL 方案

当前项目不走“全量 PPO 主推荐器”，而是走更适合单机小数据场景的路线：

1. 混合推荐负责召回，保证推荐可用。
2. PPO 负责候选重排，重点体现强化学习、探索与利用、在线更新和版本回滚。
3. 多样性和覆盖率作为辅助目标融入打分。

### 已实现的 RL 优化点

- 缓存：用户特征、电影特征、状态向量默认缓存到内存，配置 `MOVIE_REDIS_URL` 后可自动切换到 Redis。
- 批量更新：经验池达到 `MOVIE_RL_BATCH_SIZE` 后训练，默认 100。
- 模型治理：每次训练都会生成本地版本文件，并在 `model_registry` 里登记。
- 自动清理：超过 `MOVIE_KEEP_MODEL_VERSIONS` 的旧模型会自动清理。
- 探索策略：重排时加入 `ε-greedy`。
- 语义特征：简介文本会生成轻量语义标签；若配置兼容接口，可接入远程语义增强。

### 当前更适合讲的架构点

- PPO 不是主召回器，而是本地 rerank 层，专门解决小数据场景下“直接全量 PPO 收益不稳、复杂度过高”的问题。
- 在线反馈不是逐条训练，而是写入 `rl_experience` 经验池，达到阈值后再批量更新，避免本机资源抖动。
- 模型版本保存在本地文件系统，版本元数据登记到 `model_registry`，可以在后台页或 CLI 中快速回滚。
- 行为、模型、实验相关的数据库访问已经从业务模块中抽到 `repository/service`，降低 `myutils/query.py` 的职责耦合。

### 模型管理命令

```bash
python -m myutils.rl.local_ppo status
python -m myutils.rl.local_ppo bootstrap
python -m myutils.rl.local_ppo train --force
python -m myutils.rl.local_ppo list
python -m myutils.rl.local_ppo rollback <version_tag>
```

## 数据采集与样本扩容

### 电影数据

```bash
python spider.py
python crawl_movies_batch.py
python -m myutils.crawler.jobs movies --pages 8
```

### 评论数据

```bash
python spider_comments.py
python crawl_comments_batch.py
python -m myutils.crawler.jobs comments --pages-per-movie 3 --limit-movies 30
```

### 行为样本扩容

```bash
python build_behavior_dataset.py
python -m myutils.crawler.jobs behaviors --user-count 60 --min-behaviors 8 --max-behaviors 16
```

行为样本会同时输出到 `behavior_dataset.csv`，并默认写入 SQLite，方便直接驱动推荐、评估和 RL 自举。

### 爬虫增强点

- 多页爬取：支持 Top250 分页抓取。
- 反爬：UA 池、随机延迟、可选代理文件。
- 断点续爬：状态写入 `runtime/crawler/checkpoints/`。
- 状态可视化：任务状态写入 `runtime/crawler/crawler_status.json`。
- 去重清洗：标题、评论、行为样本均支持去重合并。

## 后台管理台

登录管理员后可访问：

- `/admin/dashboard`：系统总览、离线指标、RL 状态、数据采集状态。
- `/admin/models`：模型版本列表、bootstrap、训练、回滚。
- `/admin/crawler`：任务参数配置、行为扩容、采集状态查看。
- `/admin/experiments`：实验快照、折线图趋势、柱状图对比、历史快照追踪。

对应 API：

- `GET /api/v1/admin/overview`
- `GET /api/v1/admin/evaluation`
- `GET /api/v1/admin/models`
- `POST /api/v1/admin/models/bootstrap`
- `POST /api/v1/admin/models/train`
- `POST /api/v1/admin/models/<version_tag>/rollback`
- `GET /api/v1/admin/crawler/status`
- `POST /api/v1/admin/crawler/run`
- `GET /api/v1/admin/experiments`
- `POST /api/v1/admin/experiments/run`

## 推荐效果评估

离线评估使用留一验证思路，输出：

- `precision@k`
- `recall@k`
- `ndcg@k`
- `coverage`
- `diversity`

这部分能力位于 `myutils/evaluation.py`，管理台会直接展示结果，实验页会把多次快照沉淀为折线图/柱状图，适合在面试中讲“如何做数据驱动优化”和“如何保留实验迭代轨迹”。

## 配置项

```bash
export SECRET_KEY=dev-secret
export MOVIE_ADMIN_EMAILS=alice@example.com
export MOVIE_RL_ENABLED=1
export MOVIE_RL_BATCH_SIZE=100
export MOVIE_RL_MIN_FEEDBACK=5
export MOVIE_PPO_EPOCHS=6
export MOVIE_PPO_LR=0.03
export MOVIE_PPO_CLIP=0.2
export MOVIE_RL_EPSILON=0.1
export MOVIE_RL_DIVERSITY_WEIGHT=0.18
export MOVIE_RL_COVERAGE_WEIGHT=0.12
export MOVIE_KEEP_MODEL_VERSIONS=6
export MOVIE_REDIS_URL=redis://127.0.0.1:6379/0
export DOUBAN_MIN_DELAY=1.1
export DOUBAN_MAX_DELAY=2.6
export DOUBAN_PROXY_FILE=./runtime/crawler/proxies.txt
export MOVIE_LLM_BASE_URL=
export MOVIE_LLM_API_KEY=
export MOVIE_LLM_MODEL=gpt-4o-mini
```

## 当前已补齐的面试亮点

- 不是只讲算法，而是补了数据采集、缓存、训练、评估、版本回滚和后台管理。
- 不依赖独立服务器，单机即可完成采集、训练、推荐和可视化展示。
- 能明确讲清楚为什么在当前规模下选择“混合召回 + PPO 重排”而不是盲目全量 PPO。
- 能明确讲清楚为什么这里落地的是本地轻量 `Word2Vec-style / GloVe-style`，而不是直接接重型 embedding 训练框架，这是一个围绕数据规模、算力约束和可演示性的 trade-off。
- 能展示“线上行为 -> 经验池 -> 批量训练 -> 新版本上线 -> 指标观察 -> 回滚”的闭环。
- 能展示“baseline -> semantic embedding -> PPO rerank -> 实验页趋势验证”的渐进式优化路径。

## 仍然可以继续深挖的方向

- 将传统推荐进一步升级为 `ALS / Wide&Deep / DIN`。
- 增加更真实的 A/B 实验和时间切分评估。
- 用 `SQLAlchemy`、更强的 DTO 和任务调度做进一步工程化。
- 增加更强的监控和任务追踪体系。

如果继续往真正生产级靠近，下一步最值得补的是更强的特征工程和更系统的监控链路。
