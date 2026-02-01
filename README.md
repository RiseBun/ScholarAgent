# ScholarAgent - 科研猎手     

## 作者在宁波出差，（核心文件夹backend暂未上传）,将上架huggingface

ScholarAgent 是一个基于LLM的智能科研助手系统，采用混合检索架构，将大语言模型的语义理解能力与传统学术数据库检索相结合，显著提升论文搜索的精准度和相关性。

## 🔬 技术点

### 1. LLM驱动的混合检索架构
- **创新**：将LLM语义理解与传统数据库检索深度融合，而非简单叠加
- **方法**：LLM先召回领域经典论文标题 → 数据库验证存在性 → LLM语义重排序
- **优势**：解决传统关键词匹配无法理解缩写（如"VLA"="Vision-Language-Action"）的问题

### 2. 多源异构数据融合
- **arXiv**：预印本论文，时效性强
- **DBLP**：会议论文，venue信息准确
- **OpenAlex**：引用数据丰富，覆盖全面
- **策略**：三源互补，解决单一数据源的覆盖盲区

### 3. 两阶段语义重排序
| 阶段 | 方法 | 作用 |
|------|------|------|
| 粗排 | HybridScorer多维评分 | 快速筛选Top-N候选 |
| 精排 | LLM Semantic Rerank | 深度理解用户意图 |

### 4. 用户意图闭环优化
```
Query → AI解释 → 用户反馈 → 重新理解 → 搜索
```
解决科研查询的模糊性问题，用户可实时修正AI的理解偏差。

## 📊 核心算法

### 多路召回策略
```
用户查询 "VLA"
    │
    ├─→ [LLM召回] 基于领域知识生成10篇经典论文标题
    │
    ├─→ [Query扩展] "VLA" → "Vision-Language-Action OR Multimodal LLM for Robotics"
    │
    └─→ [多源并发检索]
         ├─ arXiv API (最新预印本)
         ├─ DBLP API (会议论文，venue准确)
         └─ OpenAlex API (引用数据)
```

### 多维质量评分公式
```python
Score = Relevance × 0.30 + Citation × 0.25 + Venue × 0.30 + Recency × 0.15
```

| 维度 | 计算方法 |
|------|----------|
| **Relevance** | 查询词与标题/摘要的TF-IDF相似度 |
| **Citation** | log(引用数+1) 归一化到[0,1] |
| **Venue** | 顶会=1.0, 次顶会=0.9, 其他=0 |
| **Recency** | 1 - 0.1 × (当前年 - 发表年) |

### 结果多样化：MMR算法
```python
# Maximal Marginal Relevance - 避免结果同质化
while len(selected) < top_n:
    similarity = cosine_similarity(tfidf[candidates], tfidf[selected])
    next_paper = argmin(max(similarity, axis=1))  # 选最不相似的
    selected.append(next_paper)
```

## 🎯 核心功能

### 1. 快速搜索 (Quick Search)
- 自然语言查询，支持多LLM提供商（OpenAI、Qianwen、DeepSeek、Gemini）
- 智能AI查询解释：自动理解用户意图并生成标准化查询
- 用户可控的解释反馈：可对AI解释进行修改和优化
- 多维度论文标签：顶级会议、引用量、代码可用性、开放获取
- LLM驱动的语义重排序，识别"必读baseline"论文

### 2. 知识景观生成 (Landscape Mapping)
- 生成研究领域全景图，识别热点、蓝海和红海
- 论文聚类分析，展示不同研究方向

### 3. 极速快读 (Quick Scan)
- 批量分析论文核心贡献、弱点和未来工作
- 生成论文比较表格，快速识别研究漏洞

### 4. 灵感缝合 (Idea Breeder)
- 多Agent辩论：乐观视角 vs 批判视角
- 跨学科类比搜索，可行性分析

### 5. 撞车预警 (Scoop Checker)
- 检测研究想法是否被抢占
- 提供差异化策略和方向建议

## 🔧 技术架构

```
┌─────────────────────────────────────────────────────────────┐
│                      Frontend (Streamlit)                    │
├─────────────────────────────────────────────────────────────┤
│  QueryExpander │ HybridSearch │ LLMReranker │ HybridScorer  │
├─────────────────────────────────────────────────────────────┤
│     LLMPaperRecaller    │       PaperFetcher                │
├─────────────────────────────────────────────────────────────┤
│   arXiv API   │   DBLP API   │   OpenAlex API   │   LLM API │
└─────────────────────────────────────────────────────────────┘
```

### 后端核心组件
| 组件 | 文件 | 功能 |
|------|------|------|
| **HybridSearchEngine** | `hybrid_search.py` | 混合检索主引擎，协调各组件 |
| **LLMPaperRecaller** | `llm_paper_recaller.py` | LLM直接召回经典论文标题 |
| **QueryExpander** | `query_expander.py` | AI驱动的查询语义扩展 |
| **PaperFetcher** | `search_engine.py` | 多源并发论文检索 |
| **HybridScorer** | `hybrid_scorer.py` | 多维质量评分 |
| **LLMReranker** | `llm_reranker.py` | LLM语义精排序 |
| **Tagger** | `tagger.py` | 论文多维标签生成 |

## 🚀 快速开始

### 环境要求
- Python 3.10+
- 网络连接（API调用）
- LLM API密钥（至少一个）

### 安装步骤

```bash
# 1. 克隆项目
git clone <repository-url>
cd ScholarAgent

# 2. 创建conda环境
conda create -n scholaragent python=3.10
conda activate scholaragent

# 3. 安装依赖
pip install -r requirements.txt

# 4. 启动应用
streamlit run app.py
```

### 配置API密钥
在Streamlit侧边栏输入LLM API密钥：
- [OpenAI](https://platform.openai.com/api-keys)
- [Qianwen/通义千问](https://dashscope.aliyun.com/)
- [DeepSeek](https://platform.deepseek.com/)
- [Gemini](https://makersuite.google.com/)
- [OpenRouter](https://openrouter.ai/)

## 📁 项目结构

```
ScholarAgent/
├── app.py                      # 主应用入口
├── requirements.txt            # 依赖配置
├── pages/
│   ├── 1_🔍_Search.py          # 快速搜索页面
│   └── 2_📚_Library.py         # 论文库页面
└── backend/
    ├── agent.py                # LLM客户端封装
    ├── hybrid_search.py        # 混合检索引擎
    ├── search_engine.py        # 多源论文检索
    ├── query_expander.py       # 查询语义扩展
    ├── llm_paper_recaller.py   # LLM论文召回
    ├── llm_reranker.py         # LLM语义重排序
    ├── hybrid_scorer.py        # 多维质量评分
    ├── tagger.py               # 论文标签生成
    ├── data_models.py          # 数据模型定义
    ├── content_curator.py      # 内容策展
    ├── landscape.py            # 知识景观生成
    ├── quickscan.py            # 极速快读
    ├── ideabreeder.py          # 灵感缝合
    └── scoop_checker.py        # 撞车预警
```

## 🔍 搜索流程

```
1. 用户输入查询
       ↓
2. [并行] LLM召回 + Query扩展
       ↓
3. [并行] arXiv/DBLP/OpenAlex 多源检索
       ↓
4. 智能去重与数据融合
       ↓
5. HybridScorer 多维评分（粗排）
       ↓
6. LLMReranker 语义重排序（精排）
       ↓
7. MMR 结果多样化
       ↓
8. 返回带标签的论文列表
```

## 🛠️ 技术栈

- **前端**：Streamlit
- **后端**：Python 3.10+, asyncio
- **LLM**：OpenAI API / 通义千问 / DeepSeek / Gemini
- **数据源**：arXiv API, DBLP API, OpenAlex API
- **ML**：scikit-learn (TF-IDF, 余弦相似度, 聚类)
- **数据验证**：Pydantic

## 📝 注意事项

1. **API密钥**：需要有效的LLM API密钥
2. **网络环境**：部分API可能需要代理访问
3. **速率限制**：arXiv有请求频率限制，系统已内置自动降级机制
4. **结果质量**：依赖LLM性能，推荐使用GPT-4或同等模型

## 📄 许可证

MIT License

---

**ScholarAgent - 让科研更智能，让灵感更闪耀！**
