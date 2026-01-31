# ScholarAgent - 科研猎手

ScholarAgent 是一个智能科研助手系统，专为科研人员设计，能够模拟真实科研工作者的找资料、看论文、找idea的完整旅程。

## 🎯 核心功能

### 1. 快速搜索 (Quick Search)
- 自然语言查询，支持多LLM提供商
- 智能AI查询解释：自动理解用户意图并生成标准化查询
- 用户可控的解释反馈：可对AI解释进行修改和优化
- 增强型数据源搜索：arXiv + OpenAlex（更全面的学术覆盖）
- 流式结果显示，实时展示搜索结果
- 多维度论文标签：代码可用性、顶级会议、引用量、开放获取
- 分页功能：支持多页浏览和历史记录跟踪
- 智能论文重排序：基于相关性和质量的LLM驱动排序

### 2. 知识景观生成 (Landscape Mapping)
- 生成研究领域全景图，识别热点、蓝海和红海
- 论文聚类分析，展示不同研究方向
- 可视化展示研究趋势和分布
- 帮助用户快速找到领域切入点

### 3. 极速快读 (Quick Scan)
- 批量分析论文核心贡献、弱点和未来工作
- 生成论文比较表格，快速识别研究漏洞
- 提取关键信息，节省阅读时间
- 帮助用户快速筛选有价值的论文

### 4. 灵感缝合 (Idea Breeder)
- 多Agent辩论：乐观视角 vs 批判视角
- 跨学科类比搜索，寻找灵感
- 可行性分析和资源评估
- 帮助用户完善和验证研究想法

### 5. 撞车预警 (Scoop Checker)
- 检测研究想法是否被抢占
- 提供差异化策略和方向建议
- 分析研究空白和机会
- 帮助用户避免重复工作，找到独特价值

## 🔧 技术架构

### 前端
- Streamlit：交互式Web界面
- 响应式设计，支持多模式切换
- 实时状态反馈和进度显示

### 后端
- Python 3.10+
- 多LLM提供商支持：OpenAI、Qianwen、DeepSeek、Gemini、OpenRouter
- 增强型数据源集成：arXiv API + OpenAlex API（替代Semantic Scholar，提供更全面的学术覆盖）
- 机器学习：scikit-learn（聚类和相似度计算）
- 数据可视化：matplotlib
- 智能组件：
  - QueryExpander：AI驱动的查询扩展和标准化
  - ContentCurator：LLM驱动的论文重排序和质量评估
  - 分页系统：支持多页浏览和历史记录管理

## 🚀 本地部署

### 1. 环境要求
- Python 3.10 或更高版本
- pip 包管理工具
- 网络连接（用于API调用）

### 2. 安装步骤

#### 步骤1：克隆项目
```bash
git clone <repository-url>
cd ScholarAgent
```

#### 步骤2：创建和激活conda环境
```bash
# 创建conda环境
conda create -n scholaragent python=3.10

# 激活环境
conda activate scholaragent
```

#### 步骤3：安装依赖
```bash
pip install -r requirements.txt
```

#### 步骤4：配置API密钥
在Streamlit应用中，通过侧边栏的"API Keys"部分输入您的LLM提供商API密钥：
- OpenAI: [获取API密钥](https://platform.openai.com/api-keys)
- Qianwen: [获取API密钥](https://dashscope.aliyun.com/)
- DeepSeek: [获取API密钥](https://platform.deepseek.com/)
- Gemini: [获取API密钥](https://makersuite.google.com/)
- OpenRouter: [获取API密钥](https://openrouter.ai/)

### 3. 启动应用
```bash
streamlit run app.py
```

应用将在浏览器中打开，默认地址为：`http://localhost:8501`

## 📖 使用指南

### 快速开始
1. **选择模式**：在侧边栏的"Navigation"中选择您需要的功能模式
2. **配置设置**：选择LLM提供商并输入API密钥
3. **输入查询**：根据不同模式的要求输入您的研究查询或想法
4. **AI查询解释**（快速搜索模式）：
   - 点击"🤖 Interpret Query"按钮分析查询
   - 查看AI生成的标准化查询和解释
   - 提供反馈并修改解释（如需）
   - 点击"✅ Approve Interpretation"确认满意
5. **开始搜索**：点击"🔍 Start Search"按钮开始搜索
6. **查看结果**：系统将处理您的请求并显示相应的结果
7. **分页浏览**：
   - 使用"🔄 Refresh"按钮加载下一页
   - 在"📋 View History Pages"中查看和导航到历史页面

### 功能模式说明

#### 1. 快速搜索
- **输入**：研究主题或关键词
- **AI处理**：自动分析意图，生成标准化查询
- **用户交互**：可审核和修改AI解释，提供反馈
- **输出**：相关论文列表，包含标题、摘要、链接和多维度标签
- **分页**：支持多页浏览和历史记录跟踪
- **用途**：快速获取相关论文，了解研究现状，发现研究机会

#### 2. 知识景观生成
- **输入**：研究领域（如"人工智能"、"计算机视觉"）
- **输出**：领域全景图、研究聚类和趋势分析
- **用途**：了解领域结构，找到研究方向

#### 3. 极速快读
- **输入**：研究主题
- **输出**：论文分析表格，包含核心贡献、弱点和未来工作
- **用途**：快速筛选论文，识别研究机会

#### 4. 灵感缝合
- **输入**：研究想法描述
- **输出**：多Agent辩论、跨学科灵感、可行性分析
- **用途**：完善研究想法，获得多维度反馈

#### 5. 撞车预警
- **输入**：详细的研究想法
- **输出**：撞车风险评估、相似论文、差异化建议
- **用途**：避免重复工作，找到独特价值

## 📁 项目结构

```
ScholarAgent/
├── app.py                 # 主Streamlit应用
├── requirements.txt       # 依赖项配置
├── README.md              # 项目说明
├── test_backend.py        # 后端测试
├── test_full_search.py    # 完整搜索流程测试
├── test_openalex.py       # OpenAlex API测试
├── test_openalex_detailed.py # OpenAlex详细测试
├── test_tagger.py         # 标签生成测试
└── backend/
    ├── __init__.py
    ├── agent.py           # LLM意图识别
    ├── data_models.py     # 论文数据模型
    ├── search_engine.py    # 论文搜索引擎
    ├── tagger.py          # 论文标签生成
    ├── landscape.py       # 知识景观生成
    ├── quickscan.py       # 极速快读功能
    ├── ideabreeder.py     # 灵感缝合功能
    ├── scoop_checker.py    # 撞车预警功能
    ├── query_expander.py   # AI查询扩展和解释
    └── content_curator.py  # 论文内容重排序和质量评估
```

## 🛠️ 技术栈

### 核心依赖
- `streamlit`：Web界面框架
- `openai`：OpenAI API客户端
- `pydantic`：数据模型验证
- `scikit-learn`：机器学习（聚类和相似度计算）
- `matplotlib`：数据可视化
- `numpy`：数值计算
- `joblib`：缓存和并行处理
- `httpx`：HTTP客户端（用于API调用）
- `arxiv`：arXiv API客户端
- `tenacity`：重试机制（用于API调用）

### 数据源
- **arXiv API**：快速获取预印本论文
- **OpenAlex API**：全面的学术论文数据库

### 智能组件
- **QueryExpander**：AI驱动的查询理解和扩展
- **ContentCurator**：LLM驱动的论文重排序和质量评估
- **PaperFetcher**：增强型论文搜索和获取
- **Tagger**：多维度论文标签生成

## 🔍 搜索流程

1. **用户输入**：自然语言查询
2. **AI查询解释**：
   - QueryExpander分析查询意图
   - 生成标准化搜索查询
   - 提取相关术语和venue信息
   - 显示详细解释供用户审核
3. **用户反馈**：
   - 用户可查看和修改AI解释
   - 提供反馈以优化查询理解
   - 确认满意后开始搜索
4. **增强型数据源搜索**：
   - 首先搜索arXiv（速度快）
   - 然后搜索OpenAlex（更全面的学术覆盖）
   - 智能分页：基于offset的结果获取
5. **结果处理**：
   - 去重和数据标准化
   - 多维度标签生成
   - 智能过滤和筛选
6. **LLM驱动的重排序**：
   - ContentCurator基于相关性和质量重排序
   - 考虑用户意图和研究价值
7. **流式显示**：实时展示搜索结果
8. **分页管理**：
   - 支持多页浏览
   - 历史记录跟踪
   - 页面导航功能

## 📊 数据分析流程

### 知识景观生成
1. 关键词扩展 → 多源搜索 → 论文聚类 → 趋势分析 → 可视化生成

### 极速快读
1. 论文搜索 → LLM摘要分析 → 核心信息提取 → 比较表格生成

### 灵感缝合
1. 想法输入 → 多Agent辩论 → 类比搜索 → 可行性分析 → 建议生成

### 撞车预警
1. 想法输入 → 关键词提取 → 相似论文搜索 → 相似度计算 → 风险评估 → 差异化建议

## 🎨 用户界面

- **侧边栏**：导航菜单、LLM设置、API密钥配置
- **主界面**：根据选择的模式显示相应的功能界面
- **响应式设计**：适配不同屏幕尺寸
- **实时反馈**：操作状态和进度显示

## 🌟 特色优势

1. **模拟真实科研旅程**：从找方向到验证想法的完整流程
2. **多LLM支持**：灵活选择不同的语言模型提供商
3. **增强型数据源**：arXiv + OpenAlex，兼顾速度和全面性
4. **智能AI查询理解**：自动分析意图，生成标准化查询
5. **用户可控的交互**：可修改和优化AI解释，确保查询准确性
6. **LLM驱动的智能分析**：利用语言模型提供深度论文分析
7. **智能分页系统**：支持多页浏览和历史记录管理
8. **自适应搜索策略**：无结果时自动调整搜索范围
9. **用户友好**：直观的Web界面，易于使用
10. **可扩展性**：模块化设计，易于添加新功能

## 📝 注意事项

1. **API密钥**：部分功能需要有效的LLM API密钥才能使用
2. **网络连接**：需要稳定的网络连接以调用外部API
3. **搜索速度**：取决于网络状况和API响应速度
4. **结果质量**：依赖于LLM的性能和API的返回结果
5. **数据限制**：受限于API的速率限制和使用配额

## 🤝 贡献

欢迎提交Issue和Pull Request，帮助改进ScholarAgent！

## 📄 许可证

MIT License

## 📞 联系方式

如有问题或建议，请通过Issue与我们联系。

---

**ScholarAgent - 让科研更智能，让灵感更闪耀！** ✨
#   S c h o l a r A g e n t  
 