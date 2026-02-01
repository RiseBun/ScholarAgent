# 集成 Google Scholar 到 ScholarAgent 搜索引擎

## 1. 架构设计

### 1.1 扩展 PaperFetcher 类
- 添加 `_search_google_scholar` 方法，支持通过 SerpAPI 搜索 Google Scholar
- 添加 `_parse_google_scholar_entry` 方法，解析 Google Scholar 返回的论文信息
- 添加 `_fetch_google_scholar_async` 异步包装方法
- 修改 `search_papers_async` 方法，将 Google Scholar 集成到混合搜索策略中

### 1.2 数据源优先级
- **主要数据源**：Google Scholar（通过 SerpAPI）
- **补充数据源**：arXiv（提供预印本）
- **辅助数据源**：OpenAlex（提供引用信息）

## 2. 功能实现

### 2.1 Google Scholar 搜索功能
- 支持通过 SerpAPI 搜索 Google Scholar
- 实现关键词、年份、顶会过滤等功能
- 支持 site:openreview.net 等高级搜索语法
- 提供错误处理和降级机制

### 2.2 论文解析与标准化
- 解析 Google Scholar 返回的论文信息
- 提取标题、作者、摘要、年份、引用数、venue 等字段
- 标准化不同数据源的论文格式，确保正确合并
- 增强顶会检测，特别是 ICLR、CVPR 等会议

### 2.3 智能合并与去重
- 扩展 `_smart_merge` 方法，支持 Google Scholar 论文
- 优先使用 Google Scholar 的 venue 信息
- 结合 OpenAlex 的引用信息
- 确保不同数据源论文的正确去重

### 2.4 UI 集成
- 在 Streamlit 应用中添加 Google Scholar 配置选项
- 添加 SerpAPI API Key 输入框
- 在搜索结果中显示 Google Scholar 特有的信息（如引用数、相关文章）
- 提供 Google Scholar 搜索链接，方便用户手动验证

## 3. 技术细节

### 3.1 SerpAPI 集成
```python
def _search_google_scholar(self, query, limit=10, start_year=2024, end_year=2026, venue_filter="", offset=0):
    """Search papers using Google Scholar via SerpAPI"""
    # 实现 SerpAPI 调用逻辑
    # 解析返回结果
    # 过滤和标准化论文信息
    # 返回标准化的论文列表
```

### 3.2 错误处理与降级
- 当 SerpAPI 不可用时，自动降级到 arXiv + OpenAlex
- 当 Google Scholar 返回结果较少时，补充 arXiv 和 OpenAlex 结果
- 实现请求超时和重试机制

### 3.3 性能优化
- 使用异步编程，与其他数据源并发搜索
- 合理设置搜索参数，减少 API 调用次数
- 缓存搜索结果，避免重复请求

## 4. 预期效果

### 4.1 搜索质量提升
- 更好地检索 ICLR、CVPR 等顶会论文
- 提高论文的相关性和准确性
- 增加被引数等重要信息

### 4.2 功能增强
- 支持更多高级搜索语法
- 提供更全面的论文信息
- 增强顶会检测和 venue 识别

### 4.3 用户体验改进
- 简化搜索流程，减少用户手动验证
- 提供更准确的搜索结果
- 增加 Google Scholar 特有功能的集成

## 5. 实施步骤

1. **添加 Google Scholar 搜索方法**：实现 `_search_google_scholar` 和相关方法
2. **扩展混合搜索策略**：修改 `search_papers_async` 方法，集成 Google Scholar
3. **实现 UI 配置**：在 Streamlit 应用中添加 Google Scholar 相关选项
4. **测试与优化**：测试搜索结果质量，优化 API 调用和错误处理
5. **文档更新**：更新代码注释和用户文档

通过集成 Google Scholar，ScholarAgent 将能够更全面、更准确地检索学术论文，特别是顶会论文，从而为用户提供更好的学术搜索体验。