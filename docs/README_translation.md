# 深度学习书籍翻译工具

## 📚 项目概述

这是一套完整的PDF技术书籍处理和翻译工具，可以将PDF按章节拆分并翻译为中文。

## 🎯 已完成的工作

### 1. PDF拆分 ✅
- **程序**: `pdf_to_markdown_by_chapter.py`
- **输入**: `UnderstandingDeepLearning_05_29_25_C.pdf` (537页)
- **输出**: `chapters_markdown/` (130个英文Markdown文件)

### 2. 翻译测试 ✅
- **程序**: `translate_auto.py`
- **已翻译**: 前3个章节
- **输出**: `chapters_chinese/`
- **翻译质量**: 专业计算机书籍翻译风格 ✓

## 📁 文件说明

### 核心程序

1. **pdf_to_markdown_by_chapter.py** - PDF章节拆分工具
   - 自动检测章节标题
   - 按章节拆分为Markdown文件
   - 保持原有格式

2. **translate_auto.py** - 快速翻译测试（3个文件）
   - 适合快速测试翻译质量
   - 已验证可用 ✓

3. **translate_all.py** - 完整翻译工具（所有130个文件）
   - 支持断点续传（跳过已翻译文件）
   - 自动分块处理大文件
   - API限流保护

4. **translate_to_chinese.py** - 交互式翻译工具
   - 可选择翻译范围
   - 详细进度显示

### 辅助工具

- **test_api.py** - API连接测试
- **requirements.txt** / **requirements_translate.txt** - 依赖包

## 🚀 使用方法

### 快速开始

```bash
# 1. 安装依赖
pip install pdfplumber openai

# 2. 翻译所有章节（推荐）
python translate_all.py
```

### 翻译所有130个章节

```bash
python translate_all.py
```

特点：
- 自动跳过已翻译文件（支持断点续传）
- 预计耗时：约260分钟（~4小时）
- 进度实时显示
- 失败自动记录

### 测试翻译（3个文件）

```bash
python translate_auto.py
```

## 📊 翻译质量示例

### 原文 (English)
```
Artificial intelligence, or AI, is concerned with building systems that
simulate intelligent behavior. Machine learning is a subset of AI that
learns to make decisions by fitting mathematical models to observed data.
```

### 译文 (Chinese)
```
人工智能（Artificial Intelligence，简称 AI）致力于构建能够模拟智能
行为的系统。机器学习（Machine Learning）是人工智能的一个子领域，
它通过将数学模型拟合到观测数据上来学习如何做出决策。
```

## ✨ 翻译特点

1. **专业术语准确**
   - Deep Learning → 深度学习
   - Neural Network → 神经网络
   - Gradient Descent → 梯度下降

2. **保持格式完整**
   - Markdown标题、列表、表格
   - 数学公式保持不变
   - 代码块保持不变

3. **译文流畅自然**
   - 符合中文阅读习惯
   - 专业计算机书籍风格
   - 首次出现术语附带英文

## 📈 当前状态

```
PDF拆分:      ✅ 完成 (130个文件)
翻译测试:     ✅ 完成 (3个文件)
完整翻译:     ⏳ 待运行 (130个文件)
```

## ⚙️ 配置说明

在程序中可以调整以下参数：

```python
MODEL = "qwen-plus"        # 模型选择: qwen-turbo/qwen-plus/qwen-max
MAX_LENGTH = 2000          # 分块大小
DELAY = 1.5                # API调用延迟（秒）
```

## 💡 使用建议

1. **首次使用**: 先运行 `translate_auto.py` 测试翻译质量
2. **完整翻译**: 使用 `translate_all.py` 翻译所有文件
3. **中断恢复**: 直接重新运行，程序会自动跳过已翻译文件
4. **质量检查**: 翻译完成后，随机抽查几个文件

## 📞 API配置

- **API Key**: sk-dc023de5a33f40aa8932b48e7a1f1d86
- **服务商**: 阿里云 DashScope
- **模型**: Qwen-Plus

## 📝 注意事项

⚠️ **API费用**: 翻译130个文件会产生API调用费用，建议先用测试版本验证

⚠️ **耗时**: 完整翻译预计需要4小时左右

✅ **优点**: 支持断点续传，可以随时中断和继续

## 🎓 目录结构

```
.
├── UnderstandingDeepLearning_05_29_25_C.pdf    # 原始PDF
├── chapters_markdown/                           # 英文章节(130个)
├── chapters_chinese/                            # 中文翻译(待完成)
├── pdf_to_markdown_by_chapter.py               # PDF拆分工具
├── translate_all.py                             # 完整翻译工具
├── translate_auto.py                            # 快速测试工具
└── README_translation.md                        # 本文档
```

## 🏆 成果

- ✅ 成功拆分537页PDF为130个章节
- ✅ 开发专业翻译工具
- ✅ 验证翻译质量优秀
- ⏳ 可随时完成全部翻译
