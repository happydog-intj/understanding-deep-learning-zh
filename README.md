# Understanding Deep Learning 中文翻译

> 《理解深度学习》中文翻译 —— 基于 VitePress 的在线电子书

[![License](https://img.shields.io/badge/License-CC--BY--NC--ND-orange.svg)](LICENSE)

📖 **在线阅读**：[Vercel](https://understanding-deep-learning-zh.vercel.app) | [GitBook](https://bruceinpeking.gitbook.io/understanding-deep-learning-zh)

---

## 📚 项目简介

本项目是 Simon J.D. Prince 所著 [*Understanding Deep Learning*](https://udlbook.github.io/udlbook/)（2026年2月版，MIT Press）的中文翻译。原书以 CC-BY-NC-ND 许可协议发布，本翻译仅供学习交流使用。

- **原书**：Understanding Deep Learning, Simon J.D. Prince
- **出版社**：MIT Press, 2026
- **页数**：537页（21章 + 3附录）
- **在线阅读**：使用 VitePress 构建，支持数学公式渲染、暗色模式和全文搜索

## 📖 目录

### Part I 基础知识
- 第1章 引言
- 第2章 监督学习
- 第3章 浅层神经网络
- 第4章 深度神经网络

### Part II 模型训练
- 第5章 损失函数
- 第6章 模型拟合
- 第7章 梯度与初始化
- 第8章 性能度量
- 第9章 正则化

### Part III 特殊架构
- 第10章 卷积网络
- 第11章 残差网络
- 第12章 Transformer
- 第13章 图神经网络

### Part IV 生成模型
- 第14章 无监督学习
- 第15章 生成对抗网络
- 第16章 归一化流
- 第17章 变分自编码器
- 第18章 扩散模型

### Part V 强化学习与伦理
- 第19章 强化学习
- 第20章 深度学习为何有效？
- 第21章 深度学习与伦理

### 附录
- 附录A 符号表
- 附录B 数学基础
- 附录C 概率论基础

## 🚀 本地运行

```bash
# 克隆仓库
git clone https://github.com/happydog/understanding-deep-learning-zh.git
cd understanding-deep-learning-zh

# 安装依赖
npm install

# 启动开发服务器
npm run docs:dev

# 构建静态站点
npm run docs:build
```

## ✨ 翻译规范

- 专业术语首次出现附英文原文，如：反向传播（Backpropagation）
- 保留原书 LaTeX 数学公式
- 图表引用保留原编号（如"图 1.2"）
- 不翻译专有名词（如 AlexNet、Transformer）

## 📄 许可证

原书以 [CC-BY-NC-ND](https://creativecommons.org/licenses/by-nc-nd/4.0/) 许可协议发布。本翻译遵循相同协议，仅供学习交流，禁止商业用途。

## 🙏 致谢

- 原书作者 Simon J.D. Prince 以及所有章节共同作者
- MIT Press 出版社
