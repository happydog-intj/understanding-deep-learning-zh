#!/usr/bin/env python3
"""
Markdown翻译工具 - 英文翻译为中文
参考专业计算机书籍翻译风格
"""

import os
import time
from pathlib import Path
from openai import OpenAI

# 配置
API_KEY = "sk-dc023de5a33f40aa8932b48e7a1f1d86"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL = "qwen-plus"  # qwen-plus 或 qwen-max 更适合专业翻译

INPUT_DIR = "chapters_markdown"
OUTPUT_DIR = "chapters_chinese"


def init_client():
    """初始化OpenAI客户端"""
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)


def translate_text(client, text, retry_count=3):
    """
    将英文翻译为中文
    采用专业计算机书籍翻译风格
    """
    if not text.strip():
        return text

    # 翻译提示词 - 强调专业性和术语准确性
    prompt = f"""请将以下英文技术文档翻译为中文。

翻译要求：
1. 采用专业的计算机技术书籍翻译风格
2. 保持技术术语的准确性，常见术语保留英文或使用业界通用译法：
   - Deep Learning → 深度学习
   - Neural Network → 神经网络
   - Machine Learning → 机器学习
   - Gradient Descent → 梯度下降
   - Backpropagation → 反向传播
   - Overfitting → 过拟合
   - Loss Function → 损失函数
   - Activation Function → 激活函数
   - Hyperparameter → 超参数
3. 保持Markdown格式完整（标题、列表、代码块、链接等）
4. 数学公式和代码保持原样
5. 语言流畅自然，符合中文阅读习惯
6. 专有名词首次出现时可加注英文，如：卷积神经网络（Convolutional Neural Network, CNN）

原文：
{text}

请直接输出翻译后的中文内容，不要添加任何说明或解释。"""

    for attempt in range(retry_count):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "你是一位经验丰富的计算机技术书籍译者，精通中英文，擅长将英文技术文档翻译成专业、准确、流畅的中文。"
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=4000
            )

            translated = response.choices[0].message.content.strip()
            return translated

        except Exception as e:
            print(f"    ⚠️ 翻译失败 (尝试 {attempt + 1}/{retry_count}): {str(e)}")
            if attempt < retry_count - 1:
                time.sleep(2)
            else:
                print(f"    ❌ 多次失败，返回原文")
                return text

    return text


def split_content(content, max_length=2500):
    """
    智能分割内容
    按段落分割，避免破坏结构
    """
    if len(content) <= max_length:
        return [content]

    # 按双换行分割段落
    paragraphs = content.split('\n\n')

    chunks = []
    current_chunk = []
    current_length = 0

    for para in paragraphs:
        para_length = len(para)

        if current_length + para_length > max_length and current_chunk:
            chunks.append('\n\n'.join(current_chunk))
            current_chunk = [para]
            current_length = para_length
        else:
            current_chunk.append(para)
            current_length += para_length

    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))

    return chunks


def translate_file(client, input_path, output_path):
    """翻译单个Markdown文件"""
    print(f"\n处理: {input_path.name}")

    try:
        # 读取文件
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()

        print(f"  原文大小: {len(content)} 字符")

        # 大文件分块处理
        if len(content) > 2500:
            chunks = split_content(content, max_length=2500)
            print(f"  分为 {len(chunks)} 个块进行翻译")

            translated_chunks = []
            for i, chunk in enumerate(chunks, 1):
                print(f"  [{i}/{len(chunks)}] 翻译中...", end=' ')
                translated = translate_text(client, chunk)
                translated_chunks.append(translated)
                print("✓")

                # 避免API限流
                if i < len(chunks):
                    time.sleep(1.5)

            translated_content = '\n\n'.join(translated_chunks)
        else:
            print(f"  翻译中...", end=' ')
            translated_content = translate_text(client, content)
            print("✓")

        # 保存翻译结果
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(translated_content)

        print(f"  ✅ 已保存 ({len(translated_content)} 字符)")
        return True

    except Exception as e:
        print(f"  ❌ 处理失败: {str(e)}")
        return False


def main():
    print("=" * 70)
    print("深度学习书籍翻译工具 - 英文 → 中文")
    print("=" * 70)

    # 检查输入目录
    input_path = Path(INPUT_DIR)
    if not input_path.exists():
        print(f"❌ 错误: 输入目录不存在 {INPUT_DIR}")
        return

    # 创建输出目录
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(exist_ok=True)

    print(f"\n📂 输入目录: {INPUT_DIR}")
    print(f"📁 输出目录: {OUTPUT_DIR}")
    print(f"🤖 翻译模型: {MODEL}")

    # 初始化客户端
    print("\n初始化API客户端...", end=' ')
    try:
        client = init_client()
        print("✓")
    except Exception as e:
        print(f"❌ 失败: {str(e)}")
        return

    # 获取所有markdown文件
    md_files = sorted(list(input_path.glob("*.md")))
    print(f"\n找到 {len(md_files)} 个Markdown文件")

    if not md_files:
        print("没有找到任何文件")
        return

    # 询问翻译范围
    print("\n翻译选项:")
    print("1. 翻译前5个文件（测试）")
    print("2. 翻译所有130个文件（完整翻译）")
    choice = input("\n请选择 (1/2): ").strip()

    if choice == '1':
        md_files = md_files[:5]
        print(f"\n将翻译前 5 个文件")
    elif choice == '2':
        print(f"\n将翻译全部 {len(md_files)} 个文件")
        print("⚠️  注意: 这将需要较长时间，并可能产生API费用")
        confirm = input("确认继续? (y/n): ").strip().lower()
        if confirm != 'y':
            print("已取消")
            return
    else:
        print("无效选择，已取消")
        return

    # 开始翻译
    print("\n" + "=" * 70)
    print("开始翻译")
    print("=" * 70)

    success_count = 0
    fail_count = 0
    start_time = time.time()

    for i, md_file in enumerate(md_files, 1):
        print(f"\n{'='*70}")
        print(f"进度: [{i}/{len(md_files)}]")

        output_file = output_path / md_file.name

        if translate_file(client, md_file, output_file):
            success_count += 1
        else:
            fail_count += 1

        # 文件间延迟，避免API限流
        if i < len(md_files):
            time.sleep(2)

    # 统计结果
    elapsed_time = time.time() - start_time

    print("\n" + "=" * 70)
    print("翻译完成")
    print("=" * 70)
    print(f"✅ 成功: {success_count} 个文件")
    print(f"❌ 失败: {fail_count} 个文件")
    print(f"⏱️  耗时: {elapsed_time:.1f} 秒")
    print(f"📁 输出目录: {OUTPUT_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
