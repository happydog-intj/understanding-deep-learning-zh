#!/usr/bin/env python3
"""
Markdown翻译工具 - 英文翻译为中文（自动运行版本）
翻译前3个文件作为测试
"""

import os
import time
from pathlib import Path
from openai import OpenAI

# 配置
API_KEY = "sk-dc023de5a33f40aa8932b48e7a1f1d86"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL = "qwen-plus"

INPUT_DIR = "chapters_markdown"
OUTPUT_DIR = "chapters_chinese"
MAX_FILES = 3  # 先翻译3个文件测试


def init_client():
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)


def translate_text(client, text):
    """翻译为专业中文"""
    if not text.strip():
        return text

    prompt = f"""请将以下英文技术文档翻译为中文。

翻译要求：
1. 采用专业的计算机技术书籍翻译风格
2. 技术术语使用业界通用译法：
   - Deep Learning → 深度学习
   - Neural Network → 神经网络
   - Machine Learning → 机器学习
   - Gradient Descent → 梯度下降
   - Activation Function → 激活函数
3. 保持Markdown格式完整
4. 数学公式保持原样
5. 语言流畅自然

原文：
{text}

只输出翻译结果："""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "你是专业的计算机技术书籍译者。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=4000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"    ❌ 翻译失败: {str(e)}")
        return text


def split_content(content, max_length=2000):
    """分割长文本"""
    if len(content) <= max_length:
        return [content]

    paragraphs = content.split('\n\n')
    chunks = []
    current = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para)
        if current_len + para_len > max_length and current:
            chunks.append('\n\n'.join(current))
            current = [para]
            current_len = para_len
        else:
            current.append(para)
            current_len += para_len

    if current:
        chunks.append('\n\n'.join(current))

    return chunks


def translate_file(client, input_path, output_path):
    """翻译单个文件"""
    print(f"\n处理: {input_path.name}")

    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()

        print(f"  大小: {len(content)} 字符")

        if len(content) > 2000:
            chunks = split_content(content)
            print(f"  分为 {len(chunks)} 块")

            translated = []
            for i, chunk in enumerate(chunks, 1):
                print(f"  [{i}/{len(chunks)}] 翻译中...", end=' ')
                result = translate_text(client, chunk)
                translated.append(result)
                print("✓")
                time.sleep(1.5)

            final = '\n\n'.join(translated)
        else:
            print(f"  翻译中...", end=' ')
            final = translate_text(client, content)
            print("✓")

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(final)

        print(f"  ✅ 已保存")
        return True

    except Exception as e:
        print(f"  ❌ 失败: {str(e)}")
        return False


def main():
    print("=" * 60)
    print("深度学习书籍翻译 - 英文→中文（测试版）")
    print("=" * 60)

    input_path = Path(INPUT_DIR)
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(exist_ok=True)

    print(f"\n输入: {INPUT_DIR}")
    print(f"输出: {OUTPUT_DIR}")
    print(f"模型: {MODEL}")

    print("\n初始化...", end=' ')
    client = init_client()
    print("✓")

    md_files = sorted(list(input_path.glob("*.md")))[:MAX_FILES]
    print(f"\n将翻译 {len(md_files)} 个文件")

    print("\n" + "=" * 60)
    start = time.time()
    success = 0

    for i, md_file in enumerate(md_files, 1):
        print(f"\n[{i}/{len(md_files)}]")
        output_file = output_path / md_file.name

        if translate_file(client, md_file, output_file):
            success += 1

        if i < len(md_files):
            time.sleep(2)

    elapsed = time.time() - start

    print("\n" + "=" * 60)
    print(f"完成: {success}/{len(md_files)} 个文件")
    print(f"耗时: {elapsed:.1f} 秒")
    print(f"输出: {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
