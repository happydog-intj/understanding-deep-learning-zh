#!/usr/bin/env python3
"""
深度学习书籍完整翻译工具 - 英文→中文
翻译所有130个章节
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


def init_client():
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)


def translate_text(client, text):
    """专业计算机书籍翻译"""
    if not text.strip():
        return text

    prompt = f"""请将以下英文技术文档翻译为中文。

翻译要求：
1. 采用专业的计算机技术书籍翻译风格
2. 技术术语使用业界通用译法
3. 保持Markdown格式完整
4. 数学公式和代码保持原样
5. 语言流畅自然，符合中文阅读习惯

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
        print(f"    ⚠️ 错误: {str(e)}")
        return text


def split_content(content, max_length=2000):
    """智能分割内容"""
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
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()

        if len(content) > 2000:
            chunks = split_content(content)
            print(f"  {len(chunks)}块", end=' ')

            translated = []
            for chunk in chunks:
                result = translate_text(client, chunk)
                translated.append(result)
                print(".", end='', flush=True)
                time.sleep(1.2)

            final = '\n\n'.join(translated)
        else:
            print(f"  翻译", end=' ', flush=True)
            final = translate_text(client, content)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(final)

        print(" ✓")
        return True

    except Exception as e:
        print(f" ✗ {str(e)}")
        return False


def main():
    print("=" * 60)
    print("深度学习书籍完整翻译 - 英文→中文")
    print("=" * 60)

    input_path = Path(INPUT_DIR)
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(exist_ok=True)

    print(f"\n📂 输入: {INPUT_DIR}")
    print(f"📁 输出: {OUTPUT_DIR}")

    print("\n初始化...", end=' ')
    client = init_client()
    print("✓")

    md_files = sorted(list(input_path.glob("*.md")))
    print(f"\n找到 {len(md_files)} 个文件")
    print(f"⏱️  预计耗时: {len(md_files) * 2} 分钟")
    print("\n开始翻译...\n")

    start = time.time()
    success = 0

    for i, md_file in enumerate(md_files, 1):
        output_file = output_path / md_file.name

        # 跳过已翻译的文件
        if output_file.exists():
            print(f"[{i:3d}/{len(md_files)}] {md_file.name[:40]:40s} (已存在)")
            success += 1
            continue

        print(f"[{i:3d}/{len(md_files)}] {md_file.name[:40]:40s}", end='')

        if translate_file(client, md_file, output_file):
            success += 1

        # 避免API限流
        if i < len(md_files):
            time.sleep(1.5)

    elapsed = (time.time() - start) / 60

    print("\n" + "=" * 60)
    print(f"✅ 完成: {success}/{len(md_files)} 个文件")
    print(f"⏱️  耗时: {elapsed:.1f} 分钟")
    print(f"📁 输出: {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
