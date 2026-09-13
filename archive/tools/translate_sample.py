#!/usr/bin/env python3
"""
Markdown文件翻译工具 - 示例版本（翻译前5个文件）
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
OUTPUT_DIR = "chapters_translated"
MAX_FILES = 5  # 只翻译前5个文件作为测试


def init_client():
    """初始化OpenAI客户端"""
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    return client


def translate_text(client, text):
    """使用Qwen翻译文本"""
    if not text.strip():
        return text

    # 检查是否已经主要是英文
    english_chars = sum(1 for c in text if ord(c) < 128)
    if english_chars / len(text) > 0.8:
        print("    (内容已是英文，跳过翻译)")
        return text

    prompt = f"""请将以下文本翻译为英文。保持原有的Markdown格式。

{text}

只输出翻译后的内容，不要添加任何解释。"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "你是专业的技术文档翻译助手。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=4000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"    ❌ 翻译失败: {str(e)}")
        return text


def split_content(content, max_length=2500):
    """分割长文本"""
    if len(content) <= max_length:
        return [content]

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
    """翻译单个文件"""
    print(f"\n处理: {input_path.name}")

    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()

        print(f"  大小: {len(content)} 字符")

        if len(content) > 2500:
            chunks = split_content(content)
            print(f"  分为 {len(chunks)} 块")

            translated_chunks = []
            for i, chunk in enumerate(chunks):
                print(f"  翻译块 {i+1}/{len(chunks)}...", end=' ')
                translated = translate_text(client, chunk)
                translated_chunks.append(translated)
                print("✓")
                time.sleep(1)

            translated_content = '\n\n'.join(translated_chunks)
        else:
            print(f"  翻译中...", end=' ')
            translated_content = translate_text(client, content)
            print("✓")

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(translated_content)

        print(f"  ✅ 已保存")
        return True

    except Exception as e:
        print(f"  ❌ 失败: {str(e)}")
        return False


def main():
    print("=" * 60)
    print("Markdown翻译测试 - 翻译前5个文件")
    print("=" * 60)

    input_path = Path(INPUT_DIR)
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(exist_ok=True)

    print(f"\n输入目录: {INPUT_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")

    print("\n初始化客户端...", end=' ')
    client = init_client()
    print("✓")

    md_files = sorted(list(input_path.glob("*.md")))[:MAX_FILES]
    print(f"\n将翻译 {len(md_files)} 个文件")

    success = 0
    for i, md_file in enumerate(md_files, 1):
        print(f"\n[{i}/{len(md_files)}]", end=' ')
        output_file = output_path / md_file.name

        if translate_file(client, md_file, output_file):
            success += 1

        if i < len(md_files):
            time.sleep(2)

    print("\n" + "=" * 60)
    print(f"完成: {success}/{len(md_files)} 个文件成功")
    print("=" * 60)


if __name__ == "__main__":
    main()
