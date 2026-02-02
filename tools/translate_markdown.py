#!/usr/bin/env python3
"""
Markdown文件翻译工具
使用阿里云Qwen3模型将Markdown文件翻译为英文
"""

import os
import re
import time
from pathlib import Path
from openai import OpenAI

# 配置
API_KEY = "sk-dc023de5a33f40aa8932b48e7a1f1d86"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL = "qwen-plus"  # 可选: qwen-turbo, qwen-plus, qwen-max

INPUT_DIR = "chapters_markdown"
OUTPUT_DIR = "chapters_translated"


def init_client():
    """初始化OpenAI客户端（兼容阿里云DashScope）"""
    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL
    )
    return client


def translate_text(client, text, max_retries=3):
    """
    使用Qwen3翻译文本为英文
    """
    if not text.strip():
        return text

    prompt = f"""请将以下中文文本翻译为英文。保持原有的Markdown格式，包括标题、列表、代码块等。
如果文本已经是英文，请保持原样。

文本内容：
{text}

请直接输出翻译后的英文内容，不要添加任何解释或说明。"""

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "你是一个专业的翻译助手，擅长将技术文档从中文翻译为英文，同时保持Markdown格式。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=4000
            )

            translated_text = response.choices[0].message.content.strip()
            return translated_text

        except Exception as e:
            print(f"  ⚠️ 翻译失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                print(f"  ❌ 翻译失败，返回原文")
                return text

    return text


def split_content_into_chunks(content, max_length=3000):
    """
    将长文本按段落分割成多个块
    避免超过API的token限制
    """
    # 按段落分割
    paragraphs = content.split('\n\n')

    chunks = []
    current_chunk = []
    current_length = 0

    for para in paragraphs:
        para_length = len(para)

        if current_length + para_length > max_length and current_chunk:
            # 当前块已满，保存并开始新块
            chunks.append('\n\n'.join(current_chunk))
            current_chunk = [para]
            current_length = para_length
        else:
            current_chunk.append(para)
            current_length += para_length

    # 添加最后一个块
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))

    return chunks


def translate_markdown_file(client, input_path, output_path):
    """
    翻译单个Markdown文件
    """
    print(f"\n正在处理: {input_path.name}")

    try:
        # 读取文件
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()

        print(f"  文件大小: {len(content)} 字符")

        # 如果文件太大，分块处理
        if len(content) > 3000:
            print(f"  文件较大，分块处理...")
            chunks = split_content_into_chunks(content, max_length=3000)
            print(f"  分为 {len(chunks)} 个块")

            translated_chunks = []
            for i, chunk in enumerate(chunks):
                print(f"  翻译块 {i+1}/{len(chunks)}...", end=' ')
                translated_chunk = translate_text(client, chunk)
                translated_chunks.append(translated_chunk)
                print("✓")
                time.sleep(1)  # 避免API限流

            translated_content = '\n\n'.join(translated_chunks)
        else:
            print(f"  翻译中...", end=' ')
            translated_content = translate_text(client, content)
            print("✓")

        # 保存翻译后的文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(translated_content)

        print(f"  ✅ 已保存: {output_path.name}")
        return True

    except Exception as e:
        print(f"  ❌ 处理失败: {str(e)}")
        return False


def main():
    print("=" * 60)
    print("Markdown文件翻译工具 - 使用阿里云Qwen3")
    print("=" * 60)

    # 检查输入目录
    input_path = Path(INPUT_DIR)
    if not input_path.exists():
        print(f"❌ 错误: 找不到输入目录 {INPUT_DIR}")
        return

    # 创建输出目录
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(exist_ok=True)
    print(f"\n输入目录: {INPUT_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"使用模型: {MODEL}")

    # 初始化客户端
    print("\n初始化API客户端...")
    try:
        client = init_client()
        print("✓ 客户端初始化成功")
    except Exception as e:
        print(f"❌ 客户端初始化失败: {str(e)}")
        return

    # 获取所有markdown文件
    md_files = sorted(list(input_path.glob("*.md")))
    print(f"\n找到 {len(md_files)} 个Markdown文件")

    if not md_files:
        print("没有找到任何Markdown文件")
        return

    # 准备开始翻译
    print("\n准备开始翻译...")
    print(f"注意: 这将调用阿里云API，可能产生费用")

    # 翻译所有文件
    print("\n" + "=" * 60)
    print("开始翻译")
    print("=" * 60)

    success_count = 0
    fail_count = 0

    for i, md_file in enumerate(md_files, 1):
        print(f"\n[{i}/{len(md_files)}] ", end='')

        output_file = output_path / md_file.name

        if translate_markdown_file(client, md_file, output_file):
            success_count += 1
        else:
            fail_count += 1

        # 避免API限流
        if i < len(md_files):
            time.sleep(2)

    # 统计结果
    print("\n" + "=" * 60)
    print("翻译完成")
    print("=" * 60)
    print(f"✅ 成功: {success_count} 个文件")
    print(f"❌ 失败: {fail_count} 个文件")
    print(f"📁 输出目录: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
