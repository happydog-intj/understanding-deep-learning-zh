#!/usr/bin/env python3
"""
PDF章节拆分工具
将PDF文件按章节拆分并转换为Markdown格式
"""

import os
import re
from pathlib import Path

try:
    import pdfplumber
except ImportError:
    print("请先安装pdfplumber: pip install pdfplumber")
    exit(1)


def extract_text_from_pdf(pdf_path):
    """从PDF中提取所有文本和页码信息"""
    pages_content = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                pages_content.append({
                    'page_num': i + 1,
                    'text': text
                })
    return pages_content


def detect_chapters(pages_content):
    """
    检测章节标题
    通用的章节检测模式，支持多种格式：
    - Chapter 1: Title
    - Chapter 1 Title
    - 第一章 标题
    - 1. Title
    等
    """
    chapters = []

    # 章节标题的常见模式
    patterns = [
        r'^Chapter\s+(\d+)[:\s]+(.+?)$',  # Chapter 1: Introduction
        r'^Chapter\s+(\d+)\s*$',  # Chapter 1
        r'^CHAPTER\s+(\d+)[:\s]+(.+?)$',  # CHAPTER 1: Introduction
        r'^第([一二三四五六七八九十百]+)章[:\s]+(.+?)$',  # 第一章: 标题
        r'^(\d+)\.\s+(.+?)$',  # 1. Introduction
        r'^Part\s+([IVX]+)[:\s]+(.+?)$',  # Part I: Introduction
    ]

    for page_info in pages_content:
        lines = page_info['text'].split('\n')

        for line_idx, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            # 尝试匹配各种章节模式
            for pattern in patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    chapter_num = match.group(1)
                    chapter_title = match.group(2) if len(match.groups()) > 1 else ""

                    chapters.append({
                        'number': chapter_num,
                        'title': chapter_title.strip(),
                        'start_page': page_info['page_num'],
                        'full_title': line
                    })
                    break

    return chapters


def split_content_by_chapters(pages_content, chapters):
    """根据章节信息拆分内容"""
    chapter_contents = []

    for i, chapter in enumerate(chapters):
        start_page = chapter['start_page']
        end_page = chapters[i + 1]['start_page'] - 1 if i + 1 < len(chapters) else pages_content[-1]['page_num']

        content = []
        for page_info in pages_content:
            if start_page <= page_info['page_num'] <= end_page:
                content.append(page_info['text'])

        chapter_contents.append({
            'chapter': chapter,
            'content': '\n\n'.join(content),
            'page_range': f"{start_page}-{end_page}"
        })

    return chapter_contents


def convert_to_markdown(chapter_data):
    """将章节内容转换为Markdown格式"""
    chapter = chapter_data['chapter']
    content = chapter_data['content']
    page_range = chapter_data['page_range']

    # 构建Markdown文档
    markdown = []

    # 添加标题
    if chapter['title']:
        markdown.append(f"# Chapter {chapter['number']}: {chapter['title']}\n")
    else:
        markdown.append(f"# Chapter {chapter['number']}\n")

    # 添加元信息
    markdown.append(f"*Pages: {page_range}*\n")
    markdown.append("---\n")

    # 添加内容
    markdown.append(content)

    return '\n'.join(markdown)


def save_chapters(chapter_contents, output_dir):
    """保存章节到文件"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print(f"\n保存章节到目录: {output_path}")

    for chapter_data in chapter_contents:
        chapter_num = chapter_data['chapter']['number']
        chapter_title = chapter_data['chapter']['title']

        # 生成文件名
        if chapter_title:
            # 清理标题中的特殊字符
            safe_title = re.sub(r'[^\w\s-]', '', chapter_title)
            safe_title = re.sub(r'[-\s]+', '_', safe_title)
            filename = f"chapter_{chapter_num}_{safe_title}.md"
        else:
            filename = f"chapter_{chapter_num}.md"

        # 转换为Markdown
        markdown_content = convert_to_markdown(chapter_data)

        # 保存文件
        file_path = output_path / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)

        print(f"  ✓ 已保存: {filename} (Pages: {chapter_data['page_range']})")


def main():
    pdf_file = "UnderstandingDeepLearning_05_29_25_C.pdf"
    output_dir = "chapters_markdown"

    print(f"正在处理PDF文件: {pdf_file}")

    # 检查文件是否存在
    if not os.path.exists(pdf_file):
        print(f"错误: 找不到文件 {pdf_file}")
        return

    # 1. 提取PDF文本
    print("步骤 1/4: 提取PDF文本...")
    pages_content = extract_text_from_pdf(pdf_file)
    print(f"  已提取 {len(pages_content)} 页内容")

    # 2. 检测章节
    print("步骤 2/4: 检测章节...")
    chapters = detect_chapters(pages_content)

    if not chapters:
        print("  警告: 未检测到章节标题，将尝试手动检查...")
        print("\n前5页的内容预览:")
        for page_info in pages_content[:5]:
            print(f"\n--- Page {page_info['page_num']} ---")
            print(page_info['text'][:500])
        return

    print(f"  检测到 {len(chapters)} 个章节:")
    for chapter in chapters:
        print(f"    - Chapter {chapter['number']}: {chapter['title']} (Page {chapter['start_page']})")

    # 3. 拆分内容
    print("步骤 3/4: 拆分章节内容...")
    chapter_contents = split_content_by_chapters(pages_content, chapters)

    # 4. 保存为Markdown
    print("步骤 4/4: 保存为Markdown文件...")
    save_chapters(chapter_contents, output_dir)

    print(f"\n完成! 所有章节已保存到 {output_dir}/ 目录")


if __name__ == "__main__":
    main()
