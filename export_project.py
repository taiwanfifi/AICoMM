#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
專案檔案匯出工具
掃描指定資料夾下的所有 .py、.md 和 .json 檔案，並將內容整理輸出到 .txt 檔案
格式便於 AI 讀取和理解專案結構
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple


def find_files(folder_path: str, extensions: List[str]) -> List[Tuple[str, str]]:
    """
    遞迴查找指定副檔名的所有檔案
    
    Args:
        folder_path: 要掃描的資料夾路徑
        extensions: 檔案副檔名列表，如 ['.py', '.md', '.json']
    
    Returns:
        檔案列表，每個元素為 (相對路徑, 絕對路徑) 的元組
    """
    files = []
    folder = Path(folder_path).resolve()
    
    if not folder.exists():
        raise FileNotFoundError(f"資料夾不存在: {folder_path}")
    
    if not folder.is_dir():
        raise ValueError(f"路徑不是資料夾: {folder_path}")
    
    # 遞迴掃描所有檔案
    for root, dirs, filenames in os.walk(folder):
        # 跳過 __pycache__ 和 .git 等目錄
        dirs[:] = [d for d in dirs if d not in ['__pycache__', '.git', '.vscode', '.idea', 'node_modules']]
        
        for filename in filenames:
            file_path = Path(root) / filename
            if file_path.suffix.lower() in extensions:
                # 計算相對路徑
                try:
                    rel_path = file_path.relative_to(folder)
                except ValueError:
                    rel_path = file_path
                
                files.append((str(rel_path), str(file_path)))
    
    # 按路徑排序，確保輸出順序一致
    files.sort(key=lambda x: x[0])
    return files


def read_file_content(file_path: str) -> str:
    """
    讀取檔案內容，嘗試使用 UTF-8 編碼，失敗則嘗試其他編碼
    
    Args:
        file_path: 檔案路徑
    
    Returns:
        檔案內容字串
    """
    encodings = ['utf-8', 'utf-8-sig', 'gbk', 'latin-1']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"警告: 無法讀取檔案 {file_path}: {e}")
            return f"[無法讀取檔案: {e}]"
    
    return "[檔案編碼無法識別]"


def export_project(folder_path: str, output_file: str = None) -> str:
    """
    匯出專案檔案到文字檔案
    
    Args:
        folder_path: 要掃描的資料夾路徑
        output_file: 輸出檔案路徑，如果為 None 則自動產生
    
    Returns:
        輸出檔案的路徑
    """
    # 查找所有相關檔案
    extensions = ['.py', '.md', '.json']
    print(f"正在掃描資料夾: {folder_path}")
    files = find_files(folder_path, extensions)
    
    if not files:
        print("未找到任何 .py、.md 或 .json 檔案")
        return None
    
    print(f"找到 {len(files)} 個檔案")
    
    # 產生輸出檔案名稱
    if output_file is None:
        folder_name = Path(folder_path).name or "project"
        output_file = os.path.join(folder_path, f"{folder_name}_export.txt")
    
    # 寫入檔案
    print(f"正在匯出到: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as out:
        # 寫入檔案標頭
        out.write("=" * 80 + "\n")
        out.write(f"專案檔案匯出\n")
        out.write(f"來源資料夾: {os.path.abspath(folder_path)}\n")
        out.write(f"檔案總數: {len(files)}\n")
        out.write("=" * 80 + "\n\n")
        
        # 寫入每個檔案的內容
        for idx, (rel_path, abs_path) in enumerate(files, 1):
            print(f"處理 [{idx}/{len(files)}]: {rel_path}")
            
            # 寫入檔案標題
            out.write("\n" + "=" * 80 + "\n")
            out.write(f"檔案 {idx}/{len(files)}: {rel_path}\n")
            out.write(f"完整路徑: {abs_path}\n")
            out.write("=" * 80 + "\n\n")
            
            # 讀取並寫入檔案內容
            content = read_file_content(abs_path)
            out.write(content)
            
            # 如果檔案末尾沒有換行，新增一個
            if content and not content.endswith('\n'):
                out.write('\n')
            
            out.write("\n" + "-" * 80 + "\n\n")
    
    print(f"\n匯出完成！輸出檔案: {output_file}")
    print(f"共處理 {len(files)} 個檔案")
    
    return output_file


def main():
    """主函式"""
    print("=" * 80)
    print("專案檔案匯出工具")
    print("=" * 80)
    print()
    
    # 取得輸入資料夾路徑
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
    else:
        folder_path = input("請輸入要掃描的資料夾路徑: ").strip()
    
    # 移除引號（如果使用者輸入時帶了引號）
    folder_path = folder_path.strip('"').strip("'")
    
    # 取得輸出檔案路徑（可選）
    output_file = None
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    try:
        result = export_project(folder_path, output_file)
        if result:
            print(f"\n✅ 成功！檔案已儲存到: {result}")
        else:
            print("\n❌ 匯出失敗")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ 錯誤: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

