import pdfplumber
from docling.document_converter import DocumentConverter
from markitdown import MarkItDown
import os

def run_conversion_tasks(pdf_path):
    print(f"開始處理檔案: {pdf_path}\n" + "-"*30)

    # 1. 使用 pdfplumber 實作
    try:
        output_md_plumber = "result_pdfplumber.md"
        with pdfplumber.open(pdf_path) as pdf:
            full_text = ""
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n\n"
        with open(output_md_plumber, "w", encoding="utf-8") as f:
            f.write(full_text)
        print("pdfplumber 已完成")
    except Exception as e:
        print(f"pdfplumber 執行失敗: {e}")

    # 2. 使用 Docling 實作
    try:
        output_md_docling = "result_docling.md"
        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        markdown_content = result.document.export_to_markdown()
        with open(output_md_docling, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        print("Docling 已完成")
    except Exception as e:
        print(f"Docling 執行失敗: {e}")

    # 3. 使用 MarkItDown 實作
    try:
        output_md_markitdown = "result_markitdown.md"
        md = MarkItDown()
        result = md.convert(pdf_path)
        with open(output_md_markitdown, "w", encoding="utf-8") as f:
            f.write(result.text_content)
        print("MarkItDown 已完成")
    except Exception as e:
        print(f"MarkItDown 執行失敗: {e}")

if __name__ == "__main__":
    target_pdf = "example.pdf"
    if os.path.exists(target_pdf):
        run_conversion_tasks(target_pdf)
    else:
        print(f"錯誤：在當前目錄找不到 {target_pdf} 檔案。")