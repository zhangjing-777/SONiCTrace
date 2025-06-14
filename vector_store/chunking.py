import os
import fitz
import re
from pathlib import Path
from transformers import AutoTokenizer
from ..config import get_vendor_config, EMBEDDING_MODEL_NAME

os.environ["TOKENIZERS_PARALLELISM"] = "false"

MAX_TOKENS = 512

tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)

def count_tokens_transformers(text):
    return len(tokenizer.encode(text, add_special_tokens=False))

def split_text_semantically(text, max_tokens):
    if count_tokens_transformers(text) <= max_tokens:
        return [text]

    lines = text.splitlines()
    segments = []
    buffer = ""

    for line in lines:
        tentative = buffer + "\n" + line if buffer else line
        if count_tokens_transformers(tentative) > max_tokens:
            if buffer:
                segments.append(buffer.strip())
            buffer = line
        else:
            buffer = tentative

    if buffer:
        segments.append(buffer.strip())

    return segments

# Patterns to filter out irrelevant lines like headers/footers
def is_irrelevant(line, irrelevant_patterns):
    return any(re.match(pat, line.strip()) for pat in irrelevant_patterns)

def is_section_title(line):
    return re.match(r"^\d+(\.\d+)+\s+.+", line.strip()) is not None

def is_chapter_title(line):
    return re.match(r"^Chapter\s+\d+[:：]", line.strip(), re.IGNORECASE)



def parse_pdf_by_chapter_section_split(pdf_path, start_page, end_page, irrelevant_patterns):
    doc = fitz.open(pdf_path)
    end_page = end_page or len(doc) - 1
    source = Path(pdf_path).name

    chunks = []
    current_title = "UNKNOWN"
    current_text = ""
    current_start_page = start_page

    for i in range(start_page, end_page + 1):
        lines = doc[i].get_text().splitlines()
        for line in lines:
            if is_irrelevant(line, irrelevant_patterns):
                continue

            if is_chapter_title(line) or is_section_title(line):
                if current_text.strip():
                    for part in split_text_semantically(current_text.strip(), MAX_TOKENS):
                        chunks.append({
                            "section": current_title,
                            "content": part,
                            "page_range": [current_start_page, i],
                            "source": source
                        })
                current_title = line.strip()
                current_text = ""
                current_start_page = i
            else:
                current_text += line + "\n"


    if current_text.strip():
        for part in split_text_semantically(current_text.strip(), MAX_TOKENS):
            chunks.append({
                "section": current_title,
                "content": part,
                "page_range": [current_start_page, end_page],
                "source": source
            })

    return chunks


def chunks_app(pdf_path, vendor = "broadcom_sonic"):

    cfg = get_vendor_config(vendor)
    
    start_page = cfg["start_page"]
    end_page = cfg["end_page"]
    irrelevant_patterns = cfg["ignore_patterns"]
    
    return parse_pdf_by_chapter_section_split(pdf_path, start_page, end_page, irrelevant_patterns)