"""
Document Chunking with Semantic-Aware Strategy.

This module implements a sophisticated chunking strategy that:
- Uses semantic boundaries (paragraphs, sections) for chunking
- Maintains context by keeping related content together
- Applies vendor-specific configurations for different document types
- Filters out irrelevant content (headers, footers, page numbers)

Key chunking features:
- Semantic-aware splitting based on document structure
- Configurable chunk sizes and overlap
- Vendor-specific content filtering
- Context preservation across chunks

The chunking strategy is crucial for:
- Maintaining semantic coherence in chunks
- Preserving document structure and context
- Enabling accurate retrieval and generation
- Supporting different document formats and vendors
"""

import os
import fitz
import re
from pathlib import Path
from transformers import AutoTokenizer
from ..config import get_vendor_config, EMBEDDING_MODEL_NAME
from ..logger import setup_logger

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Setup logger
logger = setup_logger("chunking", "chunking.log")

MAX_TOKENS = 512

tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)

def count_tokens_transformers(text):
    return len(tokenizer.encode(text, add_special_tokens=False))

def split_text_semantically(text, max_tokens):
    logger.info(f"Starting semantic text splitting, max tokens: {max_tokens}")
    try:
        if count_tokens_transformers(text) <= max_tokens:
            logger.info("Text within token limit, returning as single chunk")
            return [text]

        lines = text.splitlines()
        segments = []
        buffer = ""

        for line in lines:
            tentative = buffer + "\n" + line if buffer else line
            if count_tokens_transformers(tentative) > max_tokens:
                if buffer:
                    segments.append(buffer.strip())
                    logger.info(f"Created new segment with {count_tokens_transformers(buffer)} tokens")
                buffer = line
            else:
                buffer = tentative

        if buffer:
            segments.append(buffer.strip())
            logger.info(f"Created final segment with {count_tokens_transformers(buffer)} tokens")

        logger.info(f"Successfully split text into {len(segments)} segments")
        return segments
    except Exception as e:
        logger.error(f"Error during semantic text splitting: {str(e)}")
        raise

# Patterns to filter out irrelevant lines like headers/footers
def is_irrelevant(line, irrelevant_patterns):
    return any(re.match(pat, line.strip()) for pat in irrelevant_patterns)

def is_section_title(line):
    return re.match(r"^\d+(\.\d+)+\s+.+", line.strip()) is not None

def is_chapter_title(line):
    return re.match(r"^Chapter\s+\d+[:：]", line.strip(), re.IGNORECASE)



def parse_pdf_by_chapter_section_split(pdf_path, start_page, end_page, irrelevant_patterns):
    logger.info(f"Starting PDF parsing: {pdf_path}, pages {start_page}-{end_page}")
    doc = fitz.open(pdf_path)
    logger.info(f"Successfully opened PDF with {len(doc)} pages")
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
                logger.info(f"Found new section: {current_title}")
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

    logger.info(f"Successfully created {len(chunks)} chunks from document")
    return chunks


def chunks_app(pdf_path, vendor = "broadcom_sonic"):
    logger.info(f"Starting document chunking for: {pdf_path}, vendor: {vendor}")
    try:
        cfg = get_vendor_config(vendor)
        logger.info(f"Retrieved vendor configuration for: {vendor}")
        
        start_page = cfg["start_page"]
        end_page = cfg["end_page"]
        irrelevant_patterns = cfg["ignore_patterns"]
        
        chunks = parse_pdf_by_chapter_section_split(pdf_path, start_page, end_page, irrelevant_patterns)
        logger.info(f"Successfully completed chunking process")
        return chunks
    except Exception as e:
        logger.error(f"Error during document chunking: {str(e)}")
        raise