"""
Configuration settings for the RAG system.

This module contains all configuration parameters for:
- Document chunking settings for different vendors
- Embedding model configuration
- Vector store table names
- Language model settings

The configuration is centralized here to make it easy to:
- Modify system parameters
- Add support for new vendors
- Update model settings
- Maintain consistent configuration across the system
"""

# Chunking Config

VENDOR_CONFIGS = {
    "broadcom_sonic": {
        "start_page": 24,
        "end_page": 840,
        # Patterns to filter out irrelevant lines like headers/footers
        "ignore_patterns": [
            r"^Broadcom Confidential",
            r"^SONiC[- ]?UG\d+",
            r"^SONiC\s+\d+\.\d+\.\d+",
            r"^\d{1,4}$",
            r"^Enterprise SONiC.*User Guide"
        ]
    },
    "arista_eos": {
        "start_page": 10,
        "end_page": 350,
        "ignore_patterns": [
            r"^Arista Networks",
            r"^EOS\s+\d+\.\d+\.\d+",
            r"^\d{1,4}$",
            r"^Arista EOS Configuration Manual.*"
        ]
    },
    "cisco_nxos": {
        "start_page": 12,
        "end_page": 710,
        "ignore_patterns": [
            r"^Cisco Systems",
            r"^NX-OS Version.*",
            r"^\d{1,4}$",
            r"^Cisco NX-OS Configuration Guide.*"
        ]
    }
    # ... Add more vendors here
}


def get_vendor_config(vendor_name: str):
    if vendor_name not in VENDOR_CONFIGS:
        raise ValueError(f"Vendor '{vendor_name}' not supported.")
    return VENDOR_CONFIGS[vendor_name]


# Embedding Config
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en"
EMBED_DIM = 384


# VB Table Name
AVGO_TABLE_NAME = "broadcom_sonic"

# LLM 
LLM_NAME = "deepseek/deepseek-chat-v3-0324:free"