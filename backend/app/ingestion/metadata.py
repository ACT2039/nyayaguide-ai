import json
from pathlib import Path
from typing import Dict, Any, Optional
from ..config import METADATA_FILE, TARGET_DOCUMENTS


# Canonical fallback metadata in case documents.json is missing or corrupted
DEFAULT_DOCUMENT_METADATA: Dict[str, Dict[str, Any]] = {
    "RTI_Act_2005.pdf": {
        "filename": "RTI_Act_2005.pdf",
        "category": "RTI",
        "title": "The Right to Information Act, 2005",
        "source": "India Code",
        "source_url": "https://cic.gov.in/sites/default/files/RTI-Act_English.pdf"
    },
    "RTI_Rules_2012.pdf": {
        "filename": "RTI_Rules_2012.pdf",
        "category": "RTI",
        "title": "The Right to Information Rules, 2012",
        "source": "Government of India / Central Information Commission",
        "source_url": "https://nationalarchives.nic.in//sites/default/files/2023-08/RTIRules2012.pdf"
    },
    "Consumer_Protection_Act_2019.pdf": {
        "filename": "Consumer_Protection_Act_2019.pdf",
        "category": "CONSUMER",
        "title": "The Consumer Protection Act, 2019",
        "source": "Department of Consumer Affairs, Government of India",
        "source_url": "https://egazette.gov.in/WriteReadData/2019/210422.pdf"
    },
    "Consumer_Commission_and_General_Rules_2020.pdf": {
        "filename": "Consumer_Commission_and_General_Rules_2020.pdf",
        "category": "CONSUMER",
        "title": "Consumer Protection (Consumer Disputes Redressal Commissions) Rules, 2020 and Consumer Protection (General) Rules, 2020",
        "source": "Department of Consumer Affairs, Government of India",
        "source_url": "https://egazette.gov.in/WriteReadData/2020/220555.pdf"
    }
}


class MetadataRegistry:
    """Manages document metadata and category mapping for NyayaGuide AI."""

    def __init__(self, metadata_path: Optional[Path] = None):
        self.metadata_path = metadata_path or METADATA_FILE
        self.registry: Dict[str, Dict[str, Any]] = {}
        self._load_registry()

    def _load_registry(self) -> None:
        """Load document metadata from JSON file or fall back to defaults."""
        if self.metadata_path and Path(self.metadata_path).exists():
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data.get("documents", []):
                        fn = item.get("filename")
                        if fn:
                            self.registry[fn] = item
            except Exception as e:
                print(f"Warning: Could not parse metadata file ({e}), using default metadata.")

        # Ensure all target documents have entries
        for fn, meta in DEFAULT_DOCUMENT_METADATA.items():
            if fn not in self.registry:
                self.registry[fn] = meta

    def get_metadata(self, filename: str) -> Dict[str, Any]:
        """Retrieve metadata for a given PDF filename."""
        if filename in self.registry:
            return self.registry[filename]
        
        # Default fallback
        category = "RTI" if "RTI" in filename.upper() else "CONSUMER"
        return {
            "filename": filename,
            "category": category,
            "title": filename.replace(".pdf", "").replace("_", " "),
            "source": "Government of India",
            "source_url": None
        }

    def get_category(self, filename: str) -> str:
        """Return canonical category (RTI or CONSUMER)."""
        return self.get_metadata(filename).get("category", "CIVIC")

    def get_title(self, filename: str) -> str:
        """Return official document title."""
        return self.get_metadata(filename).get("title", filename)

    def get_source(self, filename: str) -> str:
        """Return official document source."""
        return self.get_metadata(filename).get("source", "Government of India")

    def get_source_url(self, filename: str) -> Optional[str]:
        """Return official source URL."""
        return self.get_metadata(filename).get("source_url")
