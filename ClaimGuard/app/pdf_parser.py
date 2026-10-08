import fitz
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path

from app.utils import clean_text, split_sentences, generate_id


@dataclass
class Sentence:
    sentence_id: str
    page: int
    section: str
    text: str
    bbox: Optional[List[float]] = None


@dataclass
class Document:
    filename: str
    num_pages: int
    word_count: int
    sections: List[str]
    sentences: List[Sentence]
    references_start_page: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class PDFParser:
    def __init__(self):
        self.section_patterns = [
            r"^\d+\.?\s+[A-Z][A-Za-z\s]+$",
            r"^[A-Z][A-Z\s]+$",
            r"^(Abstract|Introduction|Related Work|Background|Methodology|Methods|Experiments|Results|Discussion|Conclusion|References|Acknowledgements)$",
        ]

    def parse(self, file_path: str) -> Document:
        doc = fitz.open(file_path)
        filename = Path(file_path).name
        
        all_text = ""
        sentences = []
        sections = []
        current_section = "Unknown"
        references_start_page = None
        word_count = 0
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            blocks = page.get_text("dict")["blocks"]
            
            page_section = self._detect_section(text, blocks)
            if page_section:
                current_section = page_section
                if current_section not in sections:
                    sections.append(current_section)
            
            if current_section.lower() == "references" and references_start_page is None:
                references_start_page = page_num + 1
            
            page_sentences = split_sentences(text)
            for sent_text in page_sentences:
                sent_text = clean_text(sent_text)
                if len(sent_text) < 20:
                    continue
                
                sentence_id = generate_id("S", f"{page_num}{sent_text[:50]}")
                sentence = Sentence(
                    sentence_id=sentence_id,
                    page=page_num + 1,
                    section=current_section,
                    text=sent_text,
                )
                sentences.append(sentence)
                all_text += sent_text + " "
                word_count += len(sent_text.split())
        
        num_pages_total = len(doc)
        doc.close()
        
        return Document(
            filename=filename,
            num_pages=num_pages_total,
            word_count=word_count,
            sections=sections,
            sentences=sentences,
            references_start_page=references_start_page,
        )

    def _detect_section(self, text: str, blocks: List[Dict]) -> Optional[str]:
        lines = text.split("\n")
        for line in lines[:5]:
            line = line.strip()
            if not line:
                continue
            for pattern in self.section_patterns:
                if re.match(pattern, line, re.IGNORECASE):
                    return line
        return None

    def extract_tables(self, file_path: str) -> List[Dict[str, Any]]:
        doc = fitz.open(file_path)
        tables = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            tabs = page.find_tables()
            for table in tabs:
                tables.append({
                    "page": page_num + 1,
                    "data": table.extract(),
                    "bbox": table.bbox,
                })
        
        doc.close()
        return tables


def parse_document(file_path: str) -> Document:
    parser = PDFParser()
    return parser.parse(file_path)