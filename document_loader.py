import os
import fitz  # PyMuPDF
from bs4 import BeautifulSoup
import markdown
from typing import List, Dict, Any

class DocumentLoader:
    @staticmethod
    def load_document(file_path: str) -> List[Dict[str, Any]]:
        """
        Loads document based on file extension and returns a list of sections/pages 
        with text and metadata.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            return DocumentLoader._load_pdf(file_path)
        elif ext == ".html" or ext == ".htm":
            return DocumentLoader._load_html(file_path)
        elif ext == ".md" or ext == ".markdown":
            return DocumentLoader._load_markdown(file_path)
        else:
            # Fallback to plain text load
            return DocumentLoader._load_text(file_path)
            
    @staticmethod
    def _load_pdf(file_path: str) -> List[Dict[str, Any]]:
        docs = []
        doc = fitz.open(file_path)
        filename = os.path.basename(file_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            docs.append({
                "text": text,
                "metadata": {
                    "source": filename,
                    "page": page_num + 1,
                    "type": "pdf"
                }
            })
        return docs

    @staticmethod
    def _load_html(file_path: str) -> List[Dict[str, Any]]:
        filename = os.path.basename(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
            # Remove scripts and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text(separator="\n")
            # Clean empty lines
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            cleaned_text = "\n".join(lines)
            
        return [{
            "text": cleaned_text,
            "metadata": {
                "source": filename,
                "type": "html"
            }
        }]

    @staticmethod
    def _load_markdown(file_path: str) -> List[Dict[str, Any]]:
        filename = os.path.basename(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            raw_content = f.read()
            # Convert markdown to html to extract text reliably
            html_content = markdown.markdown(raw_content)
            soup = BeautifulSoup(html_content, "html.parser")
            text = soup.get_text()
            
        return [{
            "text": text,
            "metadata": {
                "source": filename,
                "type": "markdown"
            }
        }]

    @staticmethod
    def _load_text(file_path: str) -> List[Dict[str, Any]]:
        filename = os.path.basename(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        return [{
            "text": text,
            "metadata": {
                "source": filename,
                "type": "txt"
            }
        }]
