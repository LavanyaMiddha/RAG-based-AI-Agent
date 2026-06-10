from pathlib import Path

import pymupdf
import fitz
import pymupdf4llm
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter

class DataLoader:

    def _get_markdown(self, file_path: str) ->str:
        markdown = pymupdf4llm.to_markdown(file_path)
        # print(markdown)
        return markdown
    
    # def _get_doc_keywords(self, markdown:str)->list[str]:
    #     kw_model = KeyBERT()
    #     keywords = kw_model.extract_keywords(markdown, use_mmr=True, keyphrase_ngram_range=(1, 2), stop_words='english', diversity=0.7)
    #     print(keywords)
    #     return ""


    def _get_doc_metadata(self, file_path:str):
        doc = fitz.open(file_path)
        #doc_keywords = self._get_doc_keywords()
        pdf_metadata = doc.metadata or {}
        return {
            "source": str(file_path),
            "filename": Path(file_path).name,
            "title": pdf_metadata.get("title"),
            "author": pdf_metadata.get("author"),
            # "keywords": "",
            "creation_date": pdf_metadata.get("creationDate"),
            "modified_date": pdf_metadata.get("modDate"),
            "document_type": "pdf",
        }

    
    
    def split_markdown(self, file_path:str) -> list:
        markdown = self._get_markdown(file_path)
        headers_to_split_on = [
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
            ("####", "h4"),
        ]

        # Split into logical sections first
        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False,
        )

        header_docs = header_splitter.split_text(markdown)
        base_metadata = self._get_doc_metadata(file_path)

        # print(header_docs)

        recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
        separators=[
            "\n## ",
            "\n### ",
            "\n#### ",
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
            ],
        )

        for d in header_docs:
            d.metadata = {
                **base_metadata,
                **d.metadata,
            }

        # single recursive pass
        final_chunks = recursive_splitter.split_documents(header_docs)

        return final_chunks
    

    def load_pdf(self, file_path:str):
        self._get_doc_metadata(file_path)
        documents = self.split_markdown(str(file_path))
        return documents

    # def load_pdfs(self, file_directory: str):
    #     files = list(Path(file_directory).rglob("*.pdf"))
    #     print(f"Found {len(files)} files:")
    #     for f in files:
    #         print(f)
    #     all_documents = []
    #     for file_path in files:
    #         self._get_doc_metadata(file_path)
    #         all_documents.extend(self.split_markdown(str(file_path)))
    #     return all_documents
    
if __name__ == "__main__":
    data_loader = DataLoader()
    documents = data_loader.load_pdf("C:/RAG-based-AI-Agent/data/contract_files/contract_01_standard_clean.pdf")
    print(documents[0])
    print(type(documents[0]))