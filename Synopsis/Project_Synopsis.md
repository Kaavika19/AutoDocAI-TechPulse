# AutoDoc AI — Automotive Document Assistant

## Objective
Build a simple document assistant that retrieves relevant passages from automotive engineering PDFs for a user query.

## Implemented prototype
The Streamlit app extracts selectable PDF text using PyMuPDF, cleans and splits it into overlapping word chunks, then ranks passages using TF-IDF vectors and cosine similarity. Results display passage text, page references, and similarity scores; evidence can be downloaded.

## Limitations
This prototype is retrieval-based. It does not currently implement an LLM, embeddings, a vector database, OCR for scanned PDFs, structured AUTOSAR entity extraction, dependency mapping, document revision comparison, or automatic inconsistency validation.
