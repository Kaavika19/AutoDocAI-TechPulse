# AutoDoc AI — Automotive Document Assistant

A small evidence-first prototype that extracts text from an automotive PDF and retrieves passages relevant to a user question, with source page references.

## Requirements
- Python 3.11
- Anaconda Prompt or terminal

## Run on Windows with Anaconda
1. Extract this ZIP.
2. Open Anaconda Prompt and activate your environment:
   ```bash
   conda activate autodoc
   ```
3. Change directory to the folder containing `app.py` and `requirements.txt`.
4. Install dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```
5. Start the app:
   ```bash
   streamlit run app.py
   ```
6. Open `http://localhost:8501` in your browser if it does not open automatically.

## What it does
- Extracts selectable text from PDFs using PyMuPDF.
- Cleans some common repeated PDF header/footer text.
- Splits text into overlapping passages.
- Ranks passages using TF-IDF word and phrase matching.
- Displays source page references and similarity ranking scores.
- Lets users download retrieved evidence as a text file.

## Limitations
This prototype does not use a generative LLM, embeddings, a vector database, OCR, or formal AUTOSAR validation. Similarity scores are ranking scores, not accuracy/confidence scores. Always verify evidence against the source PDF.
