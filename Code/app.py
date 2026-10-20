import re
import hashlib
from collections import Counter

import fitz  # PyMuPDF
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AutoDoc AI", page_icon="🚗", layout="wide")

st.title("🚗 AutoDoc AI — Automotive Document Assistant")
st.caption("Evidence-first search for automotive engineering documents")
st.info(
    "Prototype scope: extracts PDF text and retrieves relevant passages. It does not use a generative LLM "
    "or approve engineering decisions. Always verify findings against the source document."
)

# Common repeated PDF furniture that can dominate search results.
BOILERPLATE_PATTERNS = [
    r"confidential\s*\|\s*reference solution document",
    r"automotive engineering ai\s*\|\s*project case studies",
    r"page\s+\d+\s*$",
    r"^\s*version\s+\d+(?:\.\d+)*\s*$",
]


def clean_text(text: str) -> str:
    """Normalize extracted PDF text and remove common repeated header/footer phrases."""
    text = text.replace("\u00a0", " ").replace("\u2011", "-")
    text = re.sub(r"\s+", " ", text).strip()
    for pattern in BOILERPLATE_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip(" |•-\t")
    return text


def make_chunks(text: str, max_words: int = 115, overlap: int = 22):
    """Split page text into overlapping chunks to preserve context at boundaries."""
    words = text.split()
    if not words:
        return []
    step = max(1, max_words - overlap)
    chunks = []
    for start in range(0, len(words), step):
        segment = words[start : start + max_words]
        if len(segment) < 12 and chunks:
            break
        if segment:
            chunks.append(" ".join(segment))
        if start + max_words >= len(words):
            break
    return chunks


def extract_chunks(pdf_bytes: bytes):
    chunks = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page_num, page in enumerate(doc, start=1):
            raw_text = page.get_text("text")
            text = clean_text(raw_text)
            if not text:
                continue
            for chunk_num, chunk_text in enumerate(make_chunks(text), start=1):
                chunks.append({"page": page_num, "chunk": chunk_num, "text": chunk_text})
    return chunks


def retrieve_passages(chunks, question: str, top_k: int):
    """TF-IDF retrieval with unigrams and bigrams; returns passages ranked by similarity."""
    corpus = [item["text"] for item in chunks]
    # Fit only on document passages so the query cannot distort the vocabulary/IDF values.
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
        strip_accents="unicode",
    )
    doc_matrix = vectorizer.fit_transform(corpus)
    query_vector = vectorizer.transform([question])
    scores = cosine_similarity(query_vector, doc_matrix).flatten()
    ranked = scores.argsort()[::-1]

    results = []
    seen_text = set()
    for idx in ranked:
        score = float(scores[idx])
        if score <= 0:
            break
        normalized = re.sub(r"\W+", " ", corpus[idx].lower()).strip()
        fingerprint = hashlib.md5(normalized.encode("utf-8")).hexdigest()
        if fingerprint in seen_text:
            continue
        seen_text.add(fingerprint)
        results.append({**chunks[idx], "score": score})
        if len(results) >= top_k:
            break
    return results


uploaded = st.file_uploader("Upload an automotive engineering PDF", type=["pdf"])

if uploaded:
    file_bytes = uploaded.getvalue()
    file_signature = hashlib.md5(file_bytes).hexdigest()

    if st.session_state.get("file_signature") != file_signature:
        try:
            with st.spinner("Reading PDF and preparing searchable passages..."):
                st.session_state.chunks = extract_chunks(file_bytes)
                st.session_state.file_signature = file_signature
                st.session_state.file_name = uploaded.name
                st.session_state.last_results = []
                st.session_state.last_question = ""
        except Exception as exc:
            st.error(f"Could not read this PDF: {exc}")
            st.stop()

    chunks = st.session_state.get("chunks", [])
    pages_with_text = len({chunk["page"] for chunk in chunks})
    col1, col2, col3 = st.columns(3)
    col1.metric("Passages indexed", len(chunks))
    col2.metric("Pages with text", pages_with_text)
    col3.metric("File size", f"{len(file_bytes) / (1024 * 1024):.2f} MB")
    st.success(f"Loaded **{uploaded.name}** and prepared {len(chunks)} searchable passages.")

    question = st.text_input(
        "Ask a question about the document",
        placeholder="e.g., What are the objectives of the AUTOSAR HLD assistant?",
    )
    top_k = st.slider("Number of evidence passages", min_value=1, max_value=5, value=3)

    with st.expander("Try a sample question"):
        st.write("What are the objectives of the AUTOSAR HLD Document Analysis Assistant?")
        st.write("What technologies are proposed for the solution?")
        st.write("What are the main functional capabilities?")
        st.write("What is the high-level workflow?")

    if st.button("Find evidence", type="primary", disabled=not question.strip()):
        if not chunks:
            st.error("No selectable text was found. Scanned PDFs need OCR, which this version does not include.")
        else:
            try:
                results = retrieve_passages(chunks, question.strip(), top_k)
                st.session_state.last_results = results
                st.session_state.last_question = question.strip()
            except ValueError:
                st.warning("There was not enough readable text to search in this PDF.")

    if st.session_state.get("last_question"):
        st.subheader("Search results")
        st.caption(f"Question: {st.session_state.last_question}")
        results = st.session_state.get("last_results", [])
        if not results:
            st.warning("No matching passages found. Try using key terms from the document or rephrasing your question.")
        else:
            for rank, item in enumerate(results, start=1):
                with st.container(border=True):
                    st.markdown(
                        f"**Evidence {rank} · Page {item['page']} · Passage {item['chunk']} · "
                        f"Similarity {item['score']:.3f}**"
                    )
                    st.write(item["text"])
                    st.caption(f"Source: {st.session_state.get('file_name', uploaded.name)}, page {item['page']}")
            st.caption("Similarity is a ranking score, not answer accuracy or model confidence.")
            st.download_button(
                "Download search results as text",
                data="\n\n".join(
                    f"Evidence {i}. Page {r['page']} | Similarity {r['score']:.3f}\n{r['text']}"
                    for i, r in enumerate(results, start=1)
                ),
                file_name="autodoc_search_results.txt",
                mime="text/plain",
            )
else:
    st.markdown("### How to use")
    st.markdown(
        "1. Upload a text-based automotive PDF.\n"
        "2. Enter a question using concepts from the document.\n"
        "3. Review the highest-ranked passages and verify their page references.\n"
        "4. Download the evidence if you need to include it in a report."
    )
    st.markdown("### Current limitations")
    st.markdown(
        "- Scanned/image-only PDFs are not supported without OCR.\n"
        "- The app retrieves source passages; it does not generate an LLM answer.\n"
        "- TF-IDF relies mainly on word overlap, so paraphrased questions may rank imperfectly.\n"
        "- No document is sent to an external AI service.\n"
        "- Engineering decisions must be reviewed by a qualified human."
    )
