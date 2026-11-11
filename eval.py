"""
Grade the RAG chatbot against the 30-question eval set.

It writes eval_results.csv with, for each question: the bot's answer,
which PDF pages were retrieved, whether the right page was among them,
and how long the whole answer took.
"""

import csv
import statistics
import time

from langchain_core.prompts import PromptTemplate

# Reuse the exact retriever, LLM and helper the app is built on
from rag_chain import retriever, llm, format_docs

# Same prompt as app.py, so the scores reflect the live app
prompt = PromptTemplate.from_template("""
Answer the question using ONLY the context below. The answer is often a specific
number, name, or list, so read every passage carefully before saying it is missing.
Do not add facts, examples, or models that are not in the context.
If the context truly doesn't contain the answer, say "I don't know."

Context: {context}

Question: {question}

Answer:""")

# (number, type, question, expected PDF pages; empty = not in the paper)
QUESTIONS = [
    (1, "Lookup", "What topic modeling method does the paper use to categorize research?", [1]),
    (2, "Lookup", "How many papers did the authors retrieve for their dataset?", [4]),
    (3, "Lookup", "What years were the reviewed papers published between?", [4]),
    (4, "Lookup", "Which database did the authors use to find papers?", [4]),
    (5, "Lookup", "What two search phrases did the authors use?", [4]),
    (6, "Lookup", "What framework was used to turn each abstract into a vector embedding?", [5]),
    (7, "Lookup", "Which dimension reduction technique did the BERTopic setup use?", [5]),
    (8, "Lookup", "Which clustering algorithm was used, and what is its benefit here?", [5]),
    (9, "Lookup", "Who introduced the GAN model, and in what year?", [2]),
    (10, "Lookup", "Who developed the VAE algorithm, and when?", [2]),
    (11, "Lookup", "What is FinVis-GPT?", [9]),
    (12, "Lookup", "What is FinanceBench used for?", [10]),
    (13, "Lookup", "What three stages did Roychowdhury propose for hallucination-minimized LLM solutions?", [10]),
    (14, "Lookup", "Which journal published this paper, and on what date?", [1]),
    (15, "Lookup", "What three focus areas for synthetic financial data did Assefa et al. highlight?", [11]),
    (16, "Cross-section", "How many papers fell into each BERTopic cluster, and what does each cluster represent?", [5, 7]),
    (17, "Cross-section", "What methods for reducing hallucination does the paper mention, and how does hallucination show up in its future research agenda?", [10, 14]),
    (18, "Cross-section", "Why do the authors prefer BERTopic over LDA/NMF, and how does that tie to their critique of earlier financial text mining?", [3, 5]),
    (19, "Cross-section", "Which generative models are used for synthetic financial data, and which is most common?", [7, 12]),
    (20, "Cross-section", "How does the paper connect synthetic data to benchmarking finance LLMs?", [10, 14]),
    (21, "Cross-section", "Which two clusters are most similar in the dendrogram, and what topics are they?", [5, 8]),
    (22, "Cross-section", "Give examples of general-purpose vs. finance-specific LLMs and explain how they differ.", [8, 9]),
    (23, "Cross-section", "What does the paper say about LLMs in non-English languages?", [9, 10]),
    (24, "Number/table", "How many papers were classified as outliers (cluster -1)?", [7]),
    (25, "Number/table", "How many papers discuss LLMs in finance?", [6, 7]),
    (26, "Number/table", "How many datasets and tasks does Xie et al.'s open-source evaluation benchmark cover?", [10]),
    (27, "Number/table", "At about what dissimilarity level do clusters 0 and 1 merge?", [8]),
    (28, "Unanswerable", "What accuracy did BloombergGPT score on FinanceBench?", []),
    (29, "Unanswerable", "Which of the 90 reviewed papers was cited the most?", []),
    (30, "Unanswerable", "What was IBM's total revenue in 2023?", []),
]


def pdf_page(doc):
    """PyPDFLoader stores pages counting from 0; the sheet counts from 1."""
    page = doc.metadata.get("page")
    return int(page) + 1 if page is not None else None


def main():
    rows = []
    for num, qtype, question, expected_pages in QUESTIONS:
        print(f"[{num}/30] {question}")
        start = time.time()

        # Retrieve once, then answer from those same chunks,
        # so the page check matches what the LLM actually saw.
        docs = retriever.invoke(question)
        answer = llm.invoke(prompt.format(context=format_docs(docs), question=question))

        latency = round(time.time() - start, 2)
        retrieved = sorted({p for p in (pdf_page(d) for d in docs) if p})

        if expected_pages:
            right_page = any(p in retrieved for p in expected_pages)
        else:
            right_page = "N/A"  # unanswerable: nothing to retrieve

        rows.append({
            "#": num,
            "type": qtype,
            "question": question,
            "bot_answer": answer.strip(),
            "retrieved_pages": ", ".join(map(str, retrieved)),
            "expected_pages": ", ".join(map(str, expected_pages)) or "N/A",
            "right_page_retrieved": right_page,
            "latency_s": latency,
        })
        print(f"    pages {retrieved} | right page: {right_page} | {latency}s\n")

    with open("eval_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    hits = [r for r in rows if r["right_page_retrieved"] is True]
    answerable = [r for r in rows if r["right_page_retrieved"] != "N/A"]
    print("Done. Saved eval_results.csv")
    print(f"Right page retrieved: {len(hits)} / {len(answerable)}")
    print(f"Median latency: {statistics.median(r['latency_s'] for r in rows):.2f}s")


if __name__ == "__main__":
    main()
