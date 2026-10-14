SEMANTIC SEARCH AND EVIDENCE RETRIEVAL

FROM AN AGROECOLOGY TRAINING MANUAL

Seminar Project Report

Author: \[esayas melaku]

Department of Data Science, Debre Birhan University

Submission date: \[10/14/2026]



Abstract

Finding relevant evidence in a long document can be difficult when users do not know the exact wording used by its author. This project develops a small document-retrieval prototype using a 2022 agroecology extension training manual. The system extracts PDF text and page metadata, splits text into overlapping chunks, encodes chunks with a sentence-embedding model, and retrieves passages relevant to a user question. The semantic retrieval method is compared with a TF-IDF baseline on six questions. The prototype displays retrieved passages, source pages, and retrieval scores so that users can inspect evidence instead of accepting a result without verification. Initial observations show that semantic retrieval can be useful for some conceptual questions, including soil fertility, but does not consistently return a complete answer. Neither retrieval method found a current wheat price in Ethiopia from the static manual. A simple rule-based warning was added for selected current-information questions. This is an exploratory evaluation with a small question set; it does not establish general accuracy or prove that semantic retrieval is always superior to TF-IDF.

Keywords— semantic search, information retrieval, text embeddings, TF-IDF, ChromaDB, agroecology, evidence retrieval.

I. Introduction

Training manuals and other long documents contain information that can be useful to students, extension workers, and practitioners. Finding a specific passage can nevertheless be difficult, particularly when a user's question expresses an idea differently from the source text. A user may also ask for information that the document does not contain, such as a current market price.

Traditional keyword-based retrieval ranks passages according to lexical overlap. Semantic retrieval represents text as numerical embeddings and ranks passages by similarity in embedding space. Semantic methods may retrieve related passages even when the query and source use different words, but a high-ranked passage is not necessarily a complete or correct answer.

This project implements and inspects a small semantic-search system over an agroecology training manual. It compares semantic retrieval with TF-IDF retrieval and displays the retrieved evidence and page information. The design emphasizes inspectability: a user can check the source passage instead of treating retrieval as proof of an answer.

A. Problem Statement

Users need a practical way to locate relevant passages in a document using natural-language questions. Keyword retrieval may return a heading or table-of-contents entry rather than an explanatory passage. Semantic retrieval may find conceptually related text, but it can also return general or incomplete passages. A system should also recognize that a static manual cannot establish facts that change over time.

B. Objectives

The objectives are to:

1\.	Extract text and page metadata from a PDF document.

2\.	Divide extracted text into overlapping chunks.

3\.	Create a semantic index for question-based retrieval.

4\.	Compare semantic retrieval with a TF-IDF baseline.

5\.	Inspect whether retrieved passages support the question asked.

6\.	Demonstrate a basic warning for selected current-information questions that the static document cannot answer.

II. Background and Related Concepts

A. TF-IDF Retrieval

Term Frequency–Inverse Document Frequency (TF-IDF) represents text using numerical features based on the frequency of terms and their distribution across a collection. A query can be compared with document chunks using a similarity measure such as cosine similarity. TF-IDF is relatively simple and can work well when the question and relevant passage share important words. It may be less effective when the query and passage express the same idea using different vocabulary.

B. Semantic Retrieval

Semantic retrieval uses text embeddings to represent meaning and contextual relationships. This project uses `sentence-transformers/all-MiniLM-L6-v2` to encode text chunks and questions. The embeddings are stored in ChromaDB, which supports similarity-based retrieval. The system returns ranked passages with source-page metadata and cosine distances.

A lower cosine distance indicates greater similarity under the configured metric. It is not an answer-confidence score, a probability of correctness, or a measure of factual accuracy. The TF-IDF scores and semantic cosine distances in this report use different scales and must not be compared numerically as though they were equivalent.

C. Retrieval-Augmented Generation

Retrieval-Augmented Generation (RAG) commonly combines a retriever with a generative language model. The retriever supplies relevant passages, and the language model uses those passages to produce an answer. The current project implements the ingestion and retrieval part of this workflow and provides evidence inspection plus a basic warning rule. It does not currently use a generative language model to compose complete natural-language answers. It is therefore most accurately described as a semantic-search and evidence-retrieval prototype, or an early RAG-style retrieval component, rather than a complete generative RAG system.

III. Methodology

A. Document Ingestion

The source document is a PDF titled \*Agroecology: An Extension Training Manual\* (2022). PyMuPDF is used to extract text and preserve PDF-page metadata. The recorded extraction produced text from 36 pages and approximately 10,815 words. This is an extraction count from the implemented pipeline, not a claim that every visual element was captured. Tables, figures, and scanned or complex layouts may not be represented completely by extracted text.

B. Text Chunking

The extracted text is divided into chunks of approximately 180 words with an overlap of 35 words. Overlap helps retain context when a useful passage crosses a chunk boundary. The implementation produced 85 chunks. Source and page metadata are retained so retrieved passages can be traced back to the manual.

C. Semantic Indexing and Retrieval

Each chunk is encoded with `sentence-transformers/all-MiniLM-L6-v2`. The resulting vectors are stored in a ChromaDB collection configured for cosine-distance retrieval. When a user enters a question, the question is encoded and compared with the indexed chunks. The system returns the top-ranked passages, their source pages, and distances. The retrieved text is shown for manual inspection.

D. Baseline Comparison and Evaluation

The semantic method is compared with a TF-IDF baseline using six questions covering a definition, agricultural practices, soil fertility, adoption challenges, and current market information. The top retrieved result is assessed qualitatively as relevant, partial, weak, or unsupported for the particular question.

The labels are exploratory human judgements, not formal benchmark metrics. No comprehensive gold-standard dataset was prepared, and the experiment does not calculate precision, recall, mean reciprocal rank, or normalized discounted cumulative gain. The six-question test is intended to illustrate system behavior and failure modes rather than establish general performance.

E. Rule-Based Insufficient-Evidence Warning

The demonstration includes a simple rule that recognizes selected expressions associated with current information, such as “today,” “currently,” and “current price.” For the tested question asking for today's wheat price in Ethiopia, the rule displays a warning that a static manual cannot establish the current value. This is a keyword heuristic. It does not reliably detect every unanswerable question, and it does not independently verify whether a retrieved passage supports a proposed answer.

IV. Experimental Results

A. Implementation Results

The pipeline extracted text from the manual, produced 85 overlapping chunks, created a semantic index, and retrieved passages for user questions. The demo displays retrieved text, PDF page numbers, and cosine distances. The tested wheat-price question triggered the current-information warning and showed the retrieved passages for transparency rather than presenting them as a valid price answer.

B. Retrieval Comparison

Table I. Summary of recorded top retrieval results

Question	TF-IDF top result	Semantic top result	Interpretation

What is agroecology?	PDF page 3; score 0.38	PDF page 3; distance 0.2560	Both retrieved a direct definition.

How does agroecology combine traditional and industrial agriculture?	PDF page 1; score 0.2389	PDF page 3; distance 0.2805	TF-IDF returned mainly a title; semantic retrieval was more useful but incomplete.

What are the benefits of crop diversification?	PDF page 33; score 0.2335	PDF page 15; distance 0.3271	Both returned related material, but neither fully answered the question.

How can farmers improve soil fertility?	PDF page 32; score 0.1499	PDF page 19; distance 0.3380	The semantic result more directly discussed compost, manure, crop rotations, and green manuring.

What challenges can farmers face when adopting agroecological practices?	PDF page 4; table-of-contents entry	PDF page 6; general definition and practices	Neither top result directly answered the question.

What is the current price of wheat in Ethiopia today?	PDF page 9; score 0.1212	PDF page 7; distance 0.4914	Neither result provided a current price.



\*Note:\* These entries summarize the recorded experiment outputs. TF-IDF scores and semantic cosine distances have different scales and are not directly comparable.

C. Interpretation

For “What is agroecology?”, both methods retrieved a direct definition from PDF page 3. For the soil-fertility question, the semantic result on PDF page 19 was judged more directly useful because it discussed compost, manure, crop rotations, and green manuring. These examples suggest that semantic retrieval can help locate conceptually relevant passages.

The results also show limitations. For the question about combining traditional and industrial agriculture, the semantic result was still incomplete. For the adoption-challenges question, neither method's top result directly answered the question. A passage can be related to a topic without providing the evidence required to answer a particular question.

The wheat-price question illustrates a document-boundary problem. The manual's retrieved passages did not contain a current wheat price. The prototype displayed a warning because the question requested current information. A real deployment would need a suitable current source and stronger evidence verification to answer such a question responsibly.

D. What the Results Do Not Establish

The experiment does not establish that semantic retrieval is always more accurate than TF-IDF. It includes only six questions, uses qualitative assessments, and lacks a comprehensive labelled test set. The observed results do not demonstrate the factual accuracy of a generative language model because the current prototype does not generate answers with an LLM. The warning rule's success on the wheat-price test does not establish that it will detect all unsupported questions.

V. Limitations and Future Work

The main limitations and possible improvements are:

7\.	Small evaluation set: Increase the number and variety of questions, including answerable, partially answerable, and unanswerable cases.

8\.	Human relevance judgements: Have reviewers label the relevance and sufficiency of the top-k passages before calculating retrieval metrics.

9\.	Stronger evaluation: Measure appropriate metrics such as Precision@k, Recall@k, or mean reciprocal rank once reliable relevance labels exist.

10\.	Improved chunking: Compare word-based chunking with sentence-aware or section-aware chunking.

11\.	Answerability checks: Evaluate whether the retrieved passages actually support an answer rather than relying only on keyword rules.

12\.	Generative answer stage: As future work, add a language model that answers only from retrieved evidence, cites source pages, and abstains when evidence is insufficient.

13\.	Document quality: Inspect tables, figures, and pages where PDF extraction may be incomplete.

14\.	Reproducibility: Record exact package versions, test questions, output logs, and the exact source document used.

VI. Conclusion

This project developed a small semantic-search and evidence-retrieval prototype using an agroecology training manual. The pipeline extracts text, divides it into overlapping chunks, creates embeddings, indexes them in ChromaDB, and retrieves passages for natural-language questions. An exploratory comparison with TF-IDF showed that semantic retrieval was useful for some questions but did not consistently return complete answers. The wheat-price test illustrated why a system must distinguish document evidence from time-sensitive information that the document cannot establish.

The project provides a foundation for further work on stronger evaluation, evidence sufficiency checks, and a generative RAG pipeline. Its principal lesson is that retrieving a passage is not the same as proving an answer: retrieved evidence must be inspected, and unsupported questions should be acknowledged rather than answered by guesswork.

References

\[1] \*Agroecology: An Extension Training Manual\*, 2022. Complete this reference from the PDF's title page or copyright page, including the listed author or organization, publisher, and public URL if available. Those details were not established in the project notes used to prepare this draft.

\[2] N. Reimers and I. Gurevych, “Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks,” in \*Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing\*, 2019. Available: https://aclanthology.org/D19-1410/

\[3] Scikit-learn developers, “Feature extraction of text documents,” \*scikit-learn documentation\*. Available: https://scikit-learn.org/stable/modules/feature\_extraction.html

\[4] Chroma, “Chroma documentation.” Available: https://docs.trychroma.com/

\[5] Sentence Transformers, “Sentence Transformers documentation.” Available: https://www.sbert.net/

Individual Contribution Statement

Replace this paragraph with an accurate statement of your own work. Describe the parts you personally implemented, tested, evaluated, and wrote. For example, if accurate, you may state that you worked on PDF ingestion, chunking, embedding-based retrieval, TF-IDF comparison, experiment logging, interpretation of retrieved passages, and report preparation. Do not claim tasks you did not perform.



