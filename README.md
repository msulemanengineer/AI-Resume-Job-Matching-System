# 🎯 AI Resume–Job Matching System

An educational NLP application that compares a **resume** with a **job description** using
sentence embeddings and cosine similarity, and reports where they overlap.

> ⚠️ **This is a learning prototype, not a hiring tool.** The score measures how similar two
> pieces of *text* are. It does not measure competence, it is not validated against hiring
> outcomes, and it must not be used to accept or reject candidates.

---

## Table of contents

1. [Project overview](#1-project-overview)
2. [Problem statement](#2-problem-statement)
3. [How the system works](#3-how-the-system-works)
4. [AI concepts used](#4-ai-concepts-used)
5. [Architecture](#5-architecture)
6. [Technology stack](#6-technology-stack)
7. [Folder structure](#7-folder-structure)
8. [Installation](#8-installation)
9. [Running the backend](#9-running-the-backend)
10. [Running the frontend](#10-running-the-frontend)
11. [Example input](#11-example-input)
12. [Example output](#12-example-output)
13. [Running the tests](#13-running-the-tests)
14. [Limitations](#14-limitations)
15. [Privacy considerations](#15-privacy-considerations)
16. [Future improvements](#16-future-improvements)

---

## 1. Project overview

The system takes a resume (PDF upload or pasted text) and a job description (pasted text) and
produces a matching report containing:

| Output | Meaning |
|---|---|
| **Overall match score** | A transparent weighted blend of semantic similarity and skill coverage |
| **Semantic similarity** | Pure cosine similarity between the two document embeddings |
| **Skill coverage** | Share of job-description skills that were detected in the resume |
| **Matching skills** | Skills present in both documents |
| **Potentially missing skills** | Skills the job asks for that were not detected in the resume |
| **Semantically related skills** | "Missing" skills that nevertheless resemble a passage of the resume |
| **Resume / job skills** | The full list detected in each document |

The headline number is never presented on its own — the UI always shows the formula that
produced it.

---

## 2. Problem statement

A single job posting can attract hundreds of applications, and a single candidate applies to
dozens of postings. Comparing the two by hand is slow and inconsistent.

The naive automated approach is **keyword counting**: count how many words from the job
description appear in the resume. This breaks immediately, because the same idea is written
differently in each document:

| Job description says | Resume says | Keyword match? | Actually the same? |
|---|---|---|---|
| "machine learning" | "ML modelling" | ❌ | ✅ |
| "REST API development" | "built endpoints with FastAPI" | ❌ | ✅ |
| "data analysis" | "analysed three years of transactions" | ❌ | ✅ |

This project addresses that gap by comparing **meaning** instead of exact words, while keeping
a transparent keyword baseline alongside it so the two approaches can be compared directly.

---

## 3. How the system works

```
  Resume PDF                          Job description text
      |                                       |
      v                                       |
  PyMuPDF text extraction                     |
      |                                       |
      +---------------> clean text <----------+
                            |
            +---------------+----------------+
            |                                |
            v                                v
  SEMANTIC PATH                      KEYWORD PATH
  split into ~180-word chunks        normalise text
            |                                |
  embed each chunk (MiniLM)          search 50-skill dictionary
            |                        (canonical names + aliases)
  average -> 384-dim doc vector              |
            |                        resume skills / job skills
  cosine similarity                          |
            |                        matching / missing / extra
   semantic similarity %             skill coverage %
            |                                |
            +--------> weighted blend <------+
                            |
                   OVERALL MATCH SCORE + report
```

The two paths are deliberately independent: the semantic path handles *meaning*, the keyword
path gives *explainability*. Each can be inspected — and defended — on its own.

**Score formula** (weights configurable in `.env`):

```
overall = 0.7 × semantic_similarity% + 0.3 × skill_coverage%
```

If the job description contains no dictionary skill at all, the skill term is meaningless, so
the system falls back to pure semantic similarity rather than penalising the candidate for a
blind spot in the dictionary.

---

## 4. AI concepts used

**Embeddings.** An embedding turns text into a list of numbers — here 384 of them — that
represents its meaning. The model is trained so that texts a human would call similar end up
close together in this 384-dimensional space.

**Sentence-transformers.** A library of transformer models fine-tuned specifically to produce
*sentence-level* embeddings. `all-MiniLM-L6-v2` (6 layers, ~22M parameters, ~80 MB) was trained
on roughly one billion sentence pairs with a contrastive objective: similar pairs are pulled
together, random pairs are pushed apart.

**Cosine similarity.** Measures the **angle** between two vectors, ignoring their length:

```
                 A · B
cos(A, B) = ---------------
             ||A|| × ||B||
```

Length-independence is exactly what document comparison needs: a three-page resume and a
one-paragraph job ad describing the same profile point in the same *direction*, even though one
vector is built from far more text.

**Chunking and mean pooling.** MiniLM silently truncates its input at 256 word-pieces
(~180–200 words) — far less than a resume. Each document is therefore split into overlapping
~180-word windows, every window is embedded, and the window vectors are averaged into one
document vector.

**Rule-based extraction.** Skill detection is deliberately *not* machine learning. It is a
dictionary lookup with aliases, chosen because every result can be explained in one sentence
("we found the literal string `sklearn`, which is an alias of Scikit-learn").

A full, interview-oriented explanation of every concept lives in
**[SYSTEM_GUIDE.md](SYSTEM_GUIDE.md)**, with 40 practice questions in
**[VIVA_QUESTIONS.md](VIVA_QUESTIONS.md)**.

---

## 5. Architecture

Three layers, each independently testable:

```
┌──────────────────────────────────────────────┐
│  frontend/app.py          (Streamlit, :8501) │   presentation only,
│  upload · paste · render report              │   zero AI logic
└──────────────────┬───────────────────────────┘
                   │  HTTP / JSON
┌──────────────────▼───────────────────────────┐
│  backend/main.py          (FastAPI,   :8000) │   HTTP contract,
│  routing · validation · error handling       │   Pydantic validation
└──────────────────┬───────────────────────────┘
                   │  plain Python calls
┌──────────────────▼───────────────────────────┐
│  backend/services/                           │   all the actual logic
│  pdf · embedding · matching · skill          │
└──────────────────────────────────────────────┘
```

Why split the UI from the API at all? Because the matching logic then becomes reusable by any
client (a React app, a CLI, a batch job), deployable and scalable on its own, and testable
through a stable HTTP contract rather than through the UI.

### API endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Service status, model name, whether the model is loaded |
| `GET` | `/skills` | The full skill dictionary currently in use |
| `POST` | `/match` | JSON in (`resume_text`, `job_description`) → full report |
| `POST` | `/match/upload` | Multipart PDF + job description → full report |
| `POST` | `/extract-text` | PDF → plain text (used by the UI to preview an upload) |

Interactive docs are auto-generated at <http://127.0.0.1:8000/docs>.

---

## 6. Technology stack

| Layer | Tool | Why |
|---|---|---|
| Embeddings | `sentence-transformers` + `all-MiniLM-L6-v2` | Small (~80 MB), CPU-friendly, purpose-built for sentence similarity |
| Similarity | `scikit-learn` (`cosine_similarity`) | Standard, well-tested implementation |
| Numerics | `NumPy` | Vector maths |
| PDF extraction | `PyMuPDF`, with `pypdf` as fallback | PyMuPDF is fast and accurate; a second library rescues awkward files |
| API | `FastAPI` + `Pydantic` + `Uvicorn` | Automatic validation and OpenAPI docs |
| UI | `Streamlit` | A demo-ready interface in pure Python |
| Tables | `Pandas` | Skill tables in the UI |
| Tests | `pytest` + `httpx` | Unit and API testing |

---

## 7. Folder structure

```
AI Resume-Job Matching System/
├── backend/
│   ├── main.py                     FastAPI app: routes, validation, errors
│   ├── config.py                   All tunable settings (env-overridable)
│   ├── models.py                   Pydantic request/response schemas
│   ├── text_utils.py               Cleaning, normalising, chunking
│   └── services/
│       ├── pdf_service.py          PDF -> text, with error handling
│       ├── embedding_service.py    Text -> 384-dim vectors
│       ├── matching_service.py     Cosine similarity + scoring pipeline
│       └── skill_service.py        Dictionary skill extraction + comparison
├── frontend/
│   └── app.py                      Streamlit UI (calls the API over HTTP)
├── data/
│   ├── skills.json                 Configurable 50-skill dictionary
│   ├── sample_resume.txt/.pdf      Fictional demo resume
│   └── sample_job_description.txt  Fictional demo job posting
├── scripts/
│   └── generate_sample_pdf.py      Builds the demo PDF from the .txt
├── tests/                          pytest suite (unit + API)
├── requirements.txt
├── pytest.ini
├── README.md                       This file
├── SYSTEM_GUIDE.md                 Concept-by-concept explanation
├── VIVA_QUESTIONS.md               40 interview Q&A + timed explanations
├── .env.example
└── .gitignore
```

---

## 8. Installation

### Prerequisite: Python 3.10 – 3.12

PyTorch does not publish wheels for Python 3.13 or 3.14 yet, so `pip install` will fail on those
versions. Check what you have:

```bash
python --version
py -0p            # Windows: lists every installed Python
```

If your default `python` is 3.13+, point the `venv` step at a 3.12 interpreter explicitly rather
than at `python`.

### Setup

```bash
# 1. Clone and enter the project
git clone <your-repo-url>
cd "AI Resume-Job Matching System"

# 2. Create a virtual environment with a supported Python
python -m venv .venv
# ...or, if your default python is 3.13+, name the interpreter explicitly:
#   Windows:      py -3.12 -m venv .venv
#   macOS/Linux:  python3.12 -m venv .venv

# 3. Activate it
.venv\Scripts\activate       # Windows (PowerShell / cmd)
source .venv/bin/activate     # macOS / Linux

# 4. Install dependencies (~2 GB, mostly PyTorch — takes a few minutes)
pip install -r requirements.txt

# 5. Optional: regenerate the demo PDF
python scripts/generate_sample_pdf.py
```

The embedding model (~80 MB) downloads automatically on the **first match request** and is then
cached in `~/.cache/huggingface/`. That first request takes ~10–30 seconds; later ones are
near-instant.

Optional configuration: copy `.env.example` to `.env` and edit it. Every setting has a working
default, so this step can be skipped entirely.

### Verified environment

This project was built and tested on Windows 11 with Python 3.12.7 and:

| Package | Version |
|---|---|
| sentence-transformers | 5.7.0 |
| torch | 2.14.0 (CPU) |
| scikit-learn | 1.9.1 |
| FastAPI | 0.141.1 |
| Streamlit | 1.63.0 |
| PyMuPDF | 1.28.2 |
| NumPy | 2.5.3 |

`requirements.txt` uses version ranges rather than exact pins, so newer compatible releases will
also install. Two compatibility shims exist for this reason: the embedding dimension is read via
whichever method name the installed sentence-transformers version provides, and the Streamlit UI
picks `width="stretch"` or `use_container_width=True` depending on its version.

---

## 9. Running the backend

From the project root, with the virtual environment activated:

```bash
uvicorn backend.main:app --reload --port 8000
```

Verify it:

```bash
curl http://127.0.0.1:8000/health
```

Expected response (`model_loaded` is `false` until the first match — the model loads lazily):

```json
{"status":"ok","embedding_model":"all-MiniLM-L6-v2","model_loaded":false,
 "embedding_dimension":null,"skills_in_dictionary":50}
```

Interactive API docs: <http://127.0.0.1:8000/docs>

---

## 10. Running the frontend

In a **second terminal**, with the virtual environment activated:

```bash
streamlit run frontend/app.py
```

Then open <http://localhost:8501>. The sidebar shows whether the backend is reachable.

If port 8501 is already in use, pick another and tell Streamlit where the API is:

```bash
streamlit run frontend/app.py --server.port 8502
```

> Both processes must be running: Streamlit is only a UI and calls the API for every result.
> If the backend is down, the sidebar says so and shows the command to start it.

---

## 11. Example input

**Resume** (excerpt from `data/sample_resume.txt`):

```
ALEX MORGAN — Final-year Computer Science Student

TECHNICAL SKILLS
Languages: Python, SQL, JavaScript
ML / AI: Machine Learning, Deep Learning, NLP, scikit-learn, PyTorch
Data: Pandas, NumPy, Matplotlib
Backend: FastAPI, REST APIs
Tools: Git, GitHub, Docker, Linux, pytest

PROJECTS
Semantic Document Search Engine
- Built a search tool that embeds documents with sentence-transformers and
  retrieves the closest matches using cosine similarity over a FAISS index.
```

**Job description** (excerpt from `data/sample_job_description.txt`):

```
AI/Machine Learning Intern

Required skills
- Strong Python programming skills.
- Practical experience with Machine Learning and scikit-learn.
- Familiarity with Deep Learning frameworks such as PyTorch or TensorFlow.
- Understanding of NLP fundamentals: tokenization, embeddings, transformers.
- Experience building REST APIs (FastAPI or Flask).
- Comfortable with SQL for data extraction.
```

**Running exactly this pair through the API** (the output in the next section is the real
response from this command):

```bash
curl -X POST http://127.0.0.1:8000/match/upload   -F "resume_file=@data/sample_resume.pdf"   -F "job_description=$(cat data/sample_job_description.txt)"   -F "use_semantic_skills=true"
```

---

## 12. Example output

The response below is the **actual output** of the command above — `data/sample_resume.pdf`
matched against `data/sample_job_description.txt`. Lists are abbreviated for readability;
nothing else has been edited.

```json
{
  "overall_match_score": 69.85,
  "semantic_similarity_score": 69.17,
  "match_band": "Good alignment",

  "score_breakdown": {
    "semantic_similarity_percent": 69.17,
    "skill_coverage_percent": 71.43,
    "semantic_weight": 0.7,
    "skill_weight": 0.3,
    "formula": "0.7 x semantic(69.17%) + 0.3 x skill_coverage(71.43%)"
  },

  "matching_skills": ["Python", "SQL", "Machine Learning", "Deep Learning", "NLP",
                      "Scikit-learn", "PyTorch", "Pandas", "NumPy", "FastAPI",
                      "Docker", "Git", "GitHub", "REST API", "Data Visualization"],
  "missing_skills": ["TensorFlow", "Flask", "AWS", "CI/CD", "LLM", "RAG"],
  "extra_resume_skills": ["JavaScript", "Matplotlib", "Linux", "Vector Database",
                          "Hugging Face", "Data Analysis", "Statistics", "Testing", "Agile"],

  "semantically_related_skills": [
    {
      "skill": "TensorFlow",
      "similarity": 0.5683,
      "evidence": "- Deep Learning with PyTorch (online)"
    }
  ],

  "resume_char_count": 1967,
  "job_char_count": 1461,
  "disclaimer": "This score measures TEXT SIMILARITY between a resume and a job description. ..."
}
```

**How to read it:** the resume covers 15 of the 21 skills the job mentions. Of the six it
missed, TensorFlow was flagged as semantically related — the resume lists a *PyTorch* deep
learning certification, which is the same field but not the same tool. That is exactly the kind
of judgement call the system hands back to a human instead of resolving on its own.

### The score does discriminate

The same resume run against a deliberately unrelated posting:

| Job description | Overall | Semantic | Band |
|---|---|---|---|
| AI/ML internship (`data/sample_job_description.txt`) | **69.85%** | 69.17% | Good alignment |
| Head Pastry Chef | **13.46%** | 13.46% | Low alignment |

> ⚠️ **These are illustrations of output, not performance metrics.** They come from two
> hand-written sample files with no ground-truth labels behind them, so they say nothing about
> accuracy. See [How to evaluate the system properly](SYSTEM_GUIDE.md#27-how-to-evaluate-the-system-properly)
> for what a real evaluation would require.

---

## 13. Running the tests

```bash
pytest                    # everything (loads the model — slower)
pytest -m "not slow"      # fast tests only: PDF, skills, text utils, validation
pytest tests/test_matching_service.py -v
```

The suite covers PDF extraction (including corrupt, non-PDF and scanned files), skill
extraction (including the `Java` vs `JavaScript` false-positive trap), embedding properties,
cosine-similarity maths, the end-to-end pipeline, and every API endpoint.

---

## 14. Limitations

These are real and should be stated openly in any demo:

1. **No OCR.** Scanned / image-only PDFs have no text layer and cannot be read. The system
   detects this and says so rather than matching an empty document.
2. **The skill dictionary is finite.** Only the ~50 skills in `data/skills.json` can be
   detected. Anything else is invisible, so "missing skills" can be wrong in both directions.
3. **No negation handling.** "No experience with Java" still registers as a Java mention.
4. **No proficiency or recency.** A one-weekend tutorial and three years of production work
   both count as one mention.
5. **English only.** The model is trained on English text.
6. **Chunk averaging dilutes long documents.** A five-page resume with one highly relevant page
   is averaged down by the other four.
7. **The score is uncalibrated.** Cosine values from this model typically cluster in the
   0.3–0.8 range, so 50% does not mean "half as good". Scores are useful for *comparing*
   candidates against the *same* posting, not as an absolute grade.
8. **The weights are a design choice**, not learned parameters.
9. **No evaluation against ground truth.** No labelled dataset of "good matches" exists here,
   so no accuracy figure is claimed.
10. **Resume formatting affects results.** Multi-column layouts can be extracted in the wrong
    reading order, which changes the text the model sees.

---

## 15. Privacy considerations

Resumes are personal data (name, email, phone, employment history), so the system is built to
hold them for as little time as possible:

- **No permanent storage.** Uploaded PDFs are read into memory, converted to text, used for one
  request, and dropped when the function returns. Nothing is written to disk.
- **No content logging.** Logs record only file sizes and character counts — never resume text.
  Check `backend/main.py`: every log line passes lengths, not content.
- **No third-party calls.** The embedding model runs locally. Resume text never leaves the
  machine — there is no external API in the request path.
- **No personal data in the repository.** `data/sample_resume.txt` is fictional. `.gitignore`
  additionally blocks `*_resume.pdf`, `uploads/` and `*.docx` so a real resume cannot be
  committed by accident.
- **No database, no accounts, no cookies.** There is nothing to breach.
- **If deployed publicly**, you would additionally need HTTPS, a rate limit, an explicit consent
  notice, a retention policy, and (in the EU) a GDPR lawful basis for processing.

---

## 16. Future improvements

| Improvement | What it would fix |
|---|---|
| **Richer skill ontology** (ESCO, O*NET) | Thousands of skills with real synonym/parent relations instead of 50 hand-written entries |
| **Named Entity Recognition** (spaCy, fine-tuned BERT) | Detect skills, employers, degrees and dates not present in any dictionary |
| **Section-aware parsing** | Weight the "Experience" section above "Interests"; avoids the dilution problem |
| **Experience-level extraction** | Distinguish "3 years of Python" from "used Python once" |
| **Education extraction** | Check degree requirements explicitly |
| **Better semantic skill matching** | Compare skill *embeddings* against a skill graph rather than raw sentences |
| **Cross-encoder re-ranking** | A cross-encoder reads both texts together and is far more accurate than comparing two independent vectors |
| **Multiple resumes + ranking** | Upload N resumes, rank them against one posting |
| **Vector database** (FAISS, Qdrant) | Pre-compute resume vectors so matching scales to thousands of candidates |
| **LLM-assisted explanations** | "Your FastAPI project covers the REST API requirement" in natural language |
| **Evaluation dataset** | Human-labelled resume/job pairs, so precision, recall and rank correlation can finally be measured |
| **Bias auditing** | Test whether scores shift when names, gender markers or universities are swapped |
| **Authentication + rate limiting** | Required before any real deployment |
| **Cloud deployment** (Docker + a managed host) | Shareable demo link |
| **OCR fallback** (Tesseract) | Support scanned resumes |
| **Multilingual model** (`paraphrase-multilingual-MiniLM-L12-v2`) | Handle non-English resumes |

---

## License & intended use

Educational portfolio project. Built to demonstrate an NLP pipeline end to end —
**not** to be used for real hiring decisions.
