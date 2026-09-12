# 📘 SYSTEM GUIDE — AI Resume–Job Matching System

**Purpose of this file:** explain every part of this project in plain English, so you can
answer any question about it in an interview or viva.

Read it top to bottom once. Then use the section list to revise.

> Practice questions live in **[VIVA_QUESTIONS.md](VIVA_QUESTIONS.md)** (40 Q&A plus a
> 60-second and a 2-minute spoken explanation).

---

## Contents

| # | Section | # | Section |
|---|---|---|---|
| 1 | [What is this project?](#1-what-is-this-project) | 15 | [Skill extraction](#15-skill-extraction) |
| 2 | [What problem does it solve?](#2-what-problem-does-it-solve) | 16 | [Keyword vs semantic matching](#16-exact-keyword-matching-vs-semantic-matching) |
| 3 | [Why is AI useful here?](#3-why-is-ai-useful-here) | 17 | [FastAPI architecture](#17-fastapi-architecture) |
| 4 | [What is NLP?](#4-what-is-nlp) | 18 | [Streamlit architecture](#18-streamlit-architecture) |
| 5 | [What is semantic similarity?](#5-what-is-semantic-similarity) | 19 | [Complete data flow](#19-complete-data-flow) |
| 6 | [What are embeddings?](#6-what-are-embeddings) | 20 | [Example request](#20-example-request) |
| 7 | [How sentence-transformers works](#7-how-sentence-transformers-works-at-a-high-level) | 21 | [Example response](#21-example-response) |
| 8 | [What is a vector?](#8-what-is-a-vector) | 22 | [Limitations](#22-limitations) |
| 9 | [What is cosine similarity?](#9-what-is-cosine-similarity) | 23 | [Bias & fairness](#23-biasfairness-considerations) |
| 10 | [Why cosine similarity?](#10-why-cosine-similarity) | 24 | [Privacy](#24-privacy-considerations) |
| 11 | [What does the score mean?](#11-what-does-the-similarity-score-mean) | 25 | [How to improve the system](#25-how-to-improve-the-system) |
| 12 | [Why it is NOT a hiring probability](#12-why-the-score-is-not-a-hiring-probability) | 26 | [How to scale it](#26-how-to-scale-it) |
| 13 | [PDF text extraction](#13-pdf-text-extraction) | 27 | [How to evaluate it properly](#27-how-to-evaluate-the-system-properly) |
| 14 | [Text preprocessing](#14-text-preprocessing) | | |

---

## 1. What is this project?

A web application that compares **one resume** with **one job description** and reports how
closely they align.

You give it two things:
1. A resume — uploaded as a PDF, or pasted as text.
2. A job description — pasted as text.

It gives you back:
- an **overall match score** out of 100,
- the **semantic similarity** between the two documents,
- which **skills appear in both**,
- which skills the job asks for that it **could not find** in the resume,
- and, optionally, missing skills that nevertheless **look related** to something in the resume.

It is an **educational prototype**. The whole point is to demonstrate a complete NLP pipeline —
text in, vectors in the middle, a meaningful number out — not to make hiring decisions.

**One-line version:** *"It turns a resume and a job description into vectors that represent
their meaning, then measures the angle between those vectors."*

---

## 2. What problem does it solve?

Two sides of the same problem:

- **A recruiter** posts one job and receives 400 applications. Reading them all carefully is not
  realistic, and reading them quickly is inconsistent — the 300th resume gets less attention
  than the 3rd.
- **A candidate** applies to 40 jobs and has no idea which ones they are actually a good fit
  for, or what their resume is missing.

The obvious automated solution is **keyword counting**: count how many words from the job ad
appear in the resume. It fails constantly, because people describe the same thing differently:

| Job description says | Resume says | Keyword match | Same meaning? |
|---|---|---|---|
| "machine learning" | "ML modelling" | ❌ no | ✅ yes |
| "REST API development" | "built endpoints with FastAPI" | ❌ no | ✅ yes |
| "data analysis" | "analysed three years of sales data" | ❌ no | ✅ yes |
| "team player" | "team player" | ✅ yes | ⚠️ means nothing |

Keyword matching produces **false negatives** (good candidates rejected for using synonyms) and
**false positives** (empty buzzwords scoring highly). It also rewards keyword-stuffing.

This project compares **meaning** instead — while *also* keeping a transparent keyword baseline,
so you can show both and explain the difference.

---

## 3. Why is AI useful here?

Because the mapping from "words used" to "skills possessed" is far too messy to write rules for.

You could try to hand-write rules — "ML" = "machine learning", "built endpoints" = "API
development" — but you would need thousands of them, they would never be complete, and every new
job domain would need a new rule set.

A pretrained language model has already learned these relationships from about a billion
sentence pairs. It knows that "machine learning" and "ML modelling" occur in similar contexts, so
it places them near each other in vector space — **without anyone writing a rule for that
specific pair.**

That is the core value: *generalisation to word combinations nobody anticipated.*

**Important honesty point for an interview:** AI is useful for the *similarity* half of this
system. For the *skills* half, a plain dictionary lookup is actually **better**, because a
recruiter can audit it ("we found the literal word `Docker`") whereas a neural score of 0.63 is
much harder to justify. Knowing when *not* to use ML is a real engineering skill — say so.

---

## 4. What is NLP?

**Natural Language Processing** is the field of AI that lets computers work with human language —
text and speech.

Language is hard for computers because it is:
- **ambiguous** — "Apple" the fruit or the company?
- **context-dependent** — "That's sick" is praise or a complaint depending on who says it
- **infinitely variable** — countless valid ways to express one idea
- **unstructured** — no rows, no columns, no schema

Common NLP tasks include classification (spam or not), named entity recognition (find people and
places), translation, summarisation, question answering, and — **the one this project uses** —
**semantic textual similarity**: how close in meaning are two pieces of text?

**Where NLP appears in this project:**

| Step | NLP technique |
|---|---|
| Extracting text from a PDF | Text extraction (a pre-NLP step, strictly speaking) |
| Cleaning and normalising text | Text preprocessing |
| Splitting into chunks | Document segmentation |
| Turning text into vectors | Embeddings / representation learning |
| Comparing the vectors | Semantic textual similarity |
| Finding skills | Rule-based information extraction |

---

## 5. What is semantic similarity?

**Semantic** = relating to meaning. **Semantic similarity** = how close in *meaning* two pieces
of text are, regardless of the words used.

| Sentence A | Sentence B | Word overlap | Semantic similarity |
|---|---|---|---|
| "I build ML models in Python" | "I develop machine learning systems using Python" | low | **very high** |
| "I love this job" | "I hate this job" | **very high** | **low** (opposite meaning!) |
| "The bank raised rates" | "The river bank eroded" | high ("bank") | low |

Notice rows 2 and 3: word overlap and meaning can point in completely opposite directions. That
is precisely why keyword matching is unreliable and why semantic similarity is worth the extra
machinery.

Measuring it requires two steps:
1. Convert each text into a numeric representation of its meaning → **an embedding**.
2. Measure how close those two representations are → **cosine similarity**.

---

## 6. What are embeddings?

**An embedding is a list of numbers that represents the meaning of a piece of text.**

In this project, every piece of text becomes a list of **384 numbers**:

```
"I build machine learning models in Python"
        ↓  (the embedding model)
[0.042, -0.118, 0.267, 0.009, ..., -0.053]     ← 384 numbers
```

That list is called a **vector**, and the 384-dimensional space it lives in is the
**embedding space** or **vector space**.

### Why do similar texts get similar vectors?

Because the model was **trained** to make that happen. During training it saw millions of
sentence pairs labelled as similar or unrelated, and its weights were adjusted so that:

- similar pairs → vectors **pulled together**
- unrelated pairs → vectors **pushed apart**

This is called **contrastive learning**. After a billion pairs, the model has learned a general
rule: *text that means similar things goes in similar directions.* It then applies that rule to
sentences it has never seen — including your resume.

### The intuitive picture

Think of a map of the world. Cities close on the map are close in reality. An embedding space is
the same idea, except:
- it has 384 dimensions instead of 2, and
- "distance" means *difference in meaning* rather than kilometres.

In that space, "Python developer" sits near "software engineer who writes Python", and both sit
far away from "pastry chef".

### Why 384 dimensions?

Meaning has many independent aspects — topic, tone, formality, tense, specificity, domain. Two
or three numbers cannot capture all of that; 384 gives the model enough room while staying small
enough to compute quickly. It is a property of the chosen model (`all-MiniLM-L6-v2`), not a
setting we picked.

### Word embeddings vs sentence embeddings

- **Word embeddings** (Word2Vec, GloVe, 2013–2014): one fixed vector per *word*. Famous property:
  `king − man + woman ≈ queen`. Problem: "bank" gets one vector for both meanings, and there is
  no good way to combine word vectors into a sentence.
- **Sentence embeddings** (what we use): one vector for a whole *sentence or passage*, produced
  by a transformer that reads all the words **in context**. "Bank" gets a different vector in
  "river bank" than in "investment bank".

---

## 7. How sentence-transformers works at a high level

`sentence-transformers` is a Python library that wraps transformer models fine-tuned specifically
to produce good *sentence-level* embeddings.

### The model used here: `all-MiniLM-L6-v2`

| Property | Value |
|---|---|
| Architecture | MiniLM — a distilled (compressed) BERT |
| Layers | 6 transformer layers |
| Parameters | ~22 million |
| Output dimension | **384** |
| Max input | **256 word-pieces** (~180–200 words) ← *important, see §14* |
| Size on disk | ~80 MB |
| Training data | ~1 billion sentence pairs |

### What happens inside, step by step

```
"I build ML models in Python"
        ↓ 1. TOKENIZATION
["i", "build", "ml", "models", "in", "python"]   → token IDs
        ↓ 2. EMBEDDING LOOKUP + POSITION
each token → an initial vector (+ its position in the sentence)
        ↓ 3. SIX TRANSFORMER LAYERS (self-attention)
each token's vector is updated by "looking at" every other token,
so "python" becomes context-aware (the language, not the snake)
        ↓ 4. MEAN POOLING
average all the token vectors → ONE 384-dim sentence vector
        ↓ 5. NORMALISATION
scale it to length 1
        ↓
[0.042, -0.118, ..., -0.053]
```

**Self-attention** (step 3) is the key transformer idea: every word's representation is built
by weighing every other word in the sentence. That is how the model resolves ambiguity — "Python"
surrounded by "models" and "build" lands in the programming region of the space, not the
zoology region.

**Mean pooling** (step 4) is how a variable-length sentence becomes a fixed-size vector. It is
the standard choice for this model family.

**Normalisation** (step 5) makes every vector length 1, which means the dot product of two
vectors *is* their cosine similarity — a useful computational shortcut used throughout this
codebase.

### Why "distilled"?

Full BERT has 12 layers and 110M parameters. **Knowledge distillation** trains a small "student"
model to imitate a large "teacher" model. MiniLM keeps most of the quality at roughly a fifth of
the size — which is why this project runs on a laptop CPU in milliseconds.

### Why this model rather than a bigger one?

An explicit, defensible trade-off:

| | `all-MiniLM-L6-v2` | A large model (e.g. `all-mpnet-base-v2`) |
|---|---|---|
| Size | 80 MB | 420 MB |
| Speed on CPU | milliseconds | noticeably slower |
| Quality | good | better |
| Suits a laptop demo | ✅ | ⚠️ |

For demonstrating the *concept*, MiniLM is the right call. If accuracy mattered more than speed,
swapping the model is a one-line change in `backend/config.py` — which is exactly why that value
is a configurable setting rather than a hard-coded string.

---

## 8. What is a vector?

A **vector** is just an ordered list of numbers, e.g. `[3, 4]` or `[0.04, -0.11, ..., 0.26]`.

Geometrically it is an **arrow from the origin to a point**:
- `[3, 4]` is an arrow in 2D pointing right and up.
- Its **length** (magnitude) is `√(3² + 4²) = 5`.
- Its **direction** is where it points.

For embeddings, **direction carries the meaning**. Two vectors pointing the same way represent
similar meanings — regardless of how long they are.

We cannot picture 384 dimensions, and we do not need to: the maths (dot products, lengths,
angles) works identically in any number of dimensions. Reason in 2D, compute in 384D.

**Vocabulary you may be asked about:**

| Term | Meaning |
|---|---|
| Dimension | How many numbers are in the vector (384 here) |
| Magnitude / norm | The vector's length, `√(Σxᵢ²)` — written `‖A‖` |
| Unit vector | A vector of length exactly 1 |
| Normalisation | Dividing a vector by its length to make it a unit vector |
| Dot product | `A·B = Σ(aᵢ × bᵢ)` — a single number |
| Vector space | The space all these vectors live in |

---

## 9. What is cosine similarity?

Cosine similarity measures the **angle** between two vectors.

```
                    A · B              Σ (aᵢ × bᵢ)
cosine(A, B) = ─────────────── = ──────────────────────────
                 ‖A‖ × ‖B‖        √(Σaᵢ²) × √(Σbᵢ²)
```

- **Numerator** — the dot product: large when the two vectors agree component by component.
- **Denominator** — the two lengths: divides the length out, leaving only direction.

### What the values mean

| Cosine | Angle | Interpretation |
|---|---|---|
| **1.0** | 0° | Same direction — identical meaning |
| **0.8** | ~37° | Very similar |
| **0.5** | 60° | Somewhat related |
| **0.0** | 90° | Unrelated (orthogonal) |
| **−1.0** | 180° | Opposite direction |

The theoretical range is `[−1, 1]`, but in practice this model almost always outputs values in
`[0, 1]`, because it is trained to produce vectors in a comparatively narrow cone of the space.

### A worked example (2D, so you can check it by hand)

```
A = [3, 4]        B = [4, 3]

A · B  = 3×4 + 4×3 = 24
‖A‖    = √(9 + 16) = 5
‖B‖    = √(16 + 9) = 5

cosine = 24 / (5 × 5) = 0.96      → very similar direction
```

### Where it lives in this code

[`backend/services/matching_service.py`](backend/services/matching_service.py) — the
`cosine_similarity()` function. It calls scikit-learn's implementation rather than hand-rolling
the formula, because scikit-learn handles the numerical edge cases; the maths is exactly the
formula above. Zero vectors are defined as similarity `0.0` to avoid a division-by-zero `NaN`.

---

## 10. Why cosine similarity?

### Reason 1 — it ignores document length (the big one)

A resume is 600 words; a job description is 250. Both describe the same profile. If we used a
metric sensitive to magnitude, the longer document would appear "far away" simply for being
longer.

Cosine divides magnitude out. Only direction — meaning — remains.

**A concrete demonstration:**

```
A = [1, 1]          ← short document
B = [10, 10]        ← the same content, ten times longer

cosine(A, B)    = 1.0    ✅ correctly identifies them as the same
euclidean(A, B) = 12.7   ❌ calls them very different
```

This exact property is asserted in the test suite —
`test_cosine_ignores_magnitude` in `tests/test_matching_service.py`.

### Reason 2 — a bounded, interpretable range

Cosine always falls in `[−1, 1]`. Euclidean distance is unbounded (`0` to `∞`), so there is no
natural way to turn it into a percentage without arbitrary rescaling.

### Reason 3 — it matches how the model was trained

Sentence-transformer models are trained with a cosine-based objective. Using cosine at inference
time means measuring similarity the same way the model learned it. Using something else would be
applying a different ruler than the one the model was calibrated with.

### Reason 4 — it is cheap

For unit-length vectors, cosine similarity reduces to a single dot product — one pass of
multiply-and-add. This matters at scale: comparing one job against 100,000 resumes is one
matrix multiplication.

### The honest counterpoint

When vectors are **already normalised** (as ours are), cosine similarity and Euclidean distance
are monotonically related — `euclidean² = 2 − 2·cosine` — so they rank pairs *identically*. For
this specific pipeline the choice is about interpretability and convention, not about producing a
different ordering. **Saying this in an interview shows real understanding rather than recited
dogma.** The length-independence argument is genuinely decisive when vectors are *not*
normalised, which is the general case.

---

## 11. What does the similarity score mean?

### The two numbers this system reports

**1. Semantic similarity (%)** — the pure cosine similarity between the two document vectors,
multiplied by 100. Negative values (rare) are clamped to 0 so the UI never shows a negative
percentage. This is a **linear rescale for display only** — it adds no statistical meaning.

**2. Overall match score (%)** — a transparent weighted blend:

```
overall = 0.7 × semantic_similarity%  +  0.3 × skill_coverage%
```

where `skill_coverage% = (job skills found in the resume) / (job skills total) × 100`.

### Why blend two numbers at all?

Each alone is misleading:

- **Semantic similarity alone** rewards writing *about* the right topic. A resume full of
  AI buzzwords with no actual projects can still land near the job description.
- **Skill coverage alone** is pure keyword matching, with all the synonym problems of §2.

Blending them means a strong score requires the resume to be *both* topically close *and*
concretely overlapping in named skills. The `0.7 / 0.3` split favours semantics (that is the
point of the project) while keeping skills as a meaningful check.

**Be clear in an interview: those weights are a design choice I made for readability. They were
not learned from data, because no labelled data exists here.** They live in `backend/config.py`
and can be changed in one place.

### The fallback case

If the job description mentions **no** dictionary skill at all, `skill_coverage` would be `0` —
which would unfairly drag the score down because of a gap in *our dictionary*, not a gap in the
candidate. In that case the system falls back to pure semantic similarity and says so in the
`formula` field.

### How to read a score

| Score | Band | Reasonable reading |
|---|---|---|
| 75–100 | Strong alignment | Resume and posting are clearly about the same kind of work |
| 60–75 | Good alignment | Substantially related, some gaps |
| 40–60 | Moderate alignment | Some overlap, notable differences |
| 0–40 | Low alignment | Largely different fields |

**These bands are a readability aid chosen by hand.** They are not derived from labelled hiring
data, and a different embedding model would need different bands.

### The most important caveat

Scores from this system are **relative, not absolute**. Comparing five candidates against the
*same* posting is meaningful — the same model and the same reference text are used throughout.
Comparing "62% on job A" with "58% on job B" is **not** meaningful, because the two postings have
different lengths, vocabularies and writing styles.

---

## 12. Why the score is NOT a hiring probability

This is the point most worth being crisp about. Expect to be pushed on it.

**A probability requires a prediction target and labelled outcomes.** To say "this candidate has
a 72% chance of being hired", you would need:
1. thousands of historical (resume, job, hired yes/no) records,
2. a model trained to predict that outcome,
3. a held-out test set showing the model's predicted probabilities are calibrated.

**This system has none of those.** It was never trained on hiring outcomes. It has never seen a
single hiring decision. It measures **text similarity** and nothing else.

### What the number actually measures

> "The resume and the job description use language that points in a similar direction in the
> embedding model's vector space, and they share a certain proportion of dictionary skills."

That is a statement about **two documents**, not about a person.

### Concrete failure cases that prove the point

| Situation | The score | Reality |
|---|---|---|
| Candidate copy-pastes the job ad into their resume | ~100% | Terrible candidate |
| Brilliant engineer with a terse, understated resume | ~45% | Excellent candidate |
| Resume written in the posting's exact jargon, no real experience | high | Weak candidate |
| Career-changer with strong transferable skills | low | Possibly ideal |

Every one of these is a failure of the *metric*, not a bug in the code. A high score means the
documents are similar. It does **not** mean the person can do the job.

### Why this matters beyond the technical point

Hiring tools affect people's livelihoods. A system that presents an unvalidated similarity score
as a hiring probability is not merely inaccurate — it launders a guess as a measurement and
invites people to defer to it. Amazon famously scrapped an internal resume-screening model in
2018 after finding it penalised resumes containing the word "women's".

**This is why the disclaimer is not decoration.** It is in the API response body
(`disclaimer` field), on the UI, and in the README. It is part of the deliverable.

---

## 13. PDF text extraction

### The problem

A PDF describes *how a page should look*, not what it says. It is a set of drawing instructions:
"place the glyph 'P' at coordinates (72, 100) in 11pt Helvetica". There are no paragraphs, no
reading order, no structure — extraction means reconstructing text from positioning data.

### The two kinds of PDF

| Type | How it was made | Text layer? | Can we read it? |
|---|---|---|---|
| **Digital** | Exported from Word, LaTeX, Google Docs | ✅ yes | ✅ yes |
| **Scanned** | Photographed or scanned on paper | ❌ no — it is an image | ❌ **no** |

A scanned resume is a *picture* of text. Extracting from it returns an empty string. Reading it
would require **OCR** (Optical Character Recognition, e.g. Tesseract), which this project
deliberately does not include.

**Crucially, the system detects this case and says so** rather than silently matching an empty
document — which would produce a meaningless score with no warning. Failing loudly beats failing
silently.

### The implementation

[`backend/services/pdf_service.py`](backend/services/pdf_service.py):

1. **Reject obviously bad input** — empty uploads, files over 5 MB, files whose first 1024 bytes
   do not contain the `%PDF` magic bytes.
2. **Try PyMuPDF** (`fitz`) — fast, accurate, good at reading order.
3. **Fall back to pypdf** if PyMuPDF fails — different libraries cope with different malformed
   files, so a second attempt genuinely rescues some documents.
4. **If both return no text**, raise a clear error naming the likely cause (scanned PDF) and the
   missing capability (no OCR).
5. **Clean the result** (see §14).
6. **Warn if suspiciously short** (under 50 characters) — probably a mostly-image PDF.

### Error cases, all tested

| Input | Response |
|---|---|
| Empty file (0 bytes) | "The uploaded file is empty" |
| A `.txt` renamed to `.pdf` | "This does not look like a PDF file" |
| Corrupt PDF | "The PDF could not be parsed — it may be corrupted" |
| Password-protected PDF | "This PDF is password protected" |
| Scanned / image-only PDF | "…most likely a scanned/image-only document… no OCR" |
| Over 5 MB | "File is too large… Limit is 5 MB" |

### A known weakness worth volunteering

**Multi-column layouts.** Many resume templates use two columns (skills on the left, experience
on the right). Extractors read roughly left-to-right, top-to-bottom, so text from the two columns
can interleave into scrambled output. The embedding model still sees roughly the right *words*,
so similarity survives reasonably well — but the extracted text may read as nonsense. A
layout-aware parser (e.g. `layoutparser`, or PyMuPDF's block-level API with column detection)
would be the fix.

---

## 14. Text preprocessing

### The key insight — and a great interview answer

Classic NLP preprocessing (lower-casing, stop-word removal, stemming, punctuation stripping) was
designed for **bag-of-words** models, which treat text as an unordered pile of tokens. Those
models needed aggressive normalisation to reduce vocabulary size.

**Transformers are the opposite.** They are trained on *normal human sentences*, complete with
capitalisation, punctuation and stop-words, because those carry real information:

- "**No** experience with Java" — remove the stop-word "no" and the meaning inverts.
- "Apple" vs "apple" — capitalisation distinguishes the company from the fruit.
- "I led the team" vs "I was led by the team" — word order is everything.

**So: aggressive preprocessing would actively damage a transformer's input.** This project only
fixes the mess that *PDF extraction* introduces — nothing more.

### The two separate pipelines

This is a design detail worth pointing out, because it shows deliberate thinking:

**Pipeline A — `clean_text()`, for the embedding model** (light touch):
1. Normalise line endings (`\r\n` → `\n`).
2. Re-join hyphenated line breaks: `"machine lear-\nning"` → `"machine learning"`.
3. Replace bullet glyphs (`•`, `▪`, `·`) with spaces — they are layout, not language.
4. Collapse runs of spaces and blank lines.

Capitalisation, punctuation and stop-words are all **preserved**.

**Pipeline B — `normalize_for_matching()`, for keyword lookup** (aggressive):
1. Lower-case everything.
2. Strip punctuation — *except* `+ # . / -`, so that `C++`, `C#`, `Node.js` and `CI/CD` survive.
3. Collapse whitespace.

Keyword search *wants* aggressive normalisation; the transformer does not. Same input, two
different preparations, for two different consumers.

### Chunking — and why it is not optional

**`all-MiniLM-L6-v2` silently truncates its input at 256 word-pieces** (~180–200 words). Feed it a
600-word resume and it reads roughly the first third and **discards the rest without any
warning**. Your entire projects section could vanish.

The solution, in `chunk_words()`:

```
600-word resume
      ↓  split into overlapping 180-word windows (30-word overlap)
   [chunk 1] [chunk 2] [chunk 3] [chunk 4]
      ↓  embed each one
   [v1]      [v2]      [v3]      [v4]          each 384-dim
      ↓  average them
   document vector (384-dim)
      ↓  re-normalise to unit length
```

**Why the 30-word overlap?** A sentence sitting on a chunk boundary would otherwise be cut in
half in every chunk, and neither half would carry its full meaning. Overlapping windows ensure
every sentence appears intact in at least one chunk.

**Why averaging?** It is the simple, standard choice: the document vector lands in the "centre of
mass" of its parts.

**The honest weakness — dilution.** A five-page resume with one highly relevant page gets
averaged down by the other four. A max-pooling or attention-weighted scheme would handle this
better; averaging was chosen for explainability. `test_chunked_embedding_differs_from_naive_truncation`
in the test suite proves the chunking actually changes the result — it is not decoration.

---

## 15. Skill extraction

### The method: dictionary lookup with aliases

`data/skills.json` holds ~50 technical skills. Each has a **canonical name** and a list of
**aliases**:

```json
{ "name": "Scikit-learn", "aliases": ["scikit-learn", "scikit learn", "sklearn"] }
{ "name": "NLP",          "aliases": ["nlp", "natural language processing", "text mining"] }
```

For every skill, the normalised text is searched for any of its aliases. A hit on *any* alias
reports the **canonical name**, so `sklearn` and `scikit-learn` both appear as "Scikit-learn".

### The word-boundary problem (a great detail to mention)

The naive implementation — `if "java" in text` — matches "**Java**Script" and reports Java for a
candidate who has never written a line of it. A classic false positive.

The usual fix is the regex `\b` word boundary, but `\b` breaks on symbol-heavy skill names:
`C++`, `C#`, `.NET` and `CI/CD` end in characters that are not word characters, so `\b` does not
behave as expected.

The solution used here is explicit look-around:

```python
re.compile(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", re.IGNORECASE)
```

"The alias must not be glued to another letter or digit on either side." This handles both cases
correctly, and both are pinned by tests (`test_substrings_do_not_produce_false_positives`,
`test_symbol_heavy_skill_names_survive_normalisation`).

### The comparison

```python
matching = job_skills ∩ resume_skills      # present in both
missing  = job_skills − resume_skills      # asked for, not found
extra    = resume_skills − job_skills      # have it, not asked for
coverage = |matching| / |job_skills|
```

### Why deliberately NOT machine learning

| | Dictionary lookup | An ML skill-extraction model |
|---|---|---|
| Explainable | ✅ "we found the literal string `sklearn`" | ❌ "the model gave it 0.63" |
| Training data needed | none | thousands of labelled resumes |
| Auditable by a non-technical user | ✅ open the JSON file | ❌ |
| Extensible | ✅ edit one file | ❌ retrain |
| Finds unlisted skills | ❌ | ✅ |
| Understands context | ❌ | partly |

For a portfolio project where **explaining the system matters more than squeezing out accuracy**,
the dictionary wins. Being able to justify that choice is worth more in an interview than having
used a fancier method.

### Honest weaknesses

1. **Closed vocabulary** — only the ~50 listed skills can ever be found. Rust, Svelte and Julia
   are invisible.
2. **No negation handling** — "no experience with Java" registers as a Java mention.
3. **No proficiency or recency** — a one-weekend tutorial and three years of production work both
   count as one mention.
4. **No context** — "interested in learning Docker" counts as Docker.
5. **Alias lists are manual** — every new synonym is a hand edit.

These are stated in the UI too: the list is labelled "**Potentially** missing skills", never
"missing skills".

### The optional semantic layer

For each **missing** skill, the system embeds the skill name, embeds every sentence of the resume,
and finds the best cosine match. Above a threshold of `0.45`, the skill is reported as
"possibly related" **with the matching sentence shown as evidence**.

Example it catches: the job asks for "Machine Learning"; the resume says "trained predictive
models to forecast demand". No literal match — but semantically close.

**Three deliberate design decisions here:**
1. It runs **only on the skills the exact matcher missed** — the baseline is never overridden.
2. It **never changes the score**. It only adds explanation. (Pinned by
   `test_exact_skill_baseline_is_unaffected_by_the_semantic_layer`.)
3. It **always shows its evidence**, so the user can judge for themselves rather than trusting a
   number.

This is what lets you demo exact and semantic matching side by side and explain the difference —
which is exactly what §16 is about.

---

## 16. Exact keyword matching vs semantic matching

| | **Exact / keyword matching** | **Semantic matching** |
|---|---|---|
| Compares | literal strings | meaning (vectors) |
| Finds synonyms | ❌ no | ✅ yes |
| Explainable | ✅ completely | ⚠️ partly |
| Needs a model | ❌ no | ✅ yes (~80 MB) |
| Speed | microseconds | milliseconds |
| Needs a vocabulary list | ✅ yes | ❌ no |
| Handles unseen wording | ❌ no | ✅ yes |
| Can be fooled by buzzwords | ✅ easily | ⚠️ somewhat |
| Handles negation | ❌ no | ⚠️ partly |

### Worked examples

| Job asks | Resume says | Keyword | Semantic |
|---|---|---|---|
| "Python" | "Python" | ✅ match | ✅ high |
| "Machine Learning" | "ML modelling" | ❌ miss | ✅ high |
| "REST API" | "built FastAPI endpoints" | ❌ miss | ✅ high |
| "Java" | "JavaScript" | ⚠️ false positive if careless | ✅ correctly distinguishes |
| "Docker" | "no Docker experience" | ✅ **false positive** | ⚠️ still probably high |

### Why this project uses both

They fail in *different* ways, so together they cover for each other:

- **Semantic** handles synonyms and paraphrases → fewer false negatives.
- **Keyword** provides concrete, auditable evidence → a recruiter can verify each claim.
- Showing both side by side lets the user see *why* a score is what it is.

**The single-sentence interview version:** *"Keyword matching asks 'do these documents use the
same words?'; semantic matching asks 'do these documents mean the same thing?'. I implemented
both because keyword matching is explainable but brittle, and semantic matching is flexible but
opaque — so each covers the other's weakness."*

---

## 17. FastAPI architecture

### What FastAPI is

A modern Python web framework for building APIs. Chosen for three properties:

1. **Automatic validation** via Pydantic — malformed requests are rejected before your code runs.
2. **Automatic documentation** — an interactive OpenAPI page at `/docs`, generated from the code.
3. **Async support** — file uploads use `async def` so the server is not blocked while reading.

### The endpoints

| Method | Path | Input | Output |
|---|---|---|---|
| `GET` | `/health` | — | Status, model name, whether the model is loaded |
| `GET` | `/skills` | — | The full skill dictionary |
| `POST` | `/match` | JSON: `resume_text`, `job_description` | Full match report |
| `POST` | `/match/upload` | Multipart: PDF + job description | Full match report |
| `POST` | `/extract-text` | Multipart: PDF | Extracted plain text |

### How Pydantic validation works

```python
class MatchRequest(BaseModel):
    resume_text: str = Field(..., min_length=20)
    job_description: str = Field(..., min_length=20)
    use_semantic_skills: bool = True
```

Declaring this is enough. FastAPI will now automatically:
- parse the JSON body,
- check that both fields exist, are strings, and are at least 20 characters,
- return **HTTP 422** with a precise error message if not,
- and document all of it at `/docs`.

**No hand-written validation code anywhere.** That is the point of Pydantic.

### Error handling strategy

| HTTP code | Meaning here |
|---|---|
| `200` | Success |
| `400` | Bad file — not a PDF, corrupt, scanned, too large |
| `422` | Bad data — missing field, text too short |
| `500` | Unexpected server error (logged, but details never leaked to the client) |

### Layering — the separation of concerns

```
main.py       → HTTP concerns only: routing, validation, status codes
services/     → business logic, framework-agnostic
```

Nothing in `services/` imports FastAPI. That means the matching logic could be reused in a CLI, a
Jupyter notebook or a batch job with zero changes — and it can be unit-tested without spinning up
a web server. If asked "why not put the logic in the route handlers?", *that* is the answer.

### Why the model is a lazy singleton

```python
_model = None
_model_lock = threading.Lock()

def get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:          # re-check inside the lock
                _model = SentenceTransformer(...)
    return _model
```

- **Lazy** — the model loads on the *first match request*, not at import time, so `/health` and
  the test suite stay instant.
- **Singleton** — loading takes seconds and ~100 MB of RAM; doing it per request would be
  catastrophic.
- **Lock + double-check** — if two requests arrive simultaneously on a cold server, this prevents
  loading the model twice. (The classic "double-checked locking" pattern.)

---

## 18. Streamlit architecture

### What Streamlit is

A Python library that turns a plain script into a web app. No HTML, CSS or JavaScript required —
`st.button()` renders a button, `st.dataframe()` renders a table.

### The execution model (the one thing people get wrong)

**Streamlit re-runs the entire script from top to bottom on every interaction.** Click a button,
type in a box, toggle a switch — the whole file runs again.

This is why `st.session_state` exists: it is the only thing that survives a re-run.

```python
st.session_state.setdefault("result", None)   # survives re-runs
...
st.session_state.result = call_match(...)     # stored, so the report
                                              # persists when the user
                                              # toggles something else
```

Without this, the match report would vanish the moment the user touched any other widget.

### Why the UI has no AI logic in it

`frontend/app.py` contains **zero** AI code. It collects input, makes HTTP requests, and renders
results. All intelligence lives behind the API. This means:

- The UI can be swapped for React, a mobile app, or a CLI without touching the logic.
- The backend can be tested without a browser.
- The two can be deployed and scaled independently.
- A second client can reuse the same API.

If you are asked *"why not just call the services directly from Streamlit?"* — that is the answer.
(You *could*, and for a smaller project you might; the separation is a deliberate architectural
demonstration.)

### UI details worth pointing out in a demo

- The sidebar **live-checks backend health** and shows the exact command to start it if it is down.
- The report always includes an expander titled **"How this score was calculated"** showing the
  literal formula — the number is never presented as a black box.
- Missing skills are labelled "**Potentially** missing", with a caption explaining why.
- Semantically related skills show the **evidence sentence**, not just a number.
- The disclaimer is rendered as a warning banner on every single result.

---

## 19. Complete data flow

Following one real request end to end:

```
 1. USER          uploads resume.pdf + pastes a job description
                  in the Streamlit UI, clicks "Analyze Match"
                          │
 2. STREAMLIT     POST /extract-text   (multipart, the PDF bytes)
                          │
 3. FASTAPI       validates the upload → pdf_service.extract_text_from_pdf()
                          │
 4. PDF SERVICE   checks size + %PDF magic bytes
                  PyMuPDF → text  (pypdf as fallback)
                  empty result? → HTTP 400 "scanned PDF, no OCR"
                  clean_text(): rejoin hyphens, strip bullets, collapse spaces
                          │
 5. STREAMLIT     shows "Extracted 2,431 characters from 2 page(s)"
                  POST /match  {resume_text, job_description}
                          │
 6. FASTAPI       Pydantic validates both fields (≥20 chars)
                  → matching_service.match_resume_to_job()
                          │
        ┌─────────────────┴──────────────────┐
        │                                    │
 7a. SEMANTIC PATH                  7b. KEYWORD PATH
     clean both texts                   normalize_for_matching()
        │                                    │
     chunk into 180-word windows        search 50 skills × aliases
     (30-word overlap)                  with word-boundary regex
        │                                    │
     embed each chunk → 384-dim         resume_skills, job_skills
     (MiniLM, first call loads               │
      the model ~10s)                   matching / missing / extra
        │                                    │
     average chunks → doc vector        coverage = matching/job
     re-normalise to length 1                │
        │                                    │
     cosine(resume_vec, job_vec)             │
        │                                    │
     semantic_similarity %              skill_coverage %
        │                                    │
        └─────────────┬──────────────────────┘
                      │
 8. OPTIONAL   for each MISSING skill: embed it, compare against every
    SEMANTIC   resume sentence, keep matches above 0.45, attach evidence
    SKILL LAYER        │
 9. BLEND      overall = 0.7 × semantic + 0.3 × coverage
               (or pure semantic if the job listed no dictionary skills)
                       │
10. RESPONSE   Pydantic serialises MatchResponse → JSON (with disclaimer)
                       │
11. STREAMLIT  renders metrics, progress bar, formula expander,
               skill tables, evidence table, disclaimer banner
                       │
12. PRIVACY    the PDF bytes and all text are garbage-collected.
               Nothing was written to disk. Nothing was logged but sizes.
```

---

## 20. Example request

**Text matching:**

```bash
curl -X POST http://127.0.0.1:8000/match \
  -H "Content-Type: application/json" \
  -d '{
    "resume_text": "Final-year CS student. Built machine learning models with Python and scikit-learn. Deployed a semantic search service with FastAPI and Docker. Comfortable with SQL, Pandas and Git.",
    "job_description": "We are hiring an AI/ML intern. Required: Python, scikit-learn, NLP fundamentals, and experience building REST APIs with FastAPI. SQL and Git are expected. Docker and AWS are a plus.",
    "use_semantic_skills": true
  }'
```

**PDF upload:**

```bash
curl -X POST http://127.0.0.1:8000/match/upload \
  -F "resume_file=@data/sample_resume.pdf" \
  -F "job_description=We are hiring an AI/ML intern with Python and NLP experience..." \
  -F "use_semantic_skills=true"
```

**In Python:**

```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/match",
    json={
        "resume_text": open("resume.txt").read(),
        "job_description": open("job.txt").read(),
    },
)
report = response.json()
print(report["overall_match_score"], report["missing_skills"])
```

---

## 21. Example response

This is the **real response** to the first `curl` command in §20 — nothing here is invented.

```json
{
  "overall_match_score": 65.2,
  "semantic_similarity_score": 63.14,
  "match_band": "Good alignment",

  "score_breakdown": {
    "semantic_similarity_percent": 63.14,
    "skill_coverage_percent": 70.0,
    "semantic_weight": 0.7,
    "skill_weight": 0.3,
    "formula": "0.7 x semantic(63.14%) + 0.3 x skill_coverage(70.0%)"
  },

  "resume_skills": ["Python", "SQL", "Machine Learning", "Scikit-learn",
                    "Pandas", "FastAPI", "Docker", "Git"],
  "job_skills":    ["Python", "SQL", "Machine Learning", "NLP", "Scikit-learn",
                    "FastAPI", "Docker", "AWS", "Git", "REST API"],
  "matching_skills": ["Python", "SQL", "Machine Learning", "Scikit-learn",
                      "FastAPI", "Docker", "Git"],
  "missing_skills": ["NLP", "AWS", "REST API"],

  "semantically_related_skills": [],

  "extra_resume_skills": ["Pandas"],
  "resume_char_count": 180,
  "job_char_count": 181,

  "disclaimer": "This score measures TEXT SIMILARITY between a resume and a job description. It is an educational prototype, not a hiring tool: it does not measure competence and it is not a probability of being hired or shortlisted. Always review the underlying skills and evidence yourself."
}
```

**How to read it:** the resume covers 7 of the 10 skills the job asks for. Three are missing —
NLP, AWS and REST API.

### Why is `semantically_related_skills` empty here?

This is worth understanding, because it shows the threshold doing its job. The resume *does* say
"Deployed a semantic search service with **FastAPI**", which a human would accept as evidence of
REST API work. But the semantic layer compares the skill name against individual **sentences**,
and this resume is only three short sentences long — none of them clears the `0.45` cosine
threshold against the bare string "REST API".

Run the longer sample files instead (`data/sample_resume.pdf` against
`data/sample_job_description.txt`) and the layer does fire:

```json
"missing_skills": ["TensorFlow", "Flask", "AWS", "CI/CD", "LLM", "RAG"],
"semantically_related_skills": [
  {
    "skill": "TensorFlow",
    "similarity": 0.5683,
    "evidence": "- Deep Learning with PyTorch (online)"
  }
]
```

That is a genuinely good illustration of both the value and the limits of the semantic layer: it
correctly spotted that a *PyTorch deep learning* certification is closely related to TensorFlow —
same field, adjacent tooling — and it handed the evidence back for a human to judge rather than
quietly awarding the skill.

It also shows the honest weakness: **the threshold is a tuning knob, not a truth.** Lower it and
you get more hints plus more noise; raise it and you miss real matches. There is no labelled data
here to tune it against, so `0.45` is a hand-picked default, documented as such in
`backend/config.py`.

---

## 22. Limitations

Be the first to raise these. Interviewers respect a candidate who knows where their own system
breaks.

### Technical

1. **No OCR** — scanned resumes cannot be read at all.
2. **Closed skill vocabulary** — only the ~50 skills in `skills.json` exist to the system.
3. **No negation handling** — "no experience with Java" counts as Java.
4. **No proficiency or recency** — one tutorial equals three years.
5. **English only** — the model is trained on English.
6. **Chunk averaging dilutes** long documents (§14).
7. **Multi-column PDFs** can extract in a scrambled reading order (§13).
8. **256-token model limit** — mitigated by chunking, not eliminated.

### Methodological

9. **Uncalibrated scores** — cosine values cluster in a narrow band; 50% does not mean "half".
10. **Hand-chosen weights** — the 0.7/0.3 split is a design choice, not a learned parameter.
11. **Hand-chosen bands** — the Low/Moderate/Good/Strong thresholds are a readability aid.
12. **No ground truth** — no labelled dataset exists here, so **no accuracy figure is claimed**.
13. **Scores are only comparable within one job posting**, never across postings.

### Conceptual — the deepest limitation

14. **Text similarity is not job suitability.** A resume is a *description* of a person, written
    persuasively. The system measures the description. Someone who writes well about work they
    barely did scores higher than someone who did the work and wrote about it plainly. **No
    amount of engineering fixes this — it is inherent to matching on documents.**

---

## 23. Bias/fairness considerations

### Where bias can enter

**1. The pretrained model.** `all-MiniLM-L6-v2` was trained on web text, which carries the
associations present in that text — including gendered and cultural ones. If the training data
associated certain names or phrasings with certain professions, those associations persist in
the embedding space. Nobody put them there deliberately; nobody removed them either.

**2. The skill dictionary.** Ours is heavily weighted toward Western, modern, English-language
tech stacks. A candidate using regionally common tools that are not listed appears less skilled
than they are — the dictionary's blind spot becomes the candidate's penalty.

**3. Writing style, not ability.** This is the big one. The system rewards resumes written in
confident, jargon-dense, industry-standard English. That systematically favours:
   - native English speakers over equally capable non-native speakers,
   - candidates who had access to resume coaching,
   - people already inside the industry who know its vocabulary,
   - people from backgrounds where self-promotion is culturally normal.

   **None of this correlates with ability to do the job.**

**4. Resume formatting.** Candidates using ATS-friendly templates extract cleanly. Someone using a
creative two-column design gets scrambled text and a lower score — penalised for design taste.

**5. Automation bias.** Even labelled "educational prototype", a number on a screen carries
authority. People trust numbers more than they should, and defer to them more than they intend
to.

### What this project does about it

- **Never auto-rejects.** The system only reports; it takes no action.
- **Always shows its reasoning** — the formula, the skills, the evidence sentences. A human can
  see *why* and disagree.
- **Explicit disclaimers** in the API body, the UI and the docs.
- **The skill dictionary is a visible, editable JSON file**, not hidden logic.
- **No demographic fields exist anywhere** — no name, gender, age or nationality is ever parsed
  or used.

### What a real, responsible system would need

1. **Bias auditing** — swap names (e.g. typically male/female, or of different ethnic origin) on
   otherwise identical resumes and measure whether scores shift. Any shift is bias, measurably.
2. **Human-in-the-loop by design** — the tool ranks for review, never filters automatically.
3. **Anonymisation** — strip names, addresses and universities before scoring.
4. **Adverse-impact monitoring** — track outcome rates across demographic groups over time
   (the "four-fifths rule" is the standard US benchmark).
5. **Right to explanation and appeal** — required under the EU AI Act, which classifies
   employment-related AI as **high-risk**.
6. **Regular revalidation** — job markets and language drift.

### The honest bottom line

**Any resume-matching system, including this one, risks amplifying existing inequities at scale.**
A biased human reviewer affects the resumes they personally read. A biased algorithm applies the
same bias to every applicant, consistently, invisibly, forever. *Consistency is not fairness.*

---

## 24. Privacy considerations

Resumes are dense personal data: full name, email, phone, address, employer history, education,
sometimes age or nationality. Under GDPR this is straightforwardly personal data.

### What this system does

| Practice | How |
|---|---|
| **No permanent storage** | PDFs are read into memory, converted, used, discarded. `del pdf_bytes` runs as soon as extraction finishes. Nothing touches disk. |
| **No content logging** | Every log line records *lengths and counts only*, never text. Verify in `backend/main.py` — `len(...)` everywhere, never the string. |
| **No third-party calls** | The model runs locally. Resume text never leaves the machine — there is no external API in the request path. |
| **No database** | Nothing to breach, no backups to leak, no retention policy to violate. |
| **No accounts, sessions or cookies** | No identity is ever established. |
| **No personal data in the repo** | `data/sample_resume.txt` is fictional. |
| **Defence-in-depth in `.gitignore`** | `*_resume.pdf`, `uploads/`, `*.docx` and `.env` are all blocked, so a real resume cannot be committed by accident. |

### Why "no third-party calls" matters

Had this been built on a hosted embedding API, every resume would be transmitted to an external
company, subject to *their* retention policy, *their* jurisdiction, and *their* training-data
practices. Running an 80 MB model locally means the data simply never leaves. **That is a privacy
argument for choosing a small local model, not just a cost argument** — worth saying out loud.

### What a public deployment would additionally require

1. **HTTPS everywhere** — otherwise resumes travel in plaintext.
2. **An explicit consent notice** before upload, stating purpose and retention.
3. **Rate limiting** — prevent scraping and abuse.
4. **A documented retention policy** (ideally: zero retention, as here).
5. **GDPR lawful basis** — consent or legitimate interest, documented.
6. **Data subject rights** — access, deletion, portability. *(Trivial here: we store nothing, so
   there is nothing to delete — a genuinely strong position.)*
7. **A DPIA** (Data Protection Impact Assessment) — required for large-scale processing of
   personal data in the EU.
8. **Authentication** — so arbitrary strangers cannot use your compute.

---

## 25. How to improve the system

Ordered by impact-per-effort — a useful framing if asked "what would you do next?".

### Quick wins

1. **Expand the skill dictionary** to a real ontology (ESCO, O*NET, or LinkedIn's taxonomy) —
   thousands of skills with genuine synonym and parent/child relations.
2. **Negation detection** — a small rule layer: if "no", "without", "not" or "learning" appears
   within a few tokens before a skill, flag it rather than counting it.
3. **Section-aware parsing** — detect "Experience", "Skills", "Education" headings and weight
   them differently. Ten years of work should outweigh a line in an interests list.
4. **Swap in a stronger model** — `all-mpnet-base-v2` is more accurate; it is a one-line config
   change, which is exactly why the model name is a setting.

### Bigger improvements

5. **Named Entity Recognition** — a fine-tuned model (or spaCy) to extract skills, employers,
   degrees and dates from text, finding what no dictionary contains.
6. **Cross-encoder re-ranking** — a *bi-encoder* (what we use) embeds each document separately,
   which is fast but loses interaction between them. A **cross-encoder** reads both texts
   *together* and is substantially more accurate. The standard pattern is a two-stage pipeline:
   bi-encoder to shortlist the top 50 from thousands, cross-encoder to re-rank those 50. Best of
   both.
7. **Experience-level extraction** — parse "3 years of Python" and compare against the
   requirement, instead of treating every mention as equal.
8. **Education extraction** — check degree requirements explicitly.
9. **Attention-weighted chunk pooling** — fix the dilution problem by weighting relevant chunks
   above boilerplate, instead of a flat average.
10. **LLM-assisted explanations** — "Your FastAPI project satisfies the REST API requirement" in
    natural language. Note the trade-off: an LLM introduces hallucination risk and, if hosted,
    breaks the privacy guarantee of §24.

### Making it trustworthy

11. **Build an evaluation dataset** (see §27) — without this, every other improvement is guesswork.
12. **Bias auditing pipeline** — automated name-swap tests on every model change.
13. **Calibration** — map raw cosine values onto meaningful bands using actual labelled data.
14. **OCR fallback** (Tesseract) for scanned resumes.
15. **Multilingual support** — `paraphrase-multilingual-MiniLM-L12-v2` covers 50+ languages.

---

## 26. How to scale it

The current system handles **one resume vs one job**. Scaling means **N resumes vs M jobs**.

### The bottleneck

Embedding is the expensive step (~50 ms per document on CPU). Cosine similarity is nearly free.
So: **never re-embed anything you have already embedded.**

### Step 1 — Cache embeddings

Embed each resume and each job **once**, store the vector, reuse it forever. A resume's vector
does not change unless the resume does. This alone removes almost all repeated work.

### Step 2 — Vector database

Store vectors in FAISS, Qdrant, Pinecone or pgvector. These use **Approximate Nearest Neighbour**
indexes (HNSW, IVF) to find the closest vectors among millions in milliseconds, instead of
comparing against every single one.

```
Brute force:  1 job × 1,000,000 resumes = 1,000,000 comparisons
ANN index:    1 job × 1,000,000 resumes ≈ a few thousand comparisons
```

The trade-off is in the name: *approximate*. You may miss a true nearest neighbour occasionally —
almost always an acceptable price.

### Step 3 — Batch and use a GPU

`model.encode()` accepts a list. Embedding 64 documents in one batch is far faster than 64
separate calls, because it is one large matrix multiplication. On a GPU the effect is dramatic.

### Step 4 — Scale the service

- **Horizontal scaling** — run several stateless FastAPI workers behind a load balancer. The app
  holds no per-user state, so this works with no changes.
- **Separate the model server** — keep embedding on GPU machines, API logic on cheap CPU ones.
- **Async job queue** (Celery, RQ) — bulk jobs return a job ID immediately and process in the
  background, so HTTP requests never time out.
- **Cache identical requests** (Redis, keyed by a hash of the text).

### Step 5 — The two-stage retrieval pattern

The standard production architecture for exactly this problem:

```
1,000,000 resumes
      ↓  bi-encoder + ANN index (fast, approximate)
    top 100 candidates
      ↓  cross-encoder re-ranking (slow, accurate)
    top 10 for human review
```

Fast and cheap where volume is high; slow and accurate where precision matters.

### Realistic capacity estimates

| Setup | Throughput |
|---|---|
| Current (1 CPU worker) | ~10–20 matches/second after warm-up |
| 4 CPU workers + cached embeddings | ~100 matches/second |
| GPU batch embedding | thousands/second |
| Vector DB with pre-computed vectors | effectively instant retrieval |

---

## 27. How to evaluate the system properly

**This project claims no accuracy figure, because doing so honestly requires work that has not
been done.** Explaining *how* you would evaluate it is often worth more in an interview than
quoting a number you cannot defend.

### Step 1 — You need ground truth

Semantic similarity is **unsupervised** — there is no correct answer to compare against. To
evaluate, you must create one:

- Collect real (resume, job description) pairs.
- Have **multiple human recruiters** label each pair: strong fit / partial fit / poor fit.
- Measure **inter-annotator agreement** (Cohen's κ). *If humans disagree with each other, no
  model can be "right" — and this is genuinely common in hiring.*
- Split into train / validation / test sets.

You need a few hundred pairs minimum; ideally thousands.

### Step 2 — Choose metrics that fit the task

**If framed as ranking** (the realistic framing — "show the best candidates first"):

| Metric | What it tells you |
|---|---|
| **Precision@k** | Of the top 10 ranked, how many did humans call good fits? |
| **Recall@k** | Of all good fits, how many made the top 10? |
| **NDCG** | Ranking quality, weighted so errors near the top cost more |
| **MRR** | How high does the first genuinely good candidate appear? |
| **Spearman correlation** | Does the model's ordering track the humans' ordering? |

**If framed as classification** (fit / no fit, using a threshold):

| Metric | What it tells you |
|---|---|
| **Precision** | Of those we called a match, how many really were? |
| **Recall** | Of the real matches, how many did we find? |
| **F1** | Their harmonic mean |
| **ROC-AUC** | Threshold-independent discrimination ability |
| **Confusion matrix** | Where exactly it fails |

**Accuracy alone would be misleading**, because the classes are heavily imbalanced — most
resume/job pairs are poor fits, so a model that says "no match" every time scores well on
accuracy while being useless.

### Step 3 — Understand the two error types

**False positive** — the system scores a poor candidate highly.
*Cost:* a recruiter wastes time on a bad interview. Annoying, recoverable.

**False negative** — the system scores a good candidate poorly.
*Cost:* a qualified person is never seen. **Invisible, unappealable, and far more serious.**

In hiring, **false negatives matter more than false positives**, and they are also the harder
type to detect — you never find out about the good candidate you filtered out. This asymmetry
should drive the threshold: lean toward **recall**, and let humans filter the extras.

### Step 4 — Compare against baselines

A number means nothing without a comparison. Always benchmark against:

1. **Random ordering** — the absolute floor.
2. **TF-IDF cosine similarity** — the classic keyword baseline. *If your transformer cannot beat
   TF-IDF, the added complexity is not earning its keep.*
3. **BM25** — the strong information-retrieval baseline.
4. **Exact skill overlap alone** — half of this very system.
5. **A larger embedding model** — is MiniLM costing you much?
6. **A cross-encoder** — the realistic quality ceiling for this approach.

### Step 5 — Test robustness and fairness, not just accuracy

- **Perturbation tests** — reword a resume without changing its meaning. The score should barely
  move. If it swings wildly, the system is measuring style, not substance.
- **Adversarial tests** — paste the job description into the resume. Score should ideally *not*
  hit 100%. (In this system it does — an honest, demonstrable weakness.)
- **Bias tests** — swap names of different apparent gender or ethnicity on otherwise identical
  resumes. **Any score change is measurable bias.**
- **Format tests** — the same resume as one-column vs two-column PDF should score similarly.
- **Length tests** — does padding a resume with irrelevant text raise or lower the score?

### Step 6 — Evaluate in production, carefully

- **A/B testing** against the current process, measuring downstream outcomes.
- **Track real outcomes** — did highly-scored candidates actually perform well in interviews?
- **Monitor drift** — job-market language changes; a model trained in 2021 ages.
- **Collect recruiter feedback** — a thumbs-up/down on each result is cheap labelled data.

### The summary answer for an interview

> *"I deliberately don't quote an accuracy figure, because I have no labelled evaluation data —
> quoting one would be fabricating it. To evaluate this properly I'd collect several hundred
> resume/job pairs labelled by multiple recruiters, check inter-annotator agreement first,
> then measure NDCG and Precision@k for the ranking task against TF-IDF and BM25 baselines.
> I'd weight recall over precision, because a false negative means a qualified person is never
> seen and never finds out. And I'd run perturbation and name-swap tests, because a system that
> scores on writing style rather than substance can be accurate on average and still unfair to
> individuals."*

---

## Where to look in the code

| Concept | File |
|---|---|
| Settings, model name, weights | [`backend/config.py`](backend/config.py) |
| API contract (Pydantic) | [`backend/models.py`](backend/models.py) |
| Routes, validation, errors | [`backend/main.py`](backend/main.py) |
| Cleaning, normalising, chunking | [`backend/text_utils.py`](backend/text_utils.py) |
| PDF → text | [`backend/services/pdf_service.py`](backend/services/pdf_service.py) |
| Text → vectors | [`backend/services/embedding_service.py`](backend/services/embedding_service.py) |
| Cosine similarity + scoring | [`backend/services/matching_service.py`](backend/services/matching_service.py) |
| Skill dictionary matching | [`backend/services/skill_service.py`](backend/services/skill_service.py) |
| The skill dictionary itself | [`data/skills.json`](data/skills.json) |
| UI | [`frontend/app.py`](frontend/app.py) |
| Tests | [`tests/`](tests/) |

**Next:** [VIVA_QUESTIONS.md](VIVA_QUESTIONS.md) — 40 practice questions with answers, plus the
60-second and 2-minute spoken explanations.
