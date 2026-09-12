# 🎓 VIVA / INTERVIEW QUESTIONS — AI Resume–Job Matching System

40 questions with answers, written to be **spoken out loud**, not recited from a textbook.
Every answer refers to what this project actually does.

Concept background: **[SYSTEM_GUIDE.md](SYSTEM_GUIDE.md)** · Setup and usage: **[README.md](README.md)**

---

## Contents

- [Part A — NLP fundamentals](#part-a--nlp-fundamentals) (Q1–5)
- [Part B — Embeddings](#part-b--embeddings) (Q6–12)
- [Part C — Similarity & maths](#part-c--similarity--maths) (Q13–19)
- [Part D — This system's design](#part-d--this-systems-design) (Q20–27)
- [Part E — Evaluation & error analysis](#part-e--evaluation--error-analysis) (Q28–32)
- [Part F — Limitations, ethics, scale](#part-f--limitations-ethics-scale) (Q33–40)
- [🎤 60-second project explanation](#-60-second-project-explanation)
- [🎤 2-minute technical explanation](#-2-minute-technical-explanation)
- [Rapid-fire one-liners](#rapid-fire-one-liners)
- [Questions to ask them](#questions-to-ask-them)

---

## Part A — NLP fundamentals

### Q1. What is NLP?

Natural Language Processing is the branch of AI that lets computers work with human language.
It's hard because language is ambiguous ("Apple" the fruit or the company?), context-dependent,
infinitely variable — there are countless ways to say the same thing — and completely
unstructured, with no rows or columns.

Typical NLP tasks are classification, named entity recognition, translation, summarisation and
question answering. My project uses **semantic textual similarity** — measuring how close in
meaning two documents are.

In this system the NLP steps are: extract text from a PDF, clean it, split it into chunks,
convert each chunk into an embedding, and compare the embeddings.

---

### Q2. What is semantic similarity, and how is it different from word overlap?

Semantic similarity measures how close two texts are in **meaning**, regardless of the words used.

The clearest way to show the difference is with two examples that break word overlap completely:

- *"I build ML models in Python"* and *"I develop machine learning systems using Python"* share
  almost no words, but mean nearly the same thing — **low overlap, high semantic similarity**.
- *"I love this job"* and *"I hate this job"* differ by one word out of four, so word overlap is
  very high — but they mean opposite things. **High overlap, low semantic similarity.**

Word overlap and meaning can point in completely opposite directions. That's exactly why keyword
matching is unreliable for resumes, where the same skill gets described a dozen different ways.

---

### Q3. What is text preprocessing, and what did you do here?

Preprocessing is preparing raw text for a model. The interesting part of my answer is what I
deliberately **didn't** do.

Classic preprocessing — lower-casing, removing stop-words, stemming — was designed for
bag-of-words models that treat text as an unordered pile of tokens. **Transformers are the
opposite.** They're trained on normal human sentences with capitalisation, punctuation and
stop-words, because those carry meaning. Remove "no" from "no experience with Java" and you
invert the sentence.

So for the embedding path I only fix the mess that *PDF extraction* introduces: re-joining words
split across a line break by a hyphen, stripping bullet glyphs, and collapsing whitespace.

I do have a second, aggressive pipeline — lower-casing and punctuation stripping — but it feeds
only the **keyword** matcher, which does want that. Two consumers, two different preparations.

---

### Q4. How do you handle a PDF resume?

I use PyMuPDF as the primary extractor, with pypdf as a fallback since different libraries cope
with different malformed files.

Before parsing, I reject empty uploads, files over 5 MB, and anything without the `%PDF` magic
bytes in its first kilobyte.

The important case is the **scanned PDF**. A scanned resume is a *photograph* of text — there's no
text layer, so extraction returns an empty string. Reading it would need OCR, which I deliberately
didn't include. What matters is that the system **detects this and raises a clear error** instead
of silently matching an empty document and returning a meaningless score. Failing loudly beats
failing silently.

---

### Q5. What is a token, and why does it matter here?

A token is the unit a model actually reads. Modern models use **sub-word** tokens, so a rare word
like "tokenization" might split into "token" + "ization". That lets a fixed vocabulary handle any
word it has never seen.

It matters here because of a hard limit: **`all-MiniLM-L6-v2` truncates at 256 word-pieces**,
roughly 180–200 words. A 600-word resume fed in whole would have two thirds of it **silently
discarded** — your entire projects section could vanish with no warning.

That limit is the single reason I implemented chunking, which is Q22.

---

## Part B — Embeddings

### Q6. What is an embedding?

An embedding is a list of numbers that represents the meaning of a piece of text. In my project
every piece of text becomes **384 numbers**.

The useful property is that texts with similar meanings get similar number-lists. So "Python
developer" and "software engineer who writes Python" end up close together in that
384-dimensional space, while "pastry chef" ends up far away.

The intuition I'd offer: it's like a map. On a map, cities that are close on paper are close in
reality. In an embedding space the same holds, except there are 384 dimensions instead of two,
and "distance" means difference in meaning rather than kilometres.

---

### Q7. Why do semantically similar texts get similar vectors? What makes that happen?

Because the model was explicitly **trained** for it — it's not an accident or an emergent
coincidence.

During training the model saw millions of sentence pairs labelled as similar or unrelated, and
its weights were adjusted so similar pairs got **pulled together** in the space and unrelated
pairs got **pushed apart**. That's **contrastive learning**.

After around a billion such pairs, the model has internalised a general rule — *text meaning
similar things points in similar directions* — and it applies that rule to sentences it has never
seen, including my resume.

---

### Q8. Why use embeddings at all instead of just counting keywords?

Because generalisation. I could hand-write rules — "ML" means "machine learning", "built
endpoints" means "API development" — but I'd need thousands of them, they'd never be complete, and
every new job domain would need a fresh set.

A pretrained model has already learned those relationships from a billion sentence pairs. It
knows "machine learning" and "ML modelling" appear in similar contexts **without anyone writing a
rule for that specific pair.** That's the real value: it handles word combinations nobody
anticipated.

---

### Q9. Why use a *pretrained* model rather than training your own?

Three reasons, all practical.

**Data.** Training a sentence embedding model from scratch needs around a billion sentence pairs.
I have none.

**Compute.** That's thousands of GPU-hours, costing tens of thousands of dollars.

**It would be worse.** Even with data and compute, I couldn't beat a model the research community
has tuned for years.

Using a pretrained model is **transfer learning**: general language understanding was learned
elsewhere, and I apply it to my specific task. It's the standard approach precisely because it
works.

Fine-tuning would only make sense if I had thousands of labelled resume/job pairs — and even then
I'd start from the pretrained weights, not from scratch.

---

### Q10. What is sentence-transformers?

It's a Python library of transformer models fine-tuned specifically to produce good
**sentence-level** embeddings. Plain BERT gives you one vector per token and isn't great at
sentence similarity out of the box; sentence-transformers models are trained with a
similarity objective, so their output vectors are directly comparable.

The API is genuinely two lines:

```python
model = SentenceTransformer("all-MiniLM-L6-v2")
vectors = model.encode(["some text", "other text"])
```

---

### Q11. Walk me through what happens inside the model.

Five stages:

1. **Tokenization** — the sentence splits into sub-word tokens, each mapped to an ID.
2. **Embedding lookup** — each token gets an initial vector, plus a position encoding so the model
   knows word order.
3. **Six transformer layers** — this is where **self-attention** happens. Each token's vector is
   rebuilt by weighing every other token in the sentence. That's how ambiguity resolves: "Python"
   surrounded by "models" and "build" lands in the programming region of the space, not the
   zoology one.
4. **Mean pooling** — all the token vectors are averaged into **one** 384-dimensional sentence
   vector. That's how variable-length text becomes a fixed-size vector.
5. **Normalisation** — scale it to length 1.

That last step has a nice practical consequence: for unit vectors, the **dot product *is* the
cosine similarity**, so similarity becomes one multiply-and-add pass. I use that shortcut in the
semantic skill matcher.

---

### Q12. Why `all-MiniLM-L6-v2` specifically? Why not something bigger?

It's an explicit trade-off, and I'd defend it as the right one for this project.

MiniLM is a **distilled** BERT — a small "student" model trained to imitate a large "teacher".
Six layers instead of twelve, 22 million parameters, 80 MB on disk, 384-dimensional output. It
runs on a laptop CPU in milliseconds.

A larger model like `all-mpnet-base-v2` is more accurate but five times the size and noticeably
slower. For **demonstrating the concept**, speed and the ability to run anywhere matter more than
the last few points of accuracy.

Two things I'd add. First, the model name is a **config setting**, not a hard-coded string —
swapping it is one line in `backend/config.py`. Second, there's a privacy argument, not just a
cost one: because the model is small enough to run locally, **resume text never leaves the
machine.** A hosted embedding API would have meant transmitting every resume to a third party.

---

## Part C — Similarity & maths

### Q13. What is a vector?

An ordered list of numbers — geometrically, an arrow from the origin to a point. `[3, 4]` points
right and up; its length is √(9+16) = 5.

For embeddings, the **direction carries the meaning**. Two vectors pointing the same way represent
similar meanings, no matter how long they are.

I can't picture 384 dimensions and I don't need to — the maths of dot products, lengths and
angles works identically in any number of dimensions. I reason in 2D and compute in 384D.

---

### Q14. What is cosine similarity? Give me the formula.

It measures the **angle** between two vectors:

```
cosine(A, B) = (A · B) / (‖A‖ × ‖B‖)
```

The numerator is the dot product — the sum of element-wise products, large when the vectors agree
component by component. The denominator is the two lengths, which divides magnitude out and leaves
only direction.

The range is −1 to 1: **1** means identical direction, **0** means unrelated (perpendicular),
**−1** means opposite. In practice this model almost always outputs values between 0 and 1.

A quick 2D check: A = [3,4], B = [4,3]. Dot product = 24, both lengths = 5, so cosine = 24/25 =
**0.96** — very similar direction.

---

### Q15. Why cosine similarity instead of Euclidean distance?

The main reason is **length independence**.

A resume is 600 words; a job description is 250. Both describe the same profile. Euclidean
distance is sensitive to magnitude, so the longer document looks "far away" simply for being
longer. Cosine divides magnitude out — only direction, meaning, survives.

The concrete demonstration: A = [1,1] and B = [10,10] are the same content at ten times the
length. Cosine says **1.0** — correctly, identical. Euclidean says **12.7** — wrongly, very
different. I assert that exact property in my test suite.

Three supporting reasons: cosine has a **bounded range** so it converts naturally to a percentage,
while Euclidean is unbounded; it **matches how the model was trained**, since sentence-transformer
objectives are cosine-based; and it's **cheap** — one dot product for unit vectors.

**And here's the honest caveat I'd volunteer:** when vectors are already normalised, as mine are,
cosine and Euclidean are monotonically related — `euclidean² = 2 − 2·cosine`. So for *this*
pipeline they'd rank pairs **identically**. The choice is about interpretability and convention,
not a different ordering. The length-independence argument is genuinely decisive when vectors
aren't normalised, which is the general case.

---

### Q16. What does your match score actually mean?

I report two numbers deliberately.

**Semantic similarity** is the raw cosine between the two document vectors, times 100. That's a
linear rescale for display — it adds no statistical meaning.

**Overall match score** is a transparent blend:

```
overall = 0.7 × semantic_similarity%  +  0.3 × skill_coverage%
```

I blend them because each alone is misleading. Semantic similarity alone rewards writing *about*
the right topic — a buzzword-stuffed resume with no real projects scores well. Skill coverage
alone is pure keyword matching, with all the synonym problems. Requiring both means a strong score
needs topical closeness *and* concrete named-skill overlap.

Two things I'd flag without being asked. **The 0.7/0.3 weights are a design choice, not learned
parameters** — there's no labelled data here to learn them from. And the score is **relative, not
absolute**: comparing five candidates against the *same* posting is meaningful; comparing "62% on
job A" with "58% on job B" is not, because the postings differ in length and vocabulary.

---

### Q17. Is the match score a probability of being hired?

**No, and this is the point I'd be most careful about.**

A probability requires a prediction target and labelled outcomes. To say "72% chance of being
hired" I'd need thousands of historical resume/job/hired-or-not records, a model trained to
predict that outcome, and a held-out test set showing the predictions are calibrated.

**I have none of those.** This system was never trained on hiring outcomes — it has never seen a
single hiring decision. It measures text similarity, full stop.

The clearest proof is the failure cases. If a candidate copy-pastes the job ad into their resume,
the score approaches 100% — and they're a terrible candidate. A brilliant engineer with a terse,
understated resume might score 45%. Neither is a bug; both show the metric measures **documents,
not people.**

That's why the disclaimer is in the API response body, on the UI, and in the README. It's part of
the deliverable, not decoration.

---

### Q18. Why does a good match only score around 70% and not 95%?

Because the two documents genuinely *are* different kinds of text. A resume is a personal history
written in the first person; a job description is a company's wish-list written in the second.
They're about the same subject but they're different genres, so their vectors never fully align.

More generally, cosine values from this model cluster in a fairly narrow band — most real-world
pairs land between 0.3 and 0.8. **The scores are uncalibrated**, so 50% doesn't mean "half as
good".

This is exactly why I say scores should be used **comparatively**. If five candidates score 71,
64, 58, 44 and 31 against the same posting, that ordering is informative. The absolute value
isn't.

---

### Q19. What's the difference between a bi-encoder and a cross-encoder?

A **bi-encoder** — what I use — embeds each document **separately**, then compares the two
vectors. The advantage is huge: you can pre-compute and cache every resume vector, so matching is
just a dot product. The cost is that the two texts never "see" each other; each is summarised into
384 numbers before any comparison happens.

A **cross-encoder** feeds **both texts together** into the model, which attends across them
jointly and outputs a similarity score directly. It's substantially more accurate — but you can't
cache anything, and every pair needs a full forward pass, so it doesn't scale.

The standard production pattern uses both: a bi-encoder to shortlist the top 100 from a million,
then a cross-encoder to re-rank those 100. Fast where volume is high, accurate where precision
matters. That's the top item on my improvements list.

---

## Part D — This system's design

### Q20. Walk me through what happens when a resume is uploaded.

1. Streamlit posts the PDF bytes to `POST /extract-text`.
2. The PDF service checks size and the `%PDF` magic bytes, then tries PyMuPDF, falling back to
   pypdf.
3. If no text comes out, it raises a 400 saying the PDF is probably scanned and there's no OCR.
4. The text is cleaned — hyphenated line breaks rejoined, bullets stripped, whitespace collapsed.
5. The UI shows "extracted N characters from M pages" with a preview.
6. On "Analyze Match", Streamlit posts the resume text and job description to `POST /match`.
7. Both documents are chunked into 180-word overlapping windows, every chunk is embedded, and the
   chunk vectors are averaged into one document vector per document.
8. Cosine similarity between the two document vectors gives the semantic score.
9. In parallel, the dictionary matcher finds skills in each document and computes coverage.
10. The two combine into the overall score, and the JSON report goes back with the disclaimer.
11. **The PDF bytes and all text are garbage-collected. Nothing was written to disk.**

---

### Q21. How does skill extraction work, and why isn't it machine learning?

It's a **dictionary lookup with aliases**. A JSON file holds about 50 skills, each with a
canonical name and a list of spellings — so `sklearn`, `scikit learn` and `scikit-learn` all
report as "Scikit-learn".

The detail I'd point out is the **word-boundary problem**. The naive version, `if "java" in text`,
matches "**Java**Script" and credits Java to someone who's never written it. The usual fix is the
`\b` regex boundary — but `\b` breaks on `C++`, `C#` and `CI/CD`, because those end in non-word
characters. So I use explicit look-around: the alias must not be glued to another letter or digit
on either side. Both cases are pinned by tests.

**Why not ML?** Because for *this* half of the system, the dictionary is genuinely better.
It's fully explainable — I can say "we found the literal string `sklearn`". It needs no training
data. A non-technical user can audit and extend it by opening one JSON file. An ML extractor would
give me "0.63 confidence", which no recruiter can act on or contest.

Knowing when **not** to reach for ML is the actual engineering judgement here.

---

### Q22. Why did you split documents into chunks? Isn't that over-engineering?

No — without it the system would be quietly broken.

`all-MiniLM-L6-v2` **truncates at 256 word-pieces**, about 180–200 words. Feed it a 600-word
resume and it reads the first third and **silently discards the rest**. No error, no warning — the
projects section just disappears.

So I split each document into overlapping 180-word windows, embed each, average the vectors, and
re-normalise. The **30-word overlap** exists because a sentence sitting on a chunk boundary would
otherwise be cut in half in every chunk, and neither half carries its full meaning.

I have a test, `test_chunked_embedding_differs_from_naive_truncation`, that proves chunking
actually changes the result — it's not decoration.

The honest weakness is **dilution**: a five-page resume with one highly relevant page gets
averaged down by the other four. Attention-weighted pooling would fix it; I chose averaging for
explainability.

---

### Q23. Why FastAPI and Streamlit as two separate processes? Why not one app?

You *could* import the services directly into Streamlit, and for a smaller project that'd be
reasonable. I separated them deliberately.

**FastAPI** gives me automatic Pydantic validation — I declare `min_length=20` and malformed
requests get a 422 before my code runs — plus auto-generated interactive docs at `/docs`, which is
excellent for a demo.

**The separation** means the matching logic is reusable by any client: a React frontend, a CLI, a
batch job. It's testable through a stable HTTP contract instead of through a browser. And the two
can be deployed and scaled independently — in production you'd want the model on GPU machines and
the UI on cheap ones.

The rule I kept: **nothing in `services/` imports FastAPI**, and `frontend/app.py` contains zero
AI logic. Each layer can be replaced without touching the others.

---

### Q24. How does Streamlit actually work? What's `session_state` for?

The thing people get wrong is the execution model: **Streamlit re-runs the entire script top to
bottom on every interaction.** Click a button, type in a box, flip a toggle — the whole file runs
again.

`st.session_state` is the only thing that survives a re-run. I store the match result there, so
the report stays on screen when the user toggles an unrelated option. Without it, the report would
vanish the instant they touched any other widget.

---

### Q25. Why is the model a lazy singleton? Explain the lock.

Loading the model takes several seconds and about 100 MB of RAM, so doing it per request would be
catastrophic. It's loaded once and reused.

**Lazy** means it loads on the first *match* request, not at import time — so `/health` stays
instant and the fast test suite never pays the cost.

The **lock with a double-check** handles concurrency: if two requests hit a cold server
simultaneously, both could see `_model is None` and start loading. The lock serialises them, and
the second check *inside* the lock catches the one that arrived second. It's the classic
double-checked locking pattern.

---

### Q26. What did you test, and why those things?

Six test files. The ones worth mentioning:

**Cosine properties as maths** — identical vectors give 1.0, orthogonal give 0.0, opposite give
−1.0, and crucially `cosine(A, 100A) = 1.0`, which pins the length-independence property my whole
metric choice rests on.

**The `Java` vs `JavaScript` false positive** — the classic keyword-matching trap, explicitly
tested.

**The scanned-PDF path** — I generate an image-only PDF in a fixture and assert we raise a clear
OCR error rather than returning an empty string.

**Discrimination** — the single most important behavioural test: the sample resume must score
higher against the AI/ML job than against a pastry-chef job. *A matcher that returns a high score
for everything is useless,* and no unit test of the maths would catch that.

**Score transparency** — I recompute the headline score from the published formula and assert they
match, so the breakdown can never drift from the number shown.

**Baseline isolation** — the optional semantic skill layer must not change the exact-match results
or the score. That's a design guarantee, so it's a test.

Model-dependent tests are marked `slow`, so `pytest -m "not slow"` runs the whole fast suite in
seconds.

---

### Q27. What's the hardest bug or subtlety you hit?

The 256-token truncation, because it fails **silently**. Everything ran fine and returned
plausible-looking scores — but two thirds of every resume was being discarded before the model
ever saw it. Nothing errored. I only caught it by reading the model card's max sequence length and
then checking how many chunks my documents produced.

It's the kind of bug that teaches you to read the model's constraints rather than trusting that
"it returned a number, so it worked".

The second one was the word-boundary problem: `\b` is the textbook answer for whole-word matching,
and it quietly fails on `C++`, `C#` and `CI/CD` because those end in non-word characters.

---

## Part E — Evaluation & error analysis

### Q28. How would you evaluate this system properly?

First, I'd be clear that **I currently can't**, and that's why I quote no accuracy figure —
quoting one would be fabricating it.

Semantic similarity is unsupervised; there's no correct answer to compare against. So step one is
**creating ground truth**: collect several hundred real resume/job pairs and have **multiple
recruiters** label each one. Critically, I'd measure **inter-annotator agreement** first — if
humans disagree with each other, which is common in hiring, no model can be "right".

Then I'd frame it as **ranking**, not classification, because the real task is "show the best
candidates first". Metrics: **NDCG**, **Precision@k**, **Recall@k**, and Spearman correlation
against the human ordering.

I'd deliberately avoid plain accuracy — the classes are heavily imbalanced, since most pairs are
poor fits, so "no match" every time scores well while being useless.

---

### Q29. What would you compare it against?

Baselines, always — a number means nothing alone. I'd benchmark against random ordering as the
floor, then **TF-IDF cosine similarity** and **BM25** as the classic keyword baselines, then
exact skill overlap on its own, then a larger embedding model, then a cross-encoder as the
realistic ceiling.

The TF-IDF comparison is the one that matters most: **if my transformer can't beat TF-IDF, the
added complexity isn't earning its keep.** That's the question I'd want answered first.

---

### Q30. What are false positives and false negatives here, and which is worse?

A **false positive** is scoring a poor candidate highly — a recruiter wastes time on a bad
interview. Annoying, but recoverable.

A **false negative** is scoring a good candidate poorly — a qualified person is never seen.

**False negatives are far worse**, for two reasons. The cost lands on the candidate, who has no
recourse. And they're **invisible**: you never find out about the good person you filtered out, so
the error never shows up in your feedback loop.

That asymmetry should drive the threshold — lean toward **recall**, surface more candidates than
you need, and let humans do the filtering.

---

### Q31. How would you test robustness, not just accuracy?

Four kinds of test.

**Perturbation** — reword a resume without changing its meaning. The score should barely move. If
it swings, the system is measuring style, not substance.

**Adversarial** — paste the job description into the resume. It should ideally *not* score 100%.
In my system it does, which is a real, demonstrable weakness I'd rather name than hide.

**Format** — the same resume as one-column versus two-column PDF should score similarly. Right now
it might not, because multi-column extraction can scramble reading order.

**Bias** — swap names of different apparent gender or ethnicity on otherwise identical resumes.
Any score change is **measurable** bias. That one I'd run on every model change.

---

### Q32. Your sample resume scores 72% against the sample job. Is that good?

It's a plausible number for a genuinely well-matched pair, but **I wouldn't present it as a
performance metric**, and I'd correct anyone who read it that way.

It's one output on one hand-written pair I created myself, with no ground truth behind it. The
honest statement is "this illustrates the output format". Calling it "72% accurate" would be
exactly the kind of fabricated metric I've tried to keep out of this project.

What the number *is* useful for is comparison: the same resume against a pastry-chef posting
scores far lower, which is the behaviour I actually assert in the test suite.

---

## Part F — Limitations, ethics, scale

### Q33. What are the limitations of your skill extraction?

Five, and I'd list them unprompted.

**Closed vocabulary** — only the ~50 listed skills exist to the system. Rust, Svelte and Julia are
invisible, so "missing skills" can be wrong in both directions.

**No negation handling** — "no experience with Java" counts as a Java mention.

**No proficiency or recency** — a one-weekend tutorial and three years of production work both
count as one mention.

**No context** — "interested in learning Docker" counts as Docker.

**Manual aliases** — every new synonym is a hand edit.

That's exactly why the UI says "**Potentially** missing skills", never "missing skills".

---

### Q34. How would you improve skill extraction?

Four steps, roughly in order of value.

**A real ontology** — ESCO or O*NET instead of my 50 hand-written entries. Thousands of skills
with genuine synonym and parent/child relationships.

**Named Entity Recognition** — a fine-tuned model to extract skills, employers, degrees and dates
from context, finding what no dictionary contains.

**Negation and context detection** — a small rule layer checking whether "no", "without" or
"learning" appears just before a skill.

**Experience-level extraction** — parse "3 years of Python" and compare against the requirement,
instead of treating every mention as equal.

---

### Q35. How would you handle different resume formats — Word, two-column layouts?

For **DOCX**, `python-docx` is straightforward, and I'd add it behind the same service interface,
so nothing downstream changes. That's the benefit of having extraction isolated in one module.

**Two-column layouts** are the harder problem and a genuine current weakness. Extractors read
roughly left-to-right, top-to-bottom, so a skills sidebar can interleave with the experience
column into scrambled text. The embedding model still sees roughly the right *words*, so
similarity degrades gracefully rather than collapsing — but the extracted text can read as
nonsense.

The fix is layout-aware parsing: PyMuPDF's block-level API with column detection, or a tool like
`layoutparser` that segments the page geometrically before extracting.

For **scanned resumes**, Tesseract OCR — with the caveat that OCR introduces its own errors.

---

### Q36. How would you handle multilingual resumes?

Currently it's English-only, because the model is. A French resume against an English job
description would score poorly regardless of how well they actually match.

The direct fix is a **multilingual model** — `paraphrase-multilingual-MiniLM-L12-v2` covers 50+
languages and, importantly, maps them into a **shared vector space**. That means a French resume
and an English job description can be compared *directly*, without translating either one. Same
architecture, same code, one config change.

The skill dictionary would need per-language aliases too. And I'd add language detection so the
system can at least warn when it's out of its depth.

The translation alternative — translate everything to English first — adds a failure mode and a
privacy problem if the translation service is hosted, so I'd prefer the multilingual model.

---

### Q37. Where could bias enter this system, and how would you prevent it?

Several places, and I'd be upfront that **I can't fully prevent it.**

**The pretrained model** carries associations from its web training data, including gendered and
cultural ones. Nobody put them there deliberately; nobody removed them either.

**The skill dictionary** is weighted toward Western, English-language tech stacks. Its blind spots
become the candidate's penalty.

**Writing style, not ability** — the deepest one. The system rewards confident, jargon-dense
industry English, which systematically favours native speakers, people who had resume coaching,
and people already inside the industry. **None of that correlates with ability to do the job.**

**Formatting** — ATS-friendly templates extract cleanly; creative layouts don't. That's penalising
design taste.

**Automation bias** — even labelled a prototype, a number on a screen carries authority people
defer to.

What I do: never auto-reject, always show the reasoning and the evidence so a human can disagree,
keep the dictionary as a visible editable file, and parse **no demographic fields at all** — no
name, gender, age or nationality is ever used.

What a real system would need: **name-swap bias audits** on every model change, human-in-the-loop
by design, anonymisation before scoring, adverse-impact monitoring across groups, and a right to
explanation — the EU AI Act classifies employment AI as **high-risk**.

The bottom line I'd want to land: **a biased human affects the resumes they personally read; a
biased algorithm applies the same bias to every applicant, consistently and invisibly.
Consistency is not fairness.**

---

### Q38. How do you protect user data?

Resumes are dense personal data — name, email, phone, employment history — so the design principle
was to hold them as briefly as possible.

**Nothing is stored.** PDFs are read into memory, converted, used for one request and dropped.
`del pdf_bytes` runs as soon as extraction finishes. Nothing touches disk.

**Nothing is logged.** Every log line records lengths and counts only — you can check
`backend/main.py`, it's `len(...)` everywhere, never the string itself.

**Nothing leaves the machine.** The model runs locally; there's no external API in the request
path. That's a real privacy argument for choosing a small local model, not just a cost argument.

**No database, no accounts, no cookies** — there's nothing to breach.

**And defence in depth in `.gitignore`**: `*_resume.pdf`, `uploads/`, `*.docx` and `.env` are
blocked so a real resume can't be committed by accident. The sample resume in the repo is
fictional.

For a public deployment I'd add HTTPS, a consent notice, rate limiting, a documented GDPR lawful
basis, and a DPIA. Data-subject deletion requests are trivially satisfied — **I store nothing, so
there's nothing to delete.**

---

### Q39. How would you scale this to a million resumes?

The key observation is that **embedding is the expensive step** — about 50 ms per document — while
cosine similarity is nearly free. So the rule is: never re-embed anything.

**Cache embeddings.** Embed each resume once, store the vector, reuse it forever. A resume's
vector doesn't change unless the resume does.

**Use a vector database** — FAISS, Qdrant or pgvector. They use Approximate Nearest Neighbour
indexes like HNSW, so finding the closest vectors among a million takes a few thousand
comparisons instead of a million. The trade-off is in the name: *approximate*. You'll occasionally
miss a true nearest neighbour, which is almost always acceptable.

**Batch and use a GPU.** `encode()` takes a list; 64 documents in one batch is one big matrix
multiplication, far faster than 64 calls.

**Scale the service horizontally** — my FastAPI app is stateless, so multiple workers behind a
load balancer work with no code changes. I'd separate the model server onto GPU machines and keep
API logic on cheap CPU ones, and push bulk jobs onto an async queue so HTTP requests never time
out.

**And architecturally**, the two-stage pattern: bi-encoder plus ANN to shortlist the top 100 from
a million, then a cross-encoder to re-rank those 100. Fast where volume is high, accurate where
precision matters.

---

### Q40. If I gave you two more weeks, what would you build?

**Week one — make it trustworthy.** An evaluation dataset: a few hundred resume/job pairs labelled
by multiple people, with inter-annotator agreement measured. Right now every improvement I could
make is guesswork, because I have no way to tell whether it helped. That unlocks everything else.

Alongside it, the **bias audit pipeline** — automated name-swap tests, run on every model change.

**Week two — the highest-value model work**, now that I can measure it. Cross-encoder re-ranking,
because that's the biggest accuracy jump available. Section-aware parsing, so the Experience
section outweighs the Interests line and the dilution problem goes away. And a real skill ontology
to replace my 50 hand-written entries.

I'd deliberately **not** add an LLM explanation layer first, even though it demos well. It
introduces hallucination risk and, if hosted, breaks the privacy guarantee that resume text never
leaves the machine.

---

## 🎤 60-SECOND PROJECT EXPLANATION

> I built an AI resume–job matching system. You upload a resume PDF and paste a job description,
> and it tells you how well they align and which skills overlap.
>
> The core idea is **semantic similarity**. Instead of counting keywords — which breaks the moment
> someone writes "ML modelling" where the job says "machine learning" — I convert both documents
> into **embeddings**: 384-number vectors that represent meaning, produced by a
> sentence-transformer model. Then I measure the **cosine similarity** between them, which is the
> angle between the two vectors. Similar meaning means a small angle, which means a high score.
>
> Alongside that I run a transparent dictionary-based skill matcher, so the user sees not just a
> number but exactly which skills matched and which are missing. The final score blends the two,
> and the UI always shows the formula that produced it.
>
> It's a FastAPI backend with a Streamlit frontend, fully tested, with resumes processed entirely
> in memory — nothing is stored or logged.
>
> The one thing I'm careful about: **this is a text-similarity score, not a hiring
> recommendation.** I never trained it on hiring outcomes, so it can't predict them — and I made
> sure the system says so everywhere it shows a number.

---

## 🎤 2-MINUTE TECHNICAL EXPLANATION

> The problem is that comparing resumes to job descriptions by keyword matching fails constantly,
> because people describe the same skill in different words. "Machine learning" versus "ML
> modelling", "REST API development" versus "built endpoints with FastAPI" — zero keyword overlap,
> same meaning.
>
> So I built a semantic matching pipeline. **Text, to embeddings, to cosine similarity, to a
> score.**
>
> **Extraction first.** PyMuPDF pulls text from the PDF, with pypdf as a fallback. The case I
> handle carefully is the scanned resume — it's a photograph of text, so there's no text layer and
> extraction returns nothing. I detect that and raise a clear error rather than silently matching
> an empty document.
>
> **Then preprocessing** — and the interesting part is what I *don't* do. Classic NLP strips
> stop-words and lower-cases everything, but transformers are trained on normal sentences where
> those carry meaning. Remove "no" from "no experience with Java" and the sentence inverts. So I
> only fix PDF artefacts: rejoining hyphenated line breaks, stripping bullet glyphs, collapsing
> whitespace.
>
> **Then embeddings.** I use `all-MiniLM-L6-v2` — a distilled BERT, six layers, 384-dimensional
> output, 80 MB, runs on CPU. One constraint drove a real design decision: it **truncates at 256
> word-pieces**, about 180 words. A 600-word resume would have two-thirds silently discarded. So I
> split each document into overlapping 180-word windows, embed each, and average the vectors into
> one document vector. The 30-word overlap stops sentences being cut in half at every boundary.
>
> **Then cosine similarity** — the angle between the two document vectors. I use cosine rather
> than Euclidean distance because it's length-independent: a 600-word resume and a 250-word job ad
> describing the same profile point the same *direction*, even though one vector comes from far
> more text. I'll add the honest caveat that since my vectors are already normalised, cosine and
> Euclidean would rank pairs identically here — the choice is about interpretability, and the
> length argument is decisive in the general unnormalised case.
>
> **In parallel** I run a dictionary skill matcher — about 50 skills with aliases, whole-word
> regex. Deliberately *not* machine learning, because a recruiter can audit "we found the string
> `sklearn`" but can't act on "the model gave it 0.63". Knowing when not to use ML is part of the
> design.
>
> The final score is `0.7 × semantic + 0.3 × skill coverage`, and the UI always shows that
> formula — the number is never a black box. Those weights are a design choice, not learned
> parameters.
>
> **Architecturally** it's FastAPI with Pydantic validation and auto-generated docs, Streamlit
> calling it over HTTP, and a service layer that imports no web framework at all — so the logic is
> reusable and testable without a server. The model is a lazy thread-safe singleton, because
> loading it takes seconds and 100 MB.
>
> **The limitations I'd name upfront**: no OCR, closed skill vocabulary, no negation handling,
> English-only, and no accuracy figure — because I have no labelled evaluation data, and quoting
> one would be fabricating it. To evaluate it properly I'd need several hundred recruiter-labelled
> pairs, and I'd measure NDCG and Precision@k against TF-IDF and BM25 baselines.
>
> **And the thing I'm most careful about**: it measures similarity between two documents, not
> whether someone can do a job. Someone who copy-pastes the job ad scores near 100%. That's not a
> bug — it's what the metric fundamentally is, and the system says so on every result.

---

## Rapid-fire one-liners

| Question | Answer |
|---|---|
| What's an embedding? | A list of numbers representing meaning — 384 of them here. |
| Why 384? | It's the output dimension of `all-MiniLM-L6-v2`. |
| What's cosine similarity? | The angle between two vectors: `(A·B)/(‖A‖‖B‖)`. |
| Range? | −1 to 1; in practice 0 to 1 for this model. |
| Why cosine over Euclidean? | It ignores document length — only direction matters. |
| Model size? | ~80 MB, ~22M parameters, 6 layers. |
| Max input? | 256 word-pieces — which is why I chunk. |
| Why chunk? | Otherwise most of a resume is silently truncated away. |
| Why overlap chunks? | So a sentence on a boundary isn't cut in half everywhere. |
| Score formula? | `0.7 × semantic + 0.3 × skill coverage`. |
| Are the weights learned? | No — a design choice. There's no labelled data here. |
| Is it a hiring probability? | **No.** Never trained on hiring outcomes. |
| Is skill extraction ML? | No — dictionary lookup, chosen for explainability. |
| Biggest limitation? | Text similarity isn't job suitability. |
| Worst error type? | False negatives — a good candidate is invisibly filtered out. |
| Where's data stored? | Nowhere. Memory only, never logged, never leaves the machine. |
| Accuracy? | **Unmeasured** — no labelled dataset, so no figure claimed. |
| Biggest next improvement? | An evaluation dataset; then cross-encoder re-ranking. |

---

## Questions to ask them

Having a question ready shows engagement. Some that fit naturally after this project:

1. "How does your team handle evaluation when there's no clean ground truth?"
2. "Do you use bi-encoders, cross-encoders, or both in production?"
3. "How do you decide when a problem needs ML versus a well-built rule-based system?"
4. "What does your team do about bias auditing for models that affect people?"
5. "How do you handle the privacy side when models process personal data?"

---

**Good luck. You built it, you understand it, and you know where it breaks — which is more than
most candidates can say about their portfolio projects.**
