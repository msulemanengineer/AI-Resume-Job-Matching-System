"""
Streamlit UI for the AI Resume-Job Matching System.

This file contains NO AI logic at all. It collects input, calls the FastAPI
backend over HTTP, and renders the report. Keeping the intelligence on the
server side is what makes the backend independently testable and reusable -
a point worth making when demoing the project.

Run with:  streamlit run frontend/app.py
"""

from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
REQUEST_TIMEOUT = 180  # the first request also loads the model (~5-10 s)


def _parse_version(raw: str) -> tuple:
    parts = []
    for piece in raw.split(".")[:2]:
        digits = "".join(c for c in piece if c.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


# Streamlit 1.49 replaced `use_container_width=True` with `width="stretch"`. This project
# supports both, so pick whichever the installed version understands instead of emitting
# deprecation warnings (or crashing) on one side of that boundary.
STRETCH = (
    {"width": "stretch"}
    if _parse_version(st.__version__) >= (1, 49)
    else {"use_container_width": True}
)

st.set_page_config(
    page_title="AI Resume-Job Matcher",
    page_icon="🎯",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Backend helpers
# ---------------------------------------------------------------------------


def backend_status():
    """Check whether the FastAPI backend is reachable."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        response.raise_for_status()
        return True, response.json()
    except requests.RequestException:
        return False, None


def call_match(resume_text: str, job_text: str, semantic_skills: bool) -> dict:
    response = requests.post(
        f"{API_BASE_URL}/match",
        json={
            "resume_text": resume_text,
            "job_description": job_text,
            "use_semantic_skills": semantic_skills,
        },
        timeout=REQUEST_TIMEOUT,
    )
    if response.status_code >= 400:
        raise RuntimeError(_error_detail(response))
    return response.json()


def call_extract_text(uploaded_file) -> dict:
    response = requests.post(
        f"{API_BASE_URL}/extract-text",
        files={
            "file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")
        },
        timeout=REQUEST_TIMEOUT,
    )
    if response.status_code >= 400:
        raise RuntimeError(_error_detail(response))
    return response.json()


def _error_detail(response: requests.Response) -> str:
    """Turn a FastAPI error response into a readable message."""
    try:
        detail = response.json().get("detail")
        if isinstance(detail, list) and detail:  # pydantic validation errors
            return "; ".join(item.get("msg", str(item)) for item in detail)
        return str(detail or response.text)
    except ValueError:
        return response.text


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Status")
    online, health = backend_status()
    if online:
        st.success("Backend: online")
        st.caption(f"Model: `{health['embedding_model']}`")
        st.caption(
            "Model loaded in memory: "
            + ("yes" if health["model_loaded"] else "not yet (loads on first match)")
        )
        st.caption(f"Skill dictionary: {health['skills_in_dictionary']} skills")
    else:
        st.error("Backend: offline")
        st.caption(f"Expected at `{API_BASE_URL}`")
        st.code("uvicorn backend.main:app --reload", language="bash")

    st.divider()
    st.header("🔧 Options")
    use_semantic_skills = st.toggle(
        "Semantic skill matching",
        value=True,
        help=(
            "Also check whether 'missing' skills appear in the resume in different "
            "words. Exact dictionary matching always runs as the baseline."
        ),
    )

    st.divider()
    st.caption(
        "🔒 **Privacy**: resumes are processed in memory and never saved to disk "
        "or logged. Close the app and the data is gone."
    )


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("🎯 AI Resume-Job Matching System")
st.markdown(
    "Compares a resume with a job description using **sentence embeddings** and "
    "**cosine similarity**, then lists which skills overlap."
)
st.info(
    "**Educational prototype.** The score measures how similar two pieces of text "
    "are - it is *not* a hiring recommendation and *not* a probability of getting "
    "the job.",
    icon="ℹ️",
)

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

left, right = st.columns(2)

with left:
    st.subheader("1️⃣ Resume")
    input_mode = st.radio(
        "Input method",
        ["Upload PDF", "Paste text"],
        horizontal=True,
        label_visibility="collapsed",
    )

    resume_text = ""
    if input_mode == "Upload PDF":
        uploaded = st.file_uploader("Upload your resume (text-based PDF)", type=["pdf"])
        st.caption(
            "⚠️ Scanned / image-only PDFs cannot be read - this system has no OCR."
        )
        if uploaded is not None:
            try:
                with st.spinner("Extracting text..."):
                    extracted = call_extract_text(uploaded)
                resume_text = extracted["text"]
                st.success(
                    f"Extracted {extracted['char_count']} characters "
                    f"from {extracted['page_count']} page(s)."
                )
                if extracted.get("warning"):
                    st.warning(extracted["warning"])
                with st.expander("Preview extracted text"):
                    st.text(resume_text[:3000])
            except RuntimeError as exc:
                st.error(f"Could not read the PDF: {exc}")
            except requests.RequestException:
                st.error("Backend not reachable. Start the API first.")
    else:
        resume_text = st.text_area(
            "Paste your resume text", height=280, placeholder="Paste resume text here..."
        )

with right:
    st.subheader("2️⃣ Job Description")
    job_text = st.text_area(
        "Paste the job description",
        height=340,
        placeholder="Paste the full job description here...",
    )

st.session_state.setdefault("result", None)

analyze = st.button("🔍 Analyze Match", type="primary", **STRETCH)

if analyze:
    if len(resume_text.strip()) < 20:
        st.error("Please provide a resume (at least 20 characters of text).")
    elif len(job_text.strip()) < 20:
        st.error("Please provide a job description (at least 20 characters).")
    else:
        try:
            with st.spinner("Embedding both documents and computing similarity..."):
                st.session_state.result = call_match(
                    resume_text, job_text, use_semantic_skills
                )
        except RuntimeError as exc:
            st.error(f"Matching failed: {exc}")
            st.session_state.result = None
        except requests.RequestException:
            st.error(f"Backend not reachable at {API_BASE_URL}. Is uvicorn running?")
            st.session_state.result = None


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

result = st.session_state.result
if result:
    st.divider()
    st.header("📊 Matching Report")

    breakdown = result["score_breakdown"]
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Overall Match Score",
        f"{result['overall_match_score']}%",
        help="Weighted blend of semantic similarity and skill coverage.",
    )
    c2.metric(
        "Semantic Similarity",
        f"{result['semantic_similarity_score']}%",
        help="Pure cosine similarity between the two document embeddings.",
    )
    c3.metric(
        "Skill Coverage",
        f"{breakdown['skill_coverage_percent']}%",
        help="Share of job-description skills detected in the resume.",
    )

    st.progress(min(result["overall_match_score"] / 100, 1.0))
    st.caption(f"**Interpretation:** {result['match_band']} (heuristic label)")

    with st.expander("🧮 How this score was calculated"):
        st.code(breakdown["formula"], language="text")
        st.markdown(
            f"""
- **Semantic similarity ({breakdown['semantic_similarity_percent']}%)** - the resume
  and the job description were each split into overlapping ~180-word chunks, every
  chunk was turned into a 384-dimensional vector by `all-MiniLM-L6-v2`, the chunk
  vectors were averaged into one document vector, and the **cosine of the angle**
  between the two document vectors was computed.
- **Skill coverage ({breakdown['skill_coverage_percent']}%)** - of the
  {len(result['job_skills'])} skills detected in the job description,
  {len(result['matching_skills'])} were also found in the resume.
- **Weights** - {breakdown['semantic_weight']:g} / {breakdown['skill_weight']:g}.
  These are a design choice made for readability, **not** a value learned from
  hiring data.
"""
        )

    st.subheader("🧩 Skills")
    col_match, col_missing = st.columns(2)

    with col_match:
        st.markdown("#### ✅ Matching skills")
        if result["matching_skills"]:
            st.dataframe(
                pd.DataFrame({"Skill": result["matching_skills"]}),
                hide_index=True,
                **STRETCH,
            )
        else:
            st.info("No dictionary skills from the job description were found.")

    with col_missing:
        st.markdown("#### ⚠️ Potentially missing skills")
        if result["missing_skills"]:
            st.dataframe(
                pd.DataFrame({"Skill": result["missing_skills"]}),
                hide_index=True,
                **STRETCH,
            )
            st.caption(
                "'Potentially' - the skill may be present but described in words the "
                "dictionary does not know."
            )
        else:
            st.success("Every job skill detected was also found in the resume.")

    if result.get("semantically_related_skills"):
        st.markdown("#### 🔎 Missing skills that *look* semantically related")
        st.caption(
            "These were not found literally, but a passage of the resume is close to "
            "them in embedding space. Treat as a hint, not proof."
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Skill": item["skill"],
                        "Similarity": round(item["similarity"], 3),
                        "Closest resume passage": item["evidence"],
                    }
                    for item in result["semantically_related_skills"]
                ]
            ),
            hide_index=True,
            **STRETCH,
        )

    col_r, col_j = st.columns(2)
    with col_r:
        with st.expander(f"📄 All resume skills ({len(result['resume_skills'])})"):
            st.write(", ".join(result["resume_skills"]) or "- none detected -")
    with col_j:
        with st.expander(
            f"📋 All job description skills ({len(result['job_skills'])})"
        ):
            st.write(", ".join(result["job_skills"]) or "- none detected -")

    if result.get("extra_resume_skills"):
        with st.expander(
            f"➕ Skills in your resume the job did not ask for "
            f"({len(result['extra_resume_skills'])})"
        ):
            st.write(", ".join(result["extra_resume_skills"]))

    st.divider()
    st.warning(result["disclaimer"], icon="⚠️")
