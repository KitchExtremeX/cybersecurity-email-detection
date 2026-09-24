"""Cybersecurity Email Detection — spam versus ham triage UI."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from src.predict.mbox_io import classify_mbox
from src.predict.service import load_classifier

HAM_EXAMPLE = """Subject: Design review Thursday

Hi team,

Can we keep Thursday at 15:00 for the API design review? I will bring the open questions from last week's sync. The notes are already on the shared drive, so there is nothing to install and no password to send.

Thanks,
Priya Nair
"""

SPAM_EXAMPLE = """Subject: Mailbox locks in 2 hours

Dear customer,

We detected a sign-in to your mailbox from an unrecognized device. Verify your password now or the account will be suspended.

http://mailbox-verify-login.example/secure

This link expires in 2 hours. Failure to confirm will limit outgoing mail.

Account Security Team
"""

CSV_COLUMNS = ["subject", "from", "date", "snippet", "label", "confidence"]


def _load_example(text: str) -> None:
    st.session_state["email_text"] = text


@st.cache_resource
def _classifier():
    return load_classifier(ROOT / "outputs")


def _sidebar(classifier) -> None:
    metadata = classifier.metadata or {}
    st.sidebar.header("Model")
    st.sidebar.write(f"**Loaded:** {classifier.model_name}")
    st.sidebar.caption(str(classifier.output_dir))
    best_f1 = next(
        (
            row["f1_spam"]
            for row in metadata.get("models", [])
            if row["model"] == classifier.model_name
        ),
        None,
    )
    if best_f1 is not None:
        st.sidebar.metric("Holdout spam F1", f"{best_f1:.3f}")
    if metadata.get("runner_up"):
        st.sidebar.caption(
            f"Runner-up {metadata['runner_up']} · margin {metadata.get('f1_margin', 0):.3f}"
        )
    st.sidebar.caption(
        "Winner is the highest spam-class F1. Ties break on macro F1, then accuracy, then model name."
    )
    if st.sidebar.button("Reload artifacts"):
        st.cache_resource.clear()
        st.rerun()
    with st.sidebar.expander("About this demo"):
        st.write(
            "Applied AI / cybersecurity student project by Kadien Coe (KitchExtremeX). "
            "Scores are triage hints trained on a synthetic corpus. "
            "A production filter needs a licensed real-world mail set and a fresh evaluation."
        )


def _single_email(classifier) -> None:
    st.subheader("Single email")
    st.write("Paste a subject and body. Text that starts with `Subject:` matches the training format.")
    c1, c2 = st.columns(2)
    c1.button("Load legitimate example", on_click=_load_example, args=(HAM_EXAMPLE,))
    c2.button("Load phishing example", on_click=_load_example, args=(SPAM_EXAMPLE,))
    if "email_text" not in st.session_state:
        st.session_state["email_text"] = ""
    text = st.text_area("Email text", height=240, key="email_text")
    if not text.strip():
        st.info("Paste an email to classify it.")
        return
    try:
        result = classifier.predict_text(text)
    except ValueError as exc:
        st.warning(str(exc))
        return

    label = result["label"]
    confidence = result["confidence"]
    if label == "spam":
        st.error(f"Spam · confidence {confidence:.1%}")
    else:
        st.success(f"Ham (legitimate) · confidence {confidence:.1%}")

    probabilities = result["probabilities"]
    left, right = st.columns(2)
    left.metric("Ham probability", f"{probabilities.get('ham', 0.0):.1%}")
    right.metric("Spam probability", f"{probabilities.get('spam', 0.0):.1%}")
    st.progress(float(probabilities.get("spam", 0.0)), text="Spam probability")


def _batch(classifier) -> None:
    st.subheader("Batch mbox")
    st.write(
        "Upload a Unix `.mbox` file. Each message is scored and the table can be downloaded as CSV. "
        "A synthetic sample lives at `data/sample_inbox.mbox`."
    )
    uploaded = st.file_uploader("Mailbox", type=["mbox"])
    if uploaded is None:
        st.info("Upload an .mbox file to classify every message.")
        return

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mbox", delete=False) as handle:
            handle.write(uploaded.getvalue())
            temp_path = Path(handle.name)
        with st.spinner("Classifying messages..."):
            rows = classify_mbox(classifier, temp_path)
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        return
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

    frame = pd.DataFrame(rows, columns=CSV_COLUMNS)
    spam_count = int((frame["label"] == "spam").sum())
    ham_count = int((frame["label"] == "ham").sum())
    m1, m2, m3 = st.columns(3)
    m1.metric("Messages", len(frame))
    m2.metric("Spam", spam_count)
    m3.metric("Ham", ham_count)
    st.dataframe(frame, use_container_width=True, hide_index=True)
    st.download_button(
        "Download CSV",
        data=frame.to_csv(index=False).encode("utf-8"),
        file_name="spam_ham_classifications.csv",
        mime="text/csv",
    )


def _comparison(classifier) -> None:
    models = (classifier.metadata or {}).get("models") or []
    if not models:
        return
    st.subheader("Model comparison")
    st.write(
        "Holdout precision, recall, and F1 use spam as the positive class. "
        "The saved artifact is the best spam F1 from the last training run."
    )
    frame = pd.DataFrame(models)
    show = [
        "model",
        "accuracy",
        "precision_spam",
        "recall_spam",
        "f1_spam",
        "f1_macro",
    ]
    st.dataframe(frame[show], use_container_width=True, hide_index=True)
    chart = frame.set_index("model")[["f1_spam", "accuracy"]]
    st.bar_chart(chart)


def main() -> None:
    st.set_page_config(
        page_title="Cybersecurity Email Detection",
        page_icon="🛡️",
        layout="wide",
    )
    st.title("Cybersecurity Email Detection")
    st.caption("Spam vs Ham · security triage classifier")

    try:
        classifier = _classifier()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.write("From the repo root, after installing requirements:")
        st.code("python -m src.train", language="bash")
        return

    _sidebar(classifier)
    single, batch = st.tabs(["Single email", "Batch mbox"])
    with single:
        _single_email(classifier)
    with batch:
        _batch(classifier)
    _comparison(classifier)


main()
