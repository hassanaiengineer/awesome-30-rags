import streamlit as st
import sys
import os
import json
import pandas as pd

from branding import apply_streamlit_branding

# Add package root (17_agentic_rag_math_agent/) to import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.benchmark import benchmark_math_agent  # Add this import
from data.load_gsm8k_data import load_jeebench_dataset
from rag.query_router import answer_math_question

apply_streamlit_branding(
    st,
    title="Math Tutor Agent Dashboard",
    page_icon="🧮",
    caption="Hassan Khan RAG Series • Agentic math Q&A + feedback + benchmarking",
)

tab1, tab2, tab3 = st.tabs(["📘 Ask a Question", "📁 View Feedback", "📊 Benchmark Results"])

# ---------------- TAB 1: Ask a Question ---------------- #
with tab1:
    st.subheader("📘 Ask a Math Question")
    st.markdown("Enter any math question below. The agent will try to explain it step-by-step.")

    if "last_question" not in st.session_state:
        st.session_state["last_question"] = ""
    if "last_answer" not in st.session_state:
        st.session_state["last_answer"] = ""
    if "feedback_given" not in st.session_state:
        st.session_state["feedback_given"] = False

    user_question = st.text_input("Your Question:")

    if st.button("Get Answer"):
        if user_question:
            with st.spinner("Thinking..."):
                answer = answer_math_question(user_question)
            st.session_state["last_question"] = user_question
            st.session_state["last_answer"] = answer
            st.session_state["feedback_given"] = False

    if st.session_state["last_answer"]:
        st.markdown("### ✅ Answer:")
        st.success(st.session_state["last_answer"])

        if not st.session_state["feedback_given"]:
            st.markdown("### 🙋 Was this helpful?")
            col1, col2 = st.columns(2)

            with col1:
                if st.button("👍 Yes"):
                    feedback = "positive"
                    st.session_state["feedback_given"] = True
            with col2:
                if st.button("👎 No"):
                    feedback = "negative"
                    st.session_state["feedback_given"] = True

            if st.session_state["feedback_given"]:
                log_entry = {
                    "question": st.session_state["last_question"],
                    "answer": st.session_state["last_answer"],
                    "feedback": feedback
                }

                try:
                    os.makedirs("logs", exist_ok=True)
                    log_file = "logs/feedback_log.json"

                    if os.path.exists(log_file):
                        with open(log_file, "r") as f:
                            existing_logs = json.load(f)
                    else:
                        existing_logs = []

                    existing_logs.append(log_entry)

                    with open(log_file, "w") as f:
                        json.dump(existing_logs, f, indent=2)

                    st.success(f"✅ Feedback recorded as '{feedback}'")
                    st.write("📝 Log entry:", log_entry)
                except Exception as e:
                    st.error(f"⚠️ Error saving feedback: {e}")

# ---------------- TAB 2: View Feedback ---------------- #
with tab2:
    st.subheader("📁 View Collected Feedback")
    try:
        with open("logs/feedback_log.json", "r") as f:
            feedback_logs = json.load(f)
        st.success("Loaded feedback log.")
        st.dataframe(pd.DataFrame(feedback_logs))
    except Exception as e:
        st.warning("No feedback log found or error loading.")
        st.text(str(e))

# ---------------- TAB 3: Benchmark Results ---------------- #

with tab3:
    st.subheader("📊 Benchmark Accuracy Report")

    total_math = len(load_jeebench_dataset())

    st.caption(f"📘 Benchmarking from {total_math} math questions")

    num_questions = st.slider("Select number of math questions to benchmark", min_value=3, max_value=total_math, value=10)

    if st.button("▶️ Run Benchmark Now"):
        with st.spinner(f"Benchmarking {num_questions} math questions..."):
            df_result, accuracy = benchmark_math_agent(limit=num_questions)

            # Save the result
            os.makedirs("benchmark", exist_ok=True)
            result_path = f"benchmark/results_math_{num_questions}.csv"
            df_result.to_csv(result_path, index=False)

            # Show result
            st.success(f"✅ Done! Accuracy: {accuracy:.2f}%")
            st.metric("Accuracy", f"{accuracy:.2f}%")
            st.dataframe(df_result)
            st.download_button("Download Results", data=df_result.to_csv(index=False), file_name=result_path, mime="text/csv")
