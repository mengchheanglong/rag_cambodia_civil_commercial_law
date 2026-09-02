"""
Streamlit Web UI for Cambodian Civil & Commercial Law RAG.

Supports DeepSeek (Flash / Pro) & OpenAI models with grounded article citations.
Launch with:
streamlit run src/interfaces/ui/app.py
"""

import os
import sys
from pathlib import Path

# Ensure repo root is always in sys.path
repo_root = Path(__file__).resolve().parent.parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import streamlit as st

# Sync Streamlit Cloud secrets into environment if present
try:
    if hasattr(st, "secrets"):
        for k, v in st.secrets.items():
            if isinstance(v, str):
                os.environ[k] = v.strip().strip('"').strip("'")
except Exception:
    pass

from src.application.dtos import LegalQARequest, RetrievalRequest
from src.interfaces.api.dependencies import get_hybrid_retriever, get_qa_use_case

# Page configuration
st.set_page_config(
    page_title="RAG Cambodia Law",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("⚖️ Cambodian Civil & Commercial Law Assistant")
st.caption(
    "Ground truth legal retrieval and question-answering based on official Cambodian statutes."
)

# Sidebar filters & metadata
with st.sidebar:
    st.header("⚙️ Model & Search Config")

    # LLM Model Selector (DeepSeek Flash vs Pro vs OpenAI)
    model_options = {
        "⚡ DeepSeek V3 / Flash (Fast & Efficient)": "deepseek-chat",
        "🧠 DeepSeek R1 / Pro (Deep Legal Reasoning)": "deepseek-reasoner",
        "🤖 OpenAI GPT-4o": "gpt-4o",
        "⚡ OpenAI GPT-4o-mini": "gpt-4o-mini",
    }
    selected_model_label = st.selectbox(
        "LLM Generation Model:",
        options=list(model_options.keys()),
        index=0,
    )
    selected_model = model_options[selected_model_label]

    # Optional dynamic API key override
    api_key_input = st.text_input(
        "API Key (DeepSeek / OpenAI):",
        type="password",
        placeholder="sk-... (or set in Streamlit Secrets)",
        help="Paste your DeepSeek API key (or OpenAI key if using GPT-4o).",
    )

    st.divider()
    st.subheader("🔍 Retrieval Filters")
    law_filter_option = st.selectbox(
        "Filter by Statute:",
        ["All Ingested Laws", "Civil Code 2007", "Law on Commercial Arbitration 2006"],
    )
    law_filter = None if law_filter_option == "All Ingested Laws" else law_filter_option

    top_k = st.slider("Top Articles to Retrieve:", min_value=1, max_value=10, value=5)

    st.divider()
    st.subheader("📚 Corpus Overview")
    st.markdown(
        """
        - **Civil Code 2007**: 1,297 Articles
        - **Law on Commercial Arbitration 2006**: 50 Articles
        - **Retrieval Engine**: Hybrid (BM25 + Dense Vectors)
        - **LLM Engine**: DeepSeek / OpenAI
        """
    )
    st.info(
        "💡 **Tip**: Queries can be conceptual (e.g. *'What is a defect in sold goods?'*) or exact (e.g. *'Article 336'*)."
    )

# Tabs
tab_qa, tab_search = st.tabs(["💬 Legal Q&A Assistant", "🔎 Statutory Article Explorer"])

# Tab 1: Legal Q&A
with tab_qa:
    st.subheader("Ask a Legal Question")
    example_questions = [
        "How is a contract formed by offer and acceptance under the Cambodian Civil Code?",
        "What are the formal requirements for an arbitration agreement to be valid?",
        "What is the principle of good faith in Cambodian civil obligations?",
        "What are the remedies when a seller delivers goods with a defect?",
        "Can an arbitral tribunal rule on its own jurisdiction?",
    ]

    selected_example = st.selectbox(
        "Or pick an example question:",
        ["-- Select an example --"] + example_questions,
    )

    user_query = st.text_input(
        "Your question:",
        value=selected_example if selected_example != "-- Select an example --" else "",
        placeholder="e.g., What are the legal requirements for contract formation?",
    )

    if st.button("🚀 Analyze & Answer", type="primary"):
        if not user_query.strip():
            st.warning("Please enter a question to analyze.")
        else:
            with st.spinner(f"Analyzing with {selected_model_label}..."):
                try:
                    qa_use_case = get_qa_use_case()

                    # Apply dynamic or secrets API key if provided
                    clean_key = (
                        api_key_input.strip()
                        or os.environ.get("DEEPSEEK_API_KEY", "")
                        or os.environ.get("OPENAI_API_KEY", "")
                    ).strip().strip('"').strip("'")

                    if clean_key and hasattr(qa_use_case._llm, "_client"):
                        from openai import OpenAI
                        is_openai_key = clean_key.startswith("sk-proj-")
                        if is_openai_key or "gpt-" in selected_model:
                            base_url = None
                        elif "deepseek" in selected_model:
                            base_url = "https://api.deepseek.com"
                        else:
                            base_url = None

                        qa_use_case._llm._client = OpenAI(api_key=clean_key, base_url=base_url)
                        qa_use_case._llm._api_key = clean_key

                    req = LegalQARequest(
                        question=user_query,
                        top_k=top_k,
                        law_filter=law_filter,
                        model=selected_model,
                    )
                    response = qa_use_case.execute(req)

                    # Display Reasoning Chain (if DeepSeek Reasoner)
                    if response.reasoning_content:
                        with st.expander("🧠 Deep Legal Reasoning Process (DeepSeek Reasoner)", expanded=False):
                            st.markdown(response.reasoning_content)

                    # Display Answer
                    st.markdown("### 📝 Legal Analysis")
                    st.markdown(response.answer)

                    # Display Citations
                    if response.citations:
                        st.markdown("#### 📜 Cited Statutory Authorities")
                        for cit in response.citations:
                            status_badge = "✅ Verified" if cit.get("is_verified") else "⚠️ Unverified"
                            st.markdown(
                                f"- **{cit['law_name']} — Article {cit['article_number']}** `[{status_badge}]`"
                            )
                            if cit.get("excerpt"):
                                st.caption(f"> \"{cit['excerpt']}\"")

                    # Display Retrieved Articles
                    with st.expander(f"📚 Retrieved Context Articles ({len(response.source_articles)})"):
                        for i, doc in enumerate(response.source_articles, 1):
                            st.markdown(
                                f"**{i}. {doc['law_name']} — Article {doc['article_number']}**"
                            )
                            if doc.get("chapter"):
                                st.caption(f"Context: {doc['chapter']}")
                            st.text(doc.get("content_preview", ""))
                            st.divider()

                    st.warning(f"⚖️ **Legal Disclaimer**: {response.disclaimer}")

                except Exception as e:
                    error_msg = str(e)
                    st.error(f"Generation error: {error_msg}")

                    if "401" in error_msg or "authentication" in error_msg.lower():
                        st.info(
                            """
                            💡 **How to resolve this 401 Authentication Error**:
                            - If you are using **DeepSeek Flash / Pro**:
                              1. Get a valid key from [platform.deepseek.com/api_keys](https://platform.deepseek.com/api_keys).
                              2. Ensure your account has top-up balance.
                              3. Paste the key in the **API Key** box on the sidebar.
                            - If you are using an **OpenAI key** (starts with `sk-proj-...`):
                              1. In the sidebar dropdown, switch the model to **🤖 OpenAI GPT-4o** or **⚡ OpenAI GPT-4o-mini**.
                            """
                        )

                    st.markdown("---")
                    st.markdown("### 📚 Retrieved Statutory Context Articles:")
                    retriever = get_hybrid_retriever()
                    docs = retriever.execute(
                        RetrievalRequest(query=user_query, top_k=top_k, law_filter=law_filter)
                    )
                    for doc in docs:
                        with st.container(border=True):
                            st.markdown(
                                f"**{doc.chunk.metadata.law_name} — Article {doc.chunk.metadata.article_number}**"
                            )
                            if doc.chunk.metadata.chapter:
                                st.caption(f"Context: {doc.chunk.metadata.chapter}")
                            st.write(doc.chunk.content)

# Tab 2: Statutory Search
with tab_search:
    st.subheader("Explore Statutory Articles")
    search_term = st.text_input(
        "Search keywords, concepts, or article numbers:",
        placeholder="e.g. good faith, arbitration, Article 336",
    )

    if search_term.strip():
        retriever = get_hybrid_retriever()
        results = retriever.execute(
            RetrievalRequest(query=search_term, top_k=10, law_filter=law_filter)
        )

        st.markdown(f"Found **{len(results)}** relevant articles:")
        for doc in results:
            with st.container(border=True):
                st.markdown(
                    f"### {doc.chunk.metadata.law_name} — Article {doc.chunk.metadata.article_number}"
                )
                if doc.chunk.metadata.article_title:
                    st.caption(f"**Title**: {doc.chunk.metadata.article_title}")
                if doc.chunk.metadata.chapter:
                    st.caption(f"**Chapter**: {doc.chunk.metadata.chapter}")
                if doc.chunk.metadata.section:
                    st.caption(f"**Section**: {doc.chunk.metadata.section}")
                st.markdown(f"```plaintext\n{doc.chunk.content}\n```")
