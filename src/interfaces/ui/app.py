"""
Streamlit Web UI for Cambodian Civil & Commercial Law RAG.

Powered by DeepSeek Flash (v4 / deepseek-chat) with statutory article citations.
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
    page_title="RAG Cambodia Law (DeepSeek Flash)",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("⚖️ Cambodian Civil & Commercial Law Assistant")
st.caption(
    "Ground truth legal retrieval and statutory question-answering powered by **DeepSeek Flash**."
)

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    st.markdown("⚡ **LLM Engine**: `DeepSeek Flash (v4)`")
    st.caption("Model: `deepseek-chat` via DeepSeek API")

    # API Key Input
    api_key_input = st.text_input(
        "DeepSeek API Key:",
        type="password",
        placeholder="sk-... (or set in Secrets)",
        help="Enter your API key from platform.deepseek.com",
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
        - **Retrieval Engine**: BM25 + pgvector Hybrid
        - **LLM Engine**: DeepSeek Flash (v4)
        """
    )
    st.info(
        "💡 **Tip**: Ask questions in English or Khmer (ភាសាខ្មែរ). Every statement is grounded in official statutes."
    )

# Tabs
tab_qa, tab_search = st.tabs(["💬 Legal Q&A Assistant", "🔎 Statutory Article Explorer"])

# Tab 1: Legal Q&A
with tab_qa:
    st.subheader("Ask a Legal Question")
    example_questions = [
        "តើកិច្ចសន្យាបង្កើតឡើងដោយរបៀបណា យោងតាមក្រមរដ្ឋប្បវេណី?",
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
        "Your question (English / ខ្មែរ):",
        value=selected_example if selected_example != "-- Select an example --" else "",
        placeholder="e.g. តើកិច្ចសន្យាបង្កើតឡើងដោយរបៀបណា? or What are the requirements for contract formation?",
    )

    if st.button("🚀 Analyze with DeepSeek Flash", type="primary"):
        if not user_query.strip():
            st.warning("Please enter a question to analyze.")
        else:
            with st.spinner("Analyzing with DeepSeek Flash..."):
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
                        qa_use_case._llm._client = OpenAI(
                            api_key=clean_key,
                            base_url="https://api.deepseek.com",
                        )
                        qa_use_case._llm._api_key = clean_key

                    req = LegalQARequest(
                        question=user_query,
                        top_k=top_k,
                        law_filter=law_filter,
                        model="deepseek-chat",
                    )
                    response = qa_use_case.execute(req)

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
                            1. Go to [platform.deepseek.com/api_keys](https://platform.deepseek.com/api_keys) and generate an API key.
                            2. Ensure your DeepSeek account has positive balance / credits.
                            3. Paste the key in the **DeepSeek API Key** box on the sidebar.
                            """
                        )

                    st.markdown("---")
                    st.markdown("### 📚 Retrieved Statutory Context Articles:")
                    qa_use_case = get_qa_use_case()
                    search_q = qa_use_case._prepare_search_query(user_query)
                    retriever = get_hybrid_retriever()
                    docs = retriever.execute(
                        RetrievalRequest(query=search_q, top_k=top_k, law_filter=law_filter)
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
        "Search keywords, concepts, or article numbers (English / ខ្មែរ):",
        placeholder="e.g. good faith, កិច្ចសន្យា, Article 336",
    )

    if search_term.strip():
        qa_use_case = get_qa_use_case()
        mapped_search = qa_use_case._prepare_search_query(search_term)
        retriever = get_hybrid_retriever()
        results = retriever.execute(
            RetrievalRequest(query=mapped_search, top_k=10, law_filter=law_filter)
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
