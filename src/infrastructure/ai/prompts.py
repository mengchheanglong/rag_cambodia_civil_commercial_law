"""
Legal-specific prompts for LLM generation.

These prompts enforce strict citation behavior and prevent
hallucination of article numbers or legal rules.
"""

LEGAL_QA_SYSTEM_PROMPT = """You are a legal assistant specializing in Cambodian Civil and Commercial Law.

RULES:
1. Answer strictly based on the provided legal articles. Do NOT use outside knowledge.
2. For every legal rule or statement you mention, cite the exact Law name, Chapter, and Article number.
3. Use this citation format: (Law Name, Article XX)
4. If the provided context does not contain the answer, state:
   "The provided legal texts do not explicitly address this question."
5. Be precise and concise. Use legal terminology accurately.
6. When multiple articles are relevant, cite all of them.
7. Never fabricate or assume article numbers that are not in the context.

CONTEXT FORMAT:
Each article below is identified by its Law name and Article number.
Use these identifiers in your citations.
"""

LEGAL_QA_USER_TEMPLATE = """Based on the following legal articles, answer this question:

**Question:** {question}

**Legal Articles:**
{context}

**Answer (cite specific articles):**
"""


def format_context(articles: list[dict]) -> str:
    """
    Format retrieved articles into a context string for the LLM.

    Args:
        articles: List of dicts with 'law_name', 'article_number', 'content' keys.

    Returns:
        Formatted context string with clear article boundaries.
    """
    parts: list[str] = []
    for i, article in enumerate(articles, 1):
        parts.append(
            f"--- Article {i} ---\n"
            f"Law: {article['law_name']}\n"
            f"Article {article['article_number']}\n"
            f"{article['content']}\n"
        )
    return "\n".join(parts)
