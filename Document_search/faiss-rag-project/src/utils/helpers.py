def format_docs(docs) -> str:
    """Concatenate document page contents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)