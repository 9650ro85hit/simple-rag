
import streamlit as st
from rag_graph import rag_graph

st.set_page_config(
    page_title="RAG Chat",
    page_icon="💬"
)

st.title("RAG Document Chat")
st.write("Ask a question about your documents.")

query = st.text_input(
    "Enter your question",
    placeholder="What is this document about?"
)

if st.button("Generate Answer", type="primary"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Generating answer..."):
            try:
                result = rag_graph.invoke({"query": query})

                st.subheader("Answer")
                st.write(result.get("answer", "No answer generated."))

                with st.expander("Retrieval details"):
                    st.write(
                        "Rewritten query:",
                        result.get("rewritten_query", query)
                    )
                    st.write(
                        "Classification:",
                        result.get("overall_label", "Not available")
                    )
                    st.write(
                        "Answer source:",
                        result.get("answer_source", "Not available")
                    )

            except Exception as e:
                st.error(f"Something went wrong: {e}")
