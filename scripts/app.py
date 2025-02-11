import streamlit as st
from llm_integration import NexusRAG

def main():
    st.title("NexusRAG: Offline AI Knowledge Engine")
    rag_system = NexusRAG()

    user_query = st.text_input("Ask a question about your documents:")
    if st.button("Search"):
        with st.spinner("Retrieving & generating answer..."):
            answer = rag_system.generate_answer(user_query, top_k=3)
        st.write("### Answer")
        st.write(answer)

if __name__ == "__main__":
    main()