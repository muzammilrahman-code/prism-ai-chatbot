import streamlit as st
import os
from dotenv import load_dotenv

# Import the helper functions from the other file
from langchain_helper import get_qa_chain, create_vector_db

# Load environment variables (e.g., GROQ_API_KEY)
load_dotenv()


def main():
    """Main Streamlit application function for the Prism AI Chatbot."""
    
    st.set_page_config(page_title="Prism AI Q&A Chatbot ", layout="centered", page_icon="🤖")
    st.title("Prism AI Q&A Chatbot 🤖")

    # --- 1. Create Knowledgebase Button ---
    if st.button("Create/Update Knowledgebase"):
        with st.spinner("Building vector database from 'prism_ai.csv'..."):
            create_vector_db()
        st.success(" Knowledgebase created successfully! You can now ask questions.")

    st.markdown("---")
    
    # --- 2. Question Input ---
    question = st.text_input("Ask a question about Prism AI:", placeholder="E.g., What is Prism AI?")

    # --- 3. Get Answer ---
    if question:
        # Load the QA Chain
        chain = get_qa_chain()
        
        if chain is None:
            # Display error if DB hasn't been created yet
            st.error(" Knowledgebase not found. Please click 'Create/Update Knowledgebase' first.")
        else:
            with st.spinner("Finding answer..."):
                try:
                    # Use the chain to get the response
                    response = chain.invoke({"query": question}) 

                    st.subheader("Answer:")
                    st.markdown(response["result"]) # Use markdown for better formatting
                    
                   

                except Exception as e:
                    # Catch API key errors or other runtime issues
                    st.error(f"An error occurred: Ensure your Ai is working correctly, Error details: {e}")

if __name__ == "__main__":
    main()
