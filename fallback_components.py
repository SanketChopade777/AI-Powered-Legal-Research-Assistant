import streamlit as st
import time

def show_chatbot_fallback():
    st.markdown("## 🤖 Legal Chatbot (Fallback Mode)")
    st.info("The chatbot module is not available. Here's a simple chat interface:")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a legal question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response = "I'm a fallback chatbot. In the full version, I would provide legal insights based on your question."
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})


def show_document_fallback():
    st.markdown("## 📄 Document Analysis (Fallback Mode)")
    st.info("The document analysis module is not available. Here's a simple upload interface:")

    uploaded_file = st.file_uploader("Upload legal document", type=["pdf", "docx", "txt"])
    if uploaded_file:
        st.success(f"Uploaded: {uploaded_file.name}")
        if st.button("Analyze Document"):
            with st.spinner("Analyzing document..."):
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)

                st.success("Analysis complete!")
                st.json({
                    "document_type": "Legal Contract",
                    "key_clauses": ["Termination Clause", "Liability Clause"],
                    "risk_level": "Low",
                    "recommendations": ["Review section 4.2", "Consult for clause 7.1"]
                })


def show_lawyer_fallback():
    st.markdown("## 👨‍💼 Lawyer Finder (Fallback Mode)")
    st.info("The lawyer finder module is not available. Here's a simple search interface:")

    col1, col2 = st.columns(2)
    with col1:
        specialty = st.selectbox("Specialty", ["Corporate Law", "Criminal Defense", "Family Law", "Real Estate"])
    with col2:
        location = st.selectbox("Location", ["New York", "California", "Texas", "Florida"])

    if st.button("Search Lawyers"):
        lawyers = [
            {"name": "Dr. Sarah Johnson", "specialty": "Corporate Law", "rating": 4.8, "experience": "15 years"},
            {"name": "Robert Chen", "specialty": "Criminal Defense", "rating": 4.6, "experience": "8 years"},
            {"name": "Maria Rodriguez", "specialty": "Family Law", "rating": 4.9, "experience": "12 years"}
        ]

        for lawyer in lawyers:
            with st.expander(f"👤 {lawyer['name']} - ⭐ {lawyer['rating']}"):
                st.write(f"**Specialty:** {lawyer['specialty']}")
                st.write(f"**Experience:** {lawyer['experience']}")
                st.button("Contact Lawyer", key=f"contact_{lawyer['name']}")