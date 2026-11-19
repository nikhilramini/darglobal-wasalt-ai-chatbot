
import streamlit as st

from rag_chatbot import generate_answer


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="DarGlobal + Wasalt AI Assistant",
    page_icon="🏡",
    layout="centered"
)


# ==========================================
# 2. APPLICATION HEADER
# ==========================================

st.title("🏡 DarGlobal + Wasalt AI Assistant")

st.write(
    "Your AI-powered real estate assistant. "
    "Ask questions about DarGlobal, Wasalt, "
    "and available properties in Riyadh."
)

st.divider()


# ==========================================
# 3. SIDEBAR
# ==========================================

with st.sidebar:

    st.header("About this chatbot")

    st.write(
        "This chatbot uses publicly collected "
        "DarGlobal and Wasalt information."
    )

    st.write("**Capabilities:**")

    st.write("- Property price filtering")
    st.write("- Bedroom filtering")
    st.write("- Riyadh neighborhood searches")
    st.write("- DarGlobal company information")
    st.write("- Wasalt platform information")

    st.divider()

    if st.button("Clear conversation"):

        st.session_state.messages = []

        st.rerun()


# ==========================================
# 4. CHAT HISTORY
# ==========================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# Display previous messages

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ==========================================
# 5. CHAT INPUT AND RESPONSE
# ==========================================

question = st.chat_input(
    "Ask about DarGlobal, Wasalt, or Riyadh properties..."
)


if question:

    # Display user question

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.markdown(question)


    # Generate chatbot response

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching property data and generating an answer..."
        ):

            try:

                answer = generate_answer(question)

                st.markdown(answer)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )

            except Exception:

                st.error(
                    "Something went wrong while generating "
                    "the response. Please try again."
                )

