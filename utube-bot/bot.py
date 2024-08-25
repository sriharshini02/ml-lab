import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import ChatGoogleGenerativeAI

# Global variable for chat history
chat_history = []

# Function to get transcript from a YouTube URL
def get_transcript(youtube_url):
    video_id = youtube_url.split("v=")[-1]
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join([entry["text"] for entry in transcript])
        return transcript_text
    except Exception as e:
        return str(e)

# Function to generate a response based on the question and transcript
def generate_response(question, transcript_text):
    if transcript_text is None:
        return "No transcript available. Please provide a valid YouTube URL."
    
    # Initialize the vector store and embedding model
    embeddings = HuggingFaceEmbeddings()
    vectordb = Chroma.from_texts([transcript_text], embedding=embeddings)

    # Initialize the LLM model
    model = ChatGoogleGenerativeAI(model="text-bison-001")

    # Perform the query
    result = vectordb.similarity_search_with_score(question)
    relevant_text = result[0][0]
    
    # Generate the final response
    response = model.generate(context=relevant_text, query=question)
    
    # Update chat history
    chat_history.append({"user": question, "assistant": response})
    
    return response

# Function to handle the interaction in Streamlit
def streamlit_interface(youtube_url, question):
    transcript_text = None
    if youtube_url:
        transcript_text = get_transcript(youtube_url)
    response = generate_response(question, transcript_text)
    return response

# Streamlit app
st.title("YouTube Video Chat")
st.write("Enter a YouTube URL and ask questions about the video's content.")

youtube_url = st.text_input("YouTube URL (Optional)")
question = st.text_input("Question")

if st.button("Submit"):
    if question:
        response = streamlit_interface(youtube_url, question)
        st.write(f"Assistant: {response}")
    else:
        st.warning("Please enter a question.")
