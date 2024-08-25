import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import PromptTemplate, LLMChain
from langchain_community.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
import os
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.schema import Document
import yt_dlp
from pydub import AudioSegment

api_key = 'AIzaSyDG1Key2SaOs73YXzBQyZ0kxUKH-Liosis'

if not api_key:
    st.error("API key for Gemini is not set. Please set the GEMINI_API_KEY environment variable.")
    st.stop()

chat_history = []

def get_video_id(url):
    video_id = ""
    if "shorts/" in url:
        video_id = url.split("shorts/")[1].split("?")[0]
    elif "youtu.be/" in url:
        video_id = url.split("youtu.be/")[1].split("?")[0]
    elif "watch?v=" in url:
        video_id = url.split("watch?v=")[1].split("&")[0]
    return video_id


import yt_dlp
from yt_dlp.utils import DownloadError
from pydub import AudioSegment
import os

def get_audio(video_url, output_path='audio.mp3'):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': output_path,
        'quiet': True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        return output_path
    except DownloadError as e:
        print(f"DownloadError: {e}")
        return None


def get_transcript_or_audio(video_url, file_path='transcription.txt'):
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(get_video_id(video_url))
        text = " ".join([segment["text"] for segment in transcript_list]).strip()
        with open(file_path, 'w') as file:
            file.write(text)
        return file_path
    except Exception as e:
        print(f"Error: {e}")
        print("Transcripts are disabled, downloading audio instead.")
        audio_file_path = get_audio(video_url)
        return audio_file_path


def generate_response(user_input, file_path=None):
    global chat_history
    if not user_input:
        return "No user input provided.", None
    
    if file_path and os.path.exists(file_path):
        # Existing logic to handle the transcript and generate a response
        with open(file_path, 'r') as file:
            transcription = file.read()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        documents = text_splitter.split_text(transcription)
        documents = [Document(page_content=chunk) for chunk in documents]

        try:
            if os.path.exists('db'):
                Chroma.delete("db")
            embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
            db = Chroma.from_documents(documents, embeddings)
        except Exception as e:
            return f"Error during vector store initialization: {str(e)}", file_path

        retrieved_results = db.similarity_search(user_input)

        history_context = "\n".join([f"User: {entry['user']}\nAssistant: {entry['assistant']}" for entry in chat_history])

        llm = ChatGoogleGenerativeAI(
            temperature=0.6,
            google_api_key=api_key,
            model='gemini-1.5-flash',
            max_output_tokens=2048,
            verbose=True,
        )

        prompt_template = PromptTemplate(
            input_variables=["context", "history", "question"],
            template="""You are a helpful assistant who answers questions based on the provided context and previous interactions.
                        Context: {context}
                        History: {history}
                        Question: {question}"""
        )
        llm_chain = LLMChain(llm=llm, prompt=prompt_template, verbose=True)

        output = llm_chain.run({"context": retrieved_results, "history": history_context, "question": user_input})

        chat_history.append({"user": user_input, "assistant": output})
        return output

    else:
        # Handle cases where the transcript is not available
        if not file_path:
            return "Transcripts are disabled for this video, or no transcript was found.", None
        
        history_context = "\n".join([f"User: {entry['user']}\nAssistant: {entry['assistant']}" for entry in chat_history])

        llm = ChatGoogleGenerativeAI(
            temperature=0.6,
            google_api_key=api_key,
            model='gemini-1.5-flash',
            max_output_tokens=2048,
            verbose=True,
        )

        prompt_template = PromptTemplate(
            input_variables=["history", "question"],
            template="""You are a helpful assistant who answers questions based on previous interactions.
                        History: {history}
                        Question: {question}"""
        )
        llm_chain = LLMChain(llm=llm, prompt=prompt_template, verbose=True)

        output = llm_chain.run({"history": history_context, "question": user_input})
        chat_history.append({"user": user_input, "assistant": output})
        return output


# Streamlit UI
st.title("YouTube Video Chat")
st.write("Enter a YouTube URL and ask questions about the video's content.")

youtube_url = st.text_input("YouTube URL (Optional)")
question = st.text_input("Question")

if st.button("Submit"):
    file_path = get_transcript_or_audio(youtube_url) if youtube_url else None
    response = generate_response(question, file_path)
    st.write(response)
