import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
import praw

def get_subreddit_text(subreddit_name, num_posts, keyword):
    reddit = praw.Reddit(client_id='NZckJpvlm8aLidGbFg8cTg', client_secret='tQtvC2Ze8NICddPgLECxWG7sc6lyUQ', user_agent='Get the name of the subreddit and chat with it, based on scraped information. The information available to scrape is limited.')
    subreddit = reddit.subreddit(subreddit_name)
    
    posts_text = ""
    for post in subreddit.search(keyword, limit=num_posts):
        posts_text += post.title + "\n" + post.selftext
        post.comments.replace_more(limit=None)
        for comment in post.comments.list():
            posts_text += comment.body + "\n"
    print(posts_text)
    return posts_text


def get_reddit_text(num_posts, keyword):
    reddit = praw.Reddit(client_id='NZckJpvlm8aLidGbFg8cTg', client_secret='tQtvC2Ze8NICddPgLECxWG7sc6lyUQ', user_agent='Get the name of the subreddit and chat with it, based on scraped information. The information available to scrape is limited.')
    
    posts_text = ""
    for post in reddit.subreddit("all").search(keyword, limit=num_posts):
        posts_text += post.title + "\n" + post.selftext
        post.comments.replace_more(limit=None)
        for comment in post.comments.list():
            posts_text += comment.body + "\n"
    print(posts_text)
    return posts_text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="Ask Reddit Anything", initial_sidebar_state="expanded")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Ask Reddit Anything")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your subreddit")
        subreddit_name = st.text_input("Enter the subreddit name (leave blank to search all of Reddit):")
        keyword = st.text_input("Enter the search keywords:")
        num_posts = st.number_input("Enter the number of posts:", min_value=1, value=10, step=1)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get subreddit text
                if subreddit_name:
                    raw_text = get_subreddit_text(subreddit_name, num_posts, keyword)
                else:
                    raw_text = get_reddit_text(num_posts, keyword)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)


if __name__ == '__main__':
    main()
