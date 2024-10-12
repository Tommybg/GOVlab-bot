import streamlit as st 
import os 
import time
from dotenv import load_dotenv 
from html_template import css, bot_template, user_template
from langchain_openai import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, CSVLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment variables from .env file
load_dotenv()

def on_api_key_entered(api_key):
    os.environ["OPENAI_API_KEY"] = api_key
    st.success("API Key configurada correctamente!", icon="✅")

# Initialize vector_store in session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

# Function to clear chat history
# Function to clear chat history and memory
def clear_chat_history():
    st.session_state.messages = []
    st.success("Historial de chat y memoria borrados!")


# Function to load document
@st.cache_data
def load_document(file):
    name, extension = os.path.splitext(file)

    if extension.lower() in ['.pdf', '.PDF']:
        loader = PyPDFLoader(file)
    elif extension.lower() == '.docx':
        loader = Docx2txtLoader(file)
    elif extension.lower() == '.txt':
        loader = TextLoader(file)
    elif extension.lower() == '.csv':
        loader = CSVLoader(file)
    else:
        st.error('Formato de documento no soportado!')
        return None

    try:
        docs = loader.load()
        if docs is None or len(docs) == 0:
            st.error("No se pudieron cargar documentos desde el archivo.")
            return None
        return docs
    except Exception as e:
        st.error(f"Error al cargar el documento: {str(e)}")
        return None

# Function to chunk data
def chunk_data(docs, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(docs)
    return chunks

# Function to create embeddings and Chroma DB
def create_embeddings_chroma(chunks):
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small', openai_api_key=api_key)
    try:
        vector_store = Chroma.from_documents(chunks, embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Error al crear el vector store de Chroma: {e}")
        return None

# Function to format documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Function to generate response
def generate_response(q):
    if st.session_state.vector_store is None:
        return "PorFavor suba un documento primero."

    try:
        chat_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.5, api_key=api_key)
        
        system_prompt = """
        Eres un asistente para la división del Govlab de la Universidad de la Sabana: El GovLab es un laboratorio de innovación de la Universidad de La Sabana, 
        cuyo propósito es la búsqueda de soluciones a problemas públicos a través de diferentes técnicas, métodos y enfoques soportados en la analítica de datos, 
        la co-creación y la colaboración intersectorial quien trabaja con el Gobierno Colombiano,
        Tú eres responsable de encontrar potenciales oportunidades de consultoría dentro los documentos subidos, los planes de desarrollo de los municipios del país.
        Además debes generar valor e ideas a partir de los documentos cargados.
        """

        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            HumanMessagePromptTemplate.from_template("{context}\n\nQuestion: {question}\nAnswer:")
        ])

        retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
        
        rag_chain = (
            {
                "context": lambda x: format_docs(retriever.get_relevant_documents(x["question"])),
                "question": RunnablePassthrough()
            }
            | prompt 
            | chat_model 
            | StrOutputParser()
        )

        response = rag_chain.invoke({"question": q})
        return response

    except Exception as e:
        st.error(f"Ocurrió un error al generar la respuesta: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return "Lo siento, pero ocurrió un error al procesar su solicitud. Por favor, intente de nuevo o contacte al soporte si el problema persiste."

# Streamlit UI 
st.set_page_config(page_title="GOV Copilot", layout="centered")
st.title("GOVpilot")
st.write("Tu copiloto para identificar consultorias y entender Planes de desarrollo") 

st.write(css, unsafe_allow_html=True)

# Create UPLOADED FILES folder 
documents_folder = './documents'
os.makedirs(documents_folder, exist_ok=True)

# Sidebar
with st.sidebar: 
    api_key = st.text_input("Ingresa tu API Key de OpenAI", type="password")
    if api_key:
        on_api_key_entered(api_key)
        
    uploaded_files = st.file_uploader("Sube tus documentos ACÁ", type=["pdf", "csv", "txt", "xlsx", "docx"], accept_multiple_files=True)
    add_data = st.button("Cargar Documentos")

    if uploaded_files and add_data:
        all_chunks = []  # To store all chunks from multiple documents
        with st.spinner("Procesando sus Documentos..."):
            progress_bar = st.progress(0)
            for i, uploaded_file in enumerate(uploaded_files):
                bytes_data = uploaded_file.read()
                file_name = os.path.join(documents_folder, uploaded_file.name)

                with open(file_name, "wb") as f:
                    f.write(bytes_data)

                docs = load_document(file_name)
                if docs:
                    chunks = chunk_data(docs)
                    all_chunks.extend(chunks)

                progress_bar.progress((i + 1) / len(uploaded_files))

            vector_store = create_embeddings_chroma(all_chunks)

            if vector_store:
                st.session_state.vector_store = vector_store
                st.success("Documento(s) procesado(s) con éxito!")
            else:
                st.error("Error en procesar documentos. Por favor intente de nuevo.") 

    if st.button("Borrar Historial de Chat"):
        clear_chat_history()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    if message["role"] == "user":
        st.write(user_template.replace("{{MSG}}", message["content"]), unsafe_allow_html=True)
    else:
        st.write(bot_template.replace("{{MSG}}", message["content"]), unsafe_allow_html=True)

# Chat Input 
if prompt := st.chat_input("¿En qué puedo ayudarte hoy?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("T"):
        st.markdown(user_template.replace("{{MSG}}", prompt), unsafe_allow_html=True)

    with st.chat_message("A"):
        message_placeholder = st.empty()
        full_response = generate_response(prompt)
        
        message_placeholder.markdown(bot_template.replace("{{MSG}}", full_response), unsafe_allow_html=True)

    st.session_state.messages.append({"role": "assistant", "content": full_response})