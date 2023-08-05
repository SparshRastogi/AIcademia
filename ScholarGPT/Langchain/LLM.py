from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.llms import CTransformers
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap  = 100,
    length_function = len,)
texts = text_splitter.split_text(raw_text)
llm = LlamaCpp(
    model_path=model_path,
    max_tokens=256,
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    callback_manager=callback_manager,
    n_ctx=1536,
    verbose=False,)




chain = load_qa_chain(llm, chain_type="stuff")
loader = DirectoryLoader(source_path,glob = "*.pdf",loader_cls = PyPDFLoader)



data = loader.load()



splitter = RecursiveCharacterTextSplitter(chunk_size = 500,chunk_overlap = 100)



chunks = splitter.split_documents(data)



embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')




db = FAISS.from_documents(chunks,embeddings)

db.save_local(db_path)
custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""
