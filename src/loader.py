from langchain_community.document_loaders import PyPDFLoader

def load_pdf(file_path: str):
    """
    Load a PDF file and return a list of Document objects.
    """
    
    loader = PyPDFLoader(file_path)
    
    documents = loader.load()
    
    return documents