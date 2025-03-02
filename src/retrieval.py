import os
from typing import List, Dict, Tuple, Optional
import json
from tqdm import tqdm
import pandas as pd
import time
import logging
import base64
from PIL import Image
import io
from langchain.embeddings import OpenAIEmbeddings, VertexAIEmbeddings
from langchain.vectorstores import FAISS, PostgresVectorStore
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_cloud_sql_pg import PostgresEngine

data_path="C:\Users\sarra\Desktop\medical-chatbot\data\medquad.csv"

# Configuration du logger
logger = logging.getLogger(__name__)

class LayoutLMDocumentProcessor:
    def __init__(self, api_key: Optional[str] = None, use_vertex_ai: bool = False):
        """
        Initialize document processor with embedding model.
        
        Args:
            api_key: API key for OpenAI (required if use_vertex_ai is False)
            use_vertex_ai: If True, use Vertex AI embeddings instead of OpenAI
        """
        self.use_vertex_ai = use_vertex_ai
        if use_vertex_ai:
            self.embeddings = VertexAIEmbeddings(model_name="textembedding-gecko@latest")
        else:
            if not api_key:
                raise ValueError("API key is required for OpenAI embeddings")
            self.embeddings = OpenAIEmbeddings(api_key=api_key)
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.logger = logger
    
    def process_kaggle_layoutlm_dataset(self, data_path: str) -> List[Dict]:
        """
        Process the Kaggle LayoutLM dataset to extract medical documents.
        
        Args:
            data_path: Path to the dataset directory
            
        Returns:
            List of document dictionaries with text, source, layout information and focus_area
        """
        processed_docs = []
        
        if not os.path.exists(data_path):
            self.logger.error(f"Le chemin du dataset n'existe pas : {data_path}")
            raise FileNotFoundError(f"Dataset path not found: {data_path}")
        
        # Parcourir les dossiers du dataset LayoutLM
        # Structure typique: /document_category/document_id/...
        for category_dir in os.listdir(data_path):
            category_path = os.path.join(data_path, category_dir)
            if os.path.isdir(category_path):
                for doc_id in os.listdir(category_path):
                    doc_path = os.path.join(category_path, doc_id)
                    if os.path.isdir(doc_path):
                        # Chercher le fichier de données JSON et l'image
                        ocr_json_path = os.path.join(doc_path, "ocr.json")
                        image_path = None
                        
                        # Chercher le fichier image (peut être jpg, png, etc.)
                        for file in os.listdir(doc_path):
                            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
                                image_path = os.path.join(doc_path, file)
                                break
                        
                        if os.path.exists(ocr_json_path):
                            try:
                                with open(ocr_json_path, 'r', encoding='utf-8') as f:
                                    ocr_data = json.load(f)
                                
                                # Extraire le texte et les informations de layout
                                text = ""
                                layout_boxes = []
                                
                                if 'recognitionResults' in ocr_data:
                                    for page in ocr_data['recognitionResults']:
                                        for line in page.get('lines', []):
                                            text += line.get('text', '') + "\n"
                                            # Stocker les coordonnées de la bounding box
                                            box = line.get('boundingBox', [])
                                            if len(box) >= 8:  # Format: [x1,y1,x2,y2,x3,y3,x4,y4]
                                                # Convertir en format [x0,y0,x1,y1] (haut-gauche, bas-droite)
                                                x_coords = [box[i] for i in range(0, 8, 2)]
                                                y_coords = [box[i] for i in range(1, 8, 2)]
                                                layout_boxes.append([
                                                    min(x_coords), min(y_coords),
                                                    max(x_coords), max(y_coords)
                                                ])
                                
                                # Lire l'image si elle existe
                                image_data = None
                                if image_path and os.path.exists(image_path):
                                    try:
                                        with open(image_path, 'rb') as img_file:
                                            image_data = base64.b64encode(img_file.read()).decode('utf-8')
                                    except Exception as e:
                                        self.logger.warning(f"Erreur lors de la lecture de l'image {image_path}: {e}")
                                
                                if text:
                                    processed_docs.append({
                                        "text": text,
                                        "source": f"{category_dir}/{doc_id}",
                                        "focus_area": category_dir,
                                        "has_layout": len(layout_boxes) > 0,
                                        "layout_boxes": layout_boxes,
                                        "image_data": image_data
                                    })
                                    
                            except json.JSONDecodeError as e:
                                self.logger.warning(f"Fichier JSON invalide : {ocr_json_path} - {e}")
                            except Exception as e:
                                self.logger.error(f"Erreur lors du traitement du document {doc_path} : {e}")
        
        if not processed_docs:
            self.logger.warning("Aucun document valide n'a été trouvé dans le dataset LayoutLM.")
        else:
            self.logger.info(f"Nombre de documents LayoutLM traités : {len(processed_docs)}")
        
        return processed_docs
    
    def _validate_embeddings(self, texts: List[str]) -> bool:
        """
        Validate that embeddings can be generated for the given texts.
        """
        try:
            sample_embedding = self.embeddings.embed_query(texts[0])
            return len(sample_embedding) > 0
        except Exception as e:
            self.logger.error(f"Erreur lors de la génération des embeddings : {e}")
            return False
    
    def create_faiss_vectorstore(self, documents: List[Dict], save_path: Optional[str] = None) -> FAISS:
        """
        Create a FAISS vectorstore from processed documents.
        
        Args:
            documents: List of document dictionaries
            save_path: Optional path to save the vectorstore
            
        Returns:
            FAISS vectorstore instance
        """
        # Convert to LangChain documents
        langchain_docs = []
        
        for doc in documents:
            chunks = self.text_splitter.split_text(doc["text"])
            for chunk in chunks:
                metadata = {
                    "source": doc["source"],
                    "focus_area": doc["focus_area"],
                    "has_layout": doc.get("has_layout", False)
                }
                
                # Ajouter les informations de layout si disponibles
                if doc.get("has_layout", False):
                    metadata["layout_boxes"] = doc.get("layout_boxes", [])
                    # Ne pas inclure l'image dans les métadonnées car cela peut être très volumineux
                    # Stocker un indicateur si l'image est disponible
                    metadata["has_image"] = doc.get("image_data") is not None
                
                langchain_docs.append(
                    Document(
                        page_content=chunk,
                        metadata=metadata
                    )
                )
        
        # Validate embeddings
        if not langchain_docs:
            raise ValueError("Aucun document à indexer.")
            
        if not self._validate_embeddings([doc.page_content for doc in langchain_docs[:1]]):
            raise ValueError("Erreur lors de la génération des embeddings.")
        
        # Create vectorstore
        vectorstore = FAISS.from_documents(langchain_docs, self.embeddings)
        
        # Save if path is provided
        if save_path:
            vectorstore.save_local(save_path)
            self.logger.info(f"Vectorstore sauvegardé à : {save_path}")
        
        return vectorstore
    
    def create_pg_vectorstore(self, documents: List[Dict], table_name: str) -> PostgresVectorStore:
        """
        Create a PostgresVectorStore from processed documents.
        
        Args:
            documents: List of document dictionaries
            table_name: Name of the table to store vectors
            
        Returns:
            PostgresVectorStore instance
        """
        if not self.use_vertex_ai:
            raise ValueError("PostgresVectorStore requires Vertex AI embeddings")
        
        try:
            # Initialize PostgresEngine
            engine = PostgresEngine.from_instance(
                project_id=os.getenv("PROJECT_ID"),
                region=os.getenv("REGION"),
                instance=os.getenv("INSTANCE"),
                database=os.getenv("DB_NAME"),
                user=os.getenv("DB_USER"),
                password=os.getenv("DB_PASS")
            )
            
            # Convert to LangChain documents with layout information
            langchain_docs = []
            for doc in documents:
                chunks = self.text_splitter.split_text(doc["text"])
                for chunk in chunks:
                    metadata = {
                        "source": doc["source"],
                        "focus_area": doc["focus_area"],
                        "has_layout": doc.get("has_layout", False)
                    }
                    
                    # Ajouter les informations de layout si disponibles
                    if doc.get("has_layout", False):
                        metadata["layout_boxes"] = doc.get("layout_boxes", [])
                        metadata["has_image"] = doc.get("image_data") is not None
                    
                    langchain_docs.append(
                        Document(
                            page_content=chunk,
                            metadata=metadata
                        )
                    )
            
            # Create vectorstore
            vectorstore = engine.as_vector_store(table_name=table_name, embedding=self.embeddings)
            vectorstore.add_documents(langchain_docs)
            
            self.logger.info(f"Vectorstore créé avec succès dans la table {table_name}.")
            return vectorstore
        except Exception as e:
            self.logger.error(f"Erreur lors de la création du vectorstore : {e}")
            raise
    
    @staticmethod
    def load_faiss_vectorstore(save_path: str, embeddings) -> FAISS:
        """Load a saved FAISS vectorstore."""
        return FAISS.load_local(save_path, embeddings)
    
    def get_document_image(self, document_id: str, dataset_path: str) -> Optional[bytes]:
        """
        Retrieve the image for a specific document from the dataset.
        
        Args:
            document_id: The document identifier (category/doc_id)
            dataset_path: Path to the dataset directory
            
        Returns:
            Image bytes or None if not found
        """
        if not document_id or '/' not in document_id:
            return None
        
        category, doc_id = document_id.split('/', 1)
        doc_path = os.path.join(dataset_path, category, doc_id)
        
        if not os.path.exists(doc_path) or not os.path.isdir(doc_path):
            return None
        
        # Chercher le fichier image
        for file in os.listdir(doc_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
                image_path = os.path.join(doc_path, file)
                try:
                    with open(image_path, 'rb') as f:
                        return f.read()
                except Exception as e:
                    self.logger.error(f"Erreur lors de la lecture de l'image {image_path}: {e}")
                    return None
        
        return None
        
    def evaluate_retrieval(self, vectorstore, test_queries: List[Dict]) -> pd.DataFrame:
        """
        Evaluate the retrieval performance using test queries.
        
        Args:
            vectorstore: FAISS or PostgresVectorStore instance
            test_queries: List of dicts with 'question' and 'relevant_sources'
            
        Returns:
            DataFrame with evaluation metrics
        """
        results = []
        
        for query in test_queries:
            question = query["question"]
            relevant_sources = set(query["relevant_sources"])
            
            start_time = time.time()
            
            # Retrieve documents
            if isinstance(vectorstore, FAISS):
                docs_and_scores = vectorstore.similarity_search_with_score(question, k=5)
                retrieved_sources = [doc.metadata["source"] for doc, _ in docs_and_scores]
            else:
                retriever = vectorstore.as_retriever(search_type="similarity", k=5)
                retrieved_docs = retriever.invoke(question)
                retrieved_sources = [doc.metadata["source"] for doc in retrieved_docs]
            
            elapsed_time = time.time() - start_time
            
            # Calculate metrics
            retrieved_set = set(retrieved_sources)
            relevant_retrieved = retrieved_set.intersection(relevant_sources)
            
            precision = len(relevant_retrieved) / len(retrieved_set) if retrieved_set else 0
            recall = len(relevant_retrieved) / len(relevant_sources) if relevant_sources else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
            
            results.append({
                "question": question,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "retrieved_sources": retrieved_sources,
                "response_time": elapsed_time,
                "num_relevant_retrieved": len(relevant_retrieved)
            })
        
        return pd.DataFrame(results)

def get_relevant_documents(query: str, vector_store, consider_layout: bool = True) -> Tuple[List[Document], Dict]:
    """
    Retrieve relevant documents and layout data based on a query using a vector store.
    
    Args:
        query: The search query string.
        vector_store: FAISS or PostgresVectorStore instance.
        consider_layout: Whether to include layout information in the result.
        
    Returns:
        Tuple of (List of documents relevant to the query, Dict of layout data)
    """
    layout_data = {}
    
    if isinstance(vector_store, FAISS):
        docs_and_scores = vector_store.similarity_search_with_score(query, k=5)
        docs = [doc for doc, _ in docs_and_scores]
    else:
        retriever = vector_store.as_retriever(search_type="similarity", k=5)
        docs = retriever.invoke(query)
    
    # Extract layout information if requested
    if consider_layout:
        for doc in docs:
            if doc.metadata.get("has_layout", False):
                source = doc.metadata.get("source")
                if source and source not in layout_data:
                    layout_data[source] = {
                        "boxes": doc.metadata.get("layout_boxes", []),
                        "has_image": doc.metadata.get("has_image", False)
                    }
    
    return docs, layout_data

def format_relevant_documents(documents: List[Document], consider_layout: bool = True) -> str:
    """
    Format relevant documents into a string, optionally including layout information.
    
    Args:
        documents: List of relevant documents.
        consider_layout: Whether to include layout information.
        
    Returns:
        Formatted string with document content.
    """
    formatted_docs = []
    
    for doc in documents:
        content = doc.page_content
        metadata = doc.metadata
        
        formatted_doc = f"Source: {metadata.get('source', 'Unknown')}\n"
        formatted_doc += f"Catégorie: {metadata.get('focus_area', 'General')}\n"
        
        if consider_layout and metadata.get("has_layout", False):
            formatted_doc += "Informations de layout disponibles: Oui\n"
            # Ne pas inclure les coordonnées des boxes pour ne pas encombrer le contexte
        else:
            formatted_doc += "Informations de layout disponibles: Non\n"
        
        formatted_doc += f"\nContenu:\n{content}\n"
        formatted_docs.append(formatted_doc)
    
    return "\n" + "-" * 50 + "\n".join(formatted_docs)