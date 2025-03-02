from typing import List, Dict, Tuple, Any
import logging
import os
import json
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_google_cloud_sql_pg import PostgresEngine
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from transformers import LayoutLMv2Processor, LayoutLMModel
import torch
from PIL import Image

# Configuration du logger
logger = logging.getLogger(__name__)

class MedicalChatbot:
    def __init__(self, api_key: str):
        """
        Initialize the medical chatbot with Google Vertex AI components and LayoutLM integration.
        
        Args:
            api_key: OpenAI API key
        """
        self.api_key = api_key
        
        # Initialiser le modèle Gemini
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0.2,
            max_output_tokens=1024,
            safety_settings=[
                {"category": "HARM_CATEGORY_DANGEROUS", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            ]
        )
        
        # Initialiser les embeddings de Vertex AI
        self.embeddings = VertexAIEmbeddings(model_name="textembedding-gecko")
        
        # Initialiser LayoutLM pour le traitement de documents avec layout
        self.layoutlm_processor = LayoutLMProcessor.from_pretrained("microsoft/layoutlm-base-uncased")
        self.layoutlm_model = LayoutLMModel.from_pretrained("microsoft/layoutlm-base-uncased")
        
        self.vectorstore = None
        self.qa_chain = None
        self.logger = logger
        self.logger.info("Chatbot médical initialisé avec Vertex AI et LayoutLM")
        
    def initialize_vectorstore(self, documents: List[Dict]):
        """
        Initialize vectorstore with medical documents.
        
        Args:
            documents: List of document dictionaries with source, content, and focus_area fields
        """
        try:
            # Convertir les documents au format attendu
            texts = []
            metadatas = []
            
            for doc in documents:
                texts.append(doc.get("content", ""))
                metadatas.append({
                    "source": doc.get("source", "Unknown"),
                    "focus_area": doc.get("focus_area", "General"),
                    "has_layout": False
                })
            
            # Créer le vectorstore FAISS (pour le développement local)
            self.vectorstore = FAISS.from_texts(
                texts,
                self.embeddings,
                metadatas=metadatas
            )
            
            self.logger.info(f"Vectorstore initialisé avec succès. {len(texts)} documents chargés.")
        except Exception as e:
            self.logger.error(f"Erreur lors de l'initialisation du vectorstore : {e}")
            raise
    
    def setup_qa_chain(self):
        """Setup the QA chain with custom medical prompt."""
        try:
            # Template de prompt bilingue avec prise en compte du layout
            prompt_template = """Tu es un assistant médical expert. Utilise strictement le contexte suivant 
            pour répondre à la question. Ta réponse doit être précise, factuelle et basée uniquement sur 
            les informations fournies. 
            
            Certains documents contiennent des informations sur leur mise en page (layout) que tu dois 
            prendre en compte. Les positions spatiales des éléments dans les documents médicaux peuvent 
            être importantes (par exemple, les en-têtes, les tableaux de résultats, les signatures).
            
            Si tu ne peux pas répondre à partir du contexte, dis simplement 
            "Je ne trouve pas suffisamment d'informations dans mes sources pour répondre à cette question."

            Contexte: {context}

            Question: {question}
            
            Réponse détaillée et précise:"""
            
            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            if not self.vectorstore:
                raise ValueError("Le vectorstore doit être initialisé avant de configurer la chaîne QA")
            
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(
                    search_kwargs={"k": 3}
                ),
                chain_type_kwargs={"prompt": PROMPT},
                return_source_documents=True
            )
            
            self.logger.info("Chaîne QA configurée avec succès.")
        except Exception as e:
            self.logger.error(f"Erreur lors de la configuration de la chaîne QA : {e}")
            raise