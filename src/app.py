import os
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from dotenv import load_dotenv
import logging
from datetime import datetime
import pandas as pd
import yaml
import json
import numpy as np

# Importations du projet
from model import MedicalChatbot
from database import DatabaseManager
from monitoring import Monitoring

# Importations spécifiques au TP4
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_cloud_sql_pg import PostgresEngine
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Charger les variables d'environnement
load_dotenv()

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Charger le fichier config.yaml
with open("C:/Users/sarra/Desktop/medical-chatbot/deployment/config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Vérifier les variables d'environnement
project_id = os.getenv("PROJECT_ID")
region = os.getenv("REGION")
instance = os.getenv("INSTANCE")
db_name = os.getenv("DATABASE")
db_user = os.getenv("DB_USER")
db_pass = os.getenv("DB_PASSWORD")

# Construire l'instance_connection_name
instance_connection_name = f"{project_id}:{region}:{instance}"

if not all([project_id, region, instance, db_name, db_user, db_pass]):
    print("Attention: Des variables d'environnement sont manquantes")

print(f"INSTANCE_CONNECTION_NAME: {instance_connection_name}")
print(f"DB_USER: {db_user}")
print(f"DB_NAME: {db_name}")

# Initialisation de l'application FastAPI
app = FastAPI(
    title="Medical Chatbot API",
    version="1.0.0",
    description="API pour un chatbot médical utilisant LangChain, Google Vertex AI et Gemini."
)

# Modèles Pydantic pour la validation des requêtes et réponses
class QuestionRequest(BaseModel):
    question: str
    temperature: float = 0.2
    language: str = "Français"
    consider_layout: bool = True
    max_documents: int = 5

class SourceInfo(BaseModel):
    source: str
    focus_area: str
    similarity_score: float
    has_layout: bool

class Answer(BaseModel):
    answer: str
    sources: List[SourceInfo]
    cost: float
    response_time: float

class FeedbackRequest(BaseModel):
    question: str
    feedback_score: float

class FeedbackStats(BaseModel):
    avg_score: float
    total_feedbacks: int

class DocumentResponse(BaseModel):
    page_content: str
    metadata: dict
    similarity_score: float = 1.0

# Instances globales pour optimiser la réutilisation
chatbot = None
db_manager = None
vector_store = None
embeddings = None
monitoring = Monitoring()

# Chemin du fichier CSV
csv_path = "C:/Users/sarra/Desktop/medical-chatbot/deployment/medquad.csv"

# Initialiser les structures pour le système RAG
embeddings_model = None
global_document_store = {}  # Stockage global des documents et leurs métadonnées

def initialize_rag_system():
    global embeddings_model, vector_store, global_document_store
    
    # Initialiser le modèle d'embedding
    print("Initialisation du modèle d'embedding...")
    embeddings_model = HuggingFaceEmbeddings(
        model_name=config.get("embeddings", {}).get("model_name", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
        cache_folder=config.get("embeddings", {}).get("cache_folder", "./model_cache")
    )
    
    # Vérifier si le fichier CSV existe
    if not os.path.exists(csv_path):
        print(f"ERREUR: Le fichier CSV n'existe pas à: {csv_path}")
        # Utiliser un dataset vide comme fallback
        df = pd.DataFrame(columns=["question", "answer", "source"])
    else:
        df = pd.read_csv(csv_path)
        print(f"Chargement de {len(df)} documents depuis le CSV")
    
    # Préparer les documents pour le vectorstore
    documents = []
    for idx, row in df.iterrows():
        if all(col in row for col in ['question', 'answer', 'source']):
            # Créer un document avec son contenu et métadonnées
            doc_content = f"Question: {row['question']}\nRéponse: {row['answer']}"
            doc_id = f"doc_{idx}"
            
            # Ajouter au stockage global
            global_document_store[doc_id] = {
                "content": doc_content,
                "metadata": {
                    "source": row['source'],
                    "original_question": row['question'],
                    "original_answer": row['answer'],
                    "has_layout": True,
                    "focus_area": "Médical",
                    "doc_id": doc_id,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            # Ajouter au format pour FAISS
            documents.append((doc_content, global_document_store[doc_id]["metadata"]))
    
    # Créer l'index vectoriel
    if documents:
        print(f"Création de l'index vectoriel avec {len(documents)} documents...")
        texts = [doc[0] for doc in documents]
        metadatas = [doc[1] for doc in documents]
        vector_store = FAISS.from_texts(texts=texts, embedding=embeddings_model, metadatas=metadatas)
        print("Index vectoriel créé avec succès")
    else:
        print("Aucun document valide trouvé pour créer l'index vectoriel")

# Initialiser le RAG au démarrage
initialize_rag_system()

def get_chatbot():
    """Initialisation et récupération de l'instance du chatbot."""
    global chatbot
    if chatbot is None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="GOOGLE_API_KEY non défini.")

        chatbot = MedicalChatbot(api_key=api_key)

        # Chargement des documents depuis la base de données
        documents = get_db_manager().get_all_documents()
        chatbot.initialize_vectorstore(documents)
        chatbot.setup_qa_chain()
    return chatbot

def get_db_manager():
    """Initialisation et récupération de l'instance du gestionnaire de base de données."""
    global db_manager
    if db_manager is None:
        db_params = {
            "instance_connection_name": instance_connection_name,
            "db_user": db_user,
            "db_pass": db_pass,
            "db_name": db_name
        }

        if not all(db_params.values()):
            raise HTTPException(status_code=500, detail="Certains paramètres de la base de données sont manquants.")

        db_manager = DatabaseManager(**db_params)
        db_manager.initialize_connection()
    return db_manager

def get_vector_store():
    """Initialisation et récupération de l'instance du vector store."""
    global vector_store, embeddings
    if vector_store is None:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        vector_store = FAISS.from_texts(texts=[], embedding=embeddings)
    return vector_store

@app.get("/")
def read_root():
    """Endpoint principal pour vérifier le bon fonctionnement de l'API."""
    return {"message": "Bienvenue sur l'API du Medical Chatbot", "version": "1.0.0"}

@app.post("/answer", response_model=Answer)
def get_answer(
    request: QuestionRequest,
    db: DatabaseManager = Depends(get_db_manager)
):
    """Obtenir une réponse à une question médicale en utilisant Google Gemini LLM."""
    start_time = datetime.now()
    try:
        # Récupérer les documents pertinents via le système RAG
        relevant_docs = query_rag_for_relevant_documents(
            question=request.question,
            max_documents=request.max_documents,
            consider_layout=request.consider_layout
        )
        
        # Préparer le contexte pour le LLM
        context = ""
        sources = []
        if relevant_docs:
            for i, doc in enumerate(relevant_docs):
                context += f"\nDocument {i+1}:\n{doc['page_content']}\n"
                sources.append(doc['metadata'].get('source', f'Document {i+1}'))
        
        # Si des documents pertinents sont trouvés, utilisez-les pour la réponse
        if relevant_docs and relevant_docs[0].get('similarity_score', 0) > 0.7:
            # Pour les questions très similaires à celles du dataset, utilisez directement la réponse
            answer_parts = relevant_docs[0]['page_content'].split("Réponse: ")
            if len(answer_parts) > 1:
                answer = answer_parts[1]
            else:
                answer = relevant_docs[0]['page_content']
        else:
            # Utilisez l'IA générative avec le contexte pour générer une réponse
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                return {"answer": "Erreur: Clé API Google non configurée"}
                
            llm = ChatGoogleGenerativeAI(
                model=config["model"]["name"],
                temperature=request.temperature,
                google_api_key=api_key,
                max_tokens=config["model"]["max_tokens"],
                timeout=None,
                max_retries=2,
            )
            
            # Prompt amélioré pour intégrer le contexte RAG
            rag_prompt = ChatPromptTemplate.from_messages([
                ("system", """Vous êtes un assistant médical spécialisé qui répond aux questions de santé.
                Votre rôle est de fournir des informations précises et utiles basées sur les documents médicaux fournis.
                Vous devez répondre dans la langue: {language}.
                Si les documents fournis contiennent l'information demandée, utilisez-les comme référence principale.
                Si les documents ne contiennent pas l'information nécessaire, basez-vous sur vos connaissances médicales générales.
                Incluez toujours des références aux sources quand elles sont disponibles.
                """),
                ("human", """Question: {question}
                
                Documents pertinents:
                {context}
                
                Veuillez fournir une réponse précise et documentée à cette question médicale."""),
            ])

            # Création de la chaîne RAG
            rag_chain = (
                {"context": lambda x: context, "question": lambda x: x, "language": lambda x: request.language}
                | rag_prompt 
                | llm 
                | StrOutputParser()
            )
            
            # Exécution de la chaîne RAG
            answer = rag_chain.invoke(request.question)

        response_time = (datetime.now() - start_time).total_seconds()

        # Enregistrement de la requête dans la base de données
        db.store_query(
            question=request.question,
            answer=answer,
            sources=sources,
            similarity_scores=[doc.get('similarity_score', 0) for doc in relevant_docs],
            cost=0.0,
            language=request.language
        )

        # Suivi des performances
        monitoring.log_performance(request.question, response_time, 0)  # Coût ici à mettre si calculé séparément

        return {
            "answer": answer,
            "sources": [SourceInfo(source=source, focus_area="Médical", similarity_score=1.0, has_layout=True) for source in sources],
            "cost": 0,  # Coût à mettre si calculé
            "response_time": response_time
        }
    except Exception as e:
        logger.error(f"Erreur lors de la récupération de la réponse : {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de la récupération de la réponse : {str(e)}")

@app.post("/get_sources", response_model=List[DocumentResponse])
def get_sources(request: QuestionRequest, vector_store=Depends(get_vector_store)):
    """Obtenir les documents pertinents pour une question."""
    try:
        relevant_docs = query_rag_for_relevant_documents(
            question=request.question,
            max_documents=request.max_documents,
            consider_layout=request.consider_layout
        )
        
        # Ajouter les informations de layout
        layout_data = {}
        for doc in relevant_docs:
            source = doc['metadata'].get('source', '')
            doc_id = doc['metadata'].get('doc_id', '')
            if source and request.consider_layout:
                doc['metadata']['has_layout'] = True
                layout_data[source] = synthetic_layout_info(doc_id)
        
        return [DocumentResponse(page_content=doc['page_content'], metadata=doc['metadata'], similarity_score=doc.get('similarity_score', 1.0)) for doc in relevant_docs]
    except Exception as e:
        logger.error(f"Erreur lors de la recherche des sources : {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de la recherche des sources : {str(e)}")

@app.post("/feedback", response_model=dict)
def submit_feedback(
    request: FeedbackRequest,
    db: DatabaseManager = Depends(get_db_manager)
):
    """Soumettre un feedback pour une réponse."""
    try:
        db.store_feedback(request.question, request.feedback_score)
        monitoring.log_feedback(request.question, request.feedback_score)
        return {"status": "success", "message": "Feedback enregistré avec succès"}
    except Exception as e:
        logger.error(f"Erreur lors de l'enregistrement du feedback : {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'enregistrement du feedback : {str(e)}")

@app.get("/stats", response_model=FeedbackStats)
def get_stats(db: DatabaseManager = Depends(get_db_manager)):
    """Obtenir les statistiques des feedbacks."""
    try:
        stats = db.get_feedback_statistics()
        return {
            "avg_score": stats.get("avg_feedback", 0),
            "total_feedbacks": stats.get("total_feedback", 0)
        }
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des statistiques : {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de la récupération des statistiques : {str(e)}")

def query_rag_for_relevant_documents(question: str, max_documents: int = 5, consider_layout: bool = True) -> List[Dict[str, Any]]:
    global vector_store, embeddings_model
    
    if vector_store is None:
        print("Aucun index vectoriel disponible. Retour de documents vides.")
        return []
    
    # Recherche sémantique avec les embeddings
    try:
        query_embedding = embeddings_model.embed_query(question)
        docs_with_scores = vector_store.similarity_search_with_score(question, k=max_documents)
        
        # Transformer les résultats au format attendu
        results = []
        for doc, score in docs_with_scores:
            # Normaliser le score (transformer la distance en similarité)
            similarity = 1.0 / (1.0 + score)
            
            # Extraire le contenu et les métadonnées
            metadata = doc.metadata
            content = doc.page_content
            
            # Adapter les données pour la réponse
            results.append({
                "page_content": content,
                "metadata": metadata,
                "similarity_score": similarity
            })
        
        return results
    except Exception as e:
        print(f"Erreur lors de la recherche sémantique: {str(e)}")
        return []

def synthetic_layout_info(doc_id: str) -> Dict[str, Any]:
    """Génère des informations de layout synthétiques pour la démonstration"""
    # En production, ces données viendraient d'une analyse OCR réelle
    import random
    
    # Génération de boîtes avec des positions aléatoires mais réalistes
    num_boxes = random.randint(3, 8)
    boxes = []
    for _ in range(num_boxes):
        x0 = random.randint(10, 300)
        y0 = random.randint(10, 400)
        width = random.randint(50, 200)
        height = random.randint(20, 80)
        boxes.append([x0, y0, x0 + width, y0 + height])
    
    return {
        "boxes": boxes,
        "page_width": 600,
        "page_height": 800,
        "doc_id": doc_id,
        "layout_type": "medical_form" if random.random() > 0.5 else "clinical_report"
    }

@app.on_event("shutdown")
def shutdown_event():
    db_manager.close()