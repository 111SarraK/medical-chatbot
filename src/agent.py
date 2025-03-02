# src/chatbot/agent.py
from typing import Dict, List, Any, Tuple, Annotated, TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import json
import requests
import os

# Configuration de l'API pour le dataset LayoutLM
LAYOUTLM_API_HOST = os.getenv("LAYOUTLM_API_HOST", "http://0.0.0.0:8181")
LAYOUTLM_API_ENDPOINT_RETRIEVE = os.path.join(LAYOUTLM_API_HOST, "retrieve_documents")
LAYOUTLM_API_ENDPOINT_ANSWER = os.path.join(LAYOUTLM_API_HOST, "generate_answer")

# Définition des états du graphe
class AgentState(TypedDict):
    messages: List[Any]  # Messages échangés
    context: List[Dict]  # Documents contextuels
    current_question: str  # Question actuelle
    need_more_info: bool  # Besoin de plus d'informations
    final_answer: str  # Réponse finale
    sources: List[Dict]  # Sources utilisées
    confidence: float  # Niveau de confiance

# Fonctions pour les nœuds du graphe
def retrieve_documents(state: AgentState) -> AgentState:
    """Récupère les documents pertinents pour la question à partir du dataset LayoutLM."""
    question = state["current_question"]
    
    try:
        # Appel à l'API pour récupérer les documents pertinents
        response = requests.post(
            LAYOUTLM_API_ENDPOINT_RETRIEVE,
            json={"question": question},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            documents = data.get("documents", [])
            
            # Formatage des documents pour le contexte
            context = []
            sources = []
            for i, doc in enumerate(documents):
                context.append({
                    "content": doc.get("content", ""),
                    "metadata": doc.get("metadata", {})
                })
                sources.append({
                    "source": doc.get("metadata", {}).get("source", f"Document {i+1}"),
                    "focus_area": doc.get("metadata", {}).get("focus_area", "Non spécifié"),
                    "similarity_score": doc.get("metadata", {}).get("score", 0.0)
                })
            
            # Mise à jour de l'état
            state["context"] = context
            state["sources"] = sources
        else:
            st.error(f"Erreur lors de la récupération des documents : {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"Erreur de connexion à l'API : {str(e)}")
    
    return state

def determine_answer_approach(state: AgentState) -> str:
    """Détermine si les informations sont suffisantes ou s'il faut poser des questions supplémentaires."""
    llm = ChatGoogleGenerativeAI(model="gemini-pro")
    
    # Préparation du prompt
    question = state["current_question"]
    context = json.dumps(state["context"], ensure_ascii=False)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Vous êtes un assistant médical chargé d'évaluer les informations disponibles pour répondre à une question. "
                  "Déterminez si le contexte fourni est suffisant pour répondre à la question ou si vous avez besoin de plus d'informations. "
                  "Répondez uniquement par 'suffisant' ou 'insuffisant'."),
        ("user", f"Question: {question}\n\nContexte disponible: {context}")
    ])
    
    # Exécution
    response = llm.invoke(prompt.format())
    
    # Analyse de la réponse
    need_more = "insuffisant" in response.content.lower()
    state["need_more_info"] = need_more
    
    # Décision de routage
    return "ask_follow_up" if need_more else "generate_answer"

def ask_follow_up(state: AgentState) -> AgentState:
    """Génère une question de suivi pour obtenir plus d'informations."""
    llm = ChatGoogleGenerativeAI(model="gemini-pro")
    
    # Préparation du prompt
    question = state["current_question"]
    context = json.dumps(state["context"], ensure_ascii=False)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Vous êtes un assistant médical qui a besoin de plus d'informations pour répondre à une question. "
                  "Formulez une question de suivi concise et précise pour obtenir les informations manquantes."),
        ("user", f"Question initiale: {question}\n\nContexte disponible: {context}")
    ])
    
    # Exécution
    response = llm.invoke(prompt.format())
    
    # Mise à jour de l'état
    follow_up_question = response.content
    state["messages"].append(AIMessage(content=follow_up_question))
    
    return state

def generate_answer(state: AgentState) -> AgentState:
    """Génère une réponse finale basée sur le contexte."""
    llm = ChatGoogleGenerativeAI(model="gemini-pro")
    
    # Préparation du prompt
    question = state["current_question"]
    context_str = "\n\n".join([f"Document {i+1}:\n{doc['content']}" for i, doc in enumerate(state["context"])])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Vous êtes un assistant médical expert. Utilisez uniquement les informations fournies dans les documents "
                  "pour répondre à la question. Si les documents ne contiennent pas suffisamment d'informations, "
                  "indiquez-le clairement. Citez vos sources en référençant les numéros des documents."),
        ("user", f"Question: {question}\n\nDocuments:\n{context_str}")
    ])
    
    # Exécution
    response = llm.invoke(prompt.format())
    
    # Analyse de la réponse pour le niveau de confiance
    confidence = 0.8  # Valeur par défaut
    if "je ne suis pas sûr" in response.content.lower() or "informations insuffisantes" in response.content.lower():
        confidence = 0.5
    
    # Mise à jour de l'état
    state["final_answer"] = response.content
    state["confidence"] = confidence
    state["messages"].append(AIMessage(content=response.content))
    
    return state

def handle_human_response(state: AgentState) -> str:
    """Traite la réponse humaine et détermine l'étape suivante."""
    if not state["messages"]:
        return END
    
    last_message = state["messages"][-1]
    if isinstance(last_message, HumanMessage):
        # Si c'est une réponse à une question de suivi
        if state["need_more_info"]:
            # Mettre à jour le contexte avec la nouvelle information
            state["context"].append({
                "content": last_message.content,
                "metadata": {"source": "Réponse utilisateur"}
            })
            return "generate_answer"
        else:
            # Nouvelle question
            state["current_question"] = last_message.content
            state["context"] = []
            state["sources"] = []
            state["final_answer"] = ""
            return "retrieve_documents"
    
    return END

# Définition du graphe
def create_medical_agent():
    """Crée et renvoie l'agent LangGraph pour le chatbot médical."""
    # Initialisation du graphe
    workflow = StateGraph(AgentState)
    
    # Définition des nœuds
    workflow.add_node("retrieve_documents", retrieve_documents)
    workflow.add_node("determine_answer_approach", determine_answer_approach)
    workflow.add_node("ask_follow_up", ask_follow_up)
    workflow.add_node("generate_answer", generate_answer)
    workflow.add_node("handle_human_response", handle_human_response)
    
    # Définition des arêtes
    workflow.add_edge("retrieve_documents", "determine_answer_approach")
    workflow.add_conditional_edges(
        "determine_answer_approach",
        lambda x: "ask_follow_up" if x == "ask_follow_up" else "generate_answer"
    )
    workflow.add_edge("ask_follow_up", "handle_human_response")
    workflow.add_edge("generate_answer", END)
    
    # Définition de l'état initial
    workflow.set_entry_point("handle_human_response")
    
    # Compilation du graphe
    return workflow.compile()

# Classe wrapper pour faciliter l'utilisation de l'agent
class MedicalAgent:
    def __init__(self):
        self.agent = create_medical_agent()
        self.state = {
            "messages": [],
            "context": [],
            "current_question": "",
            "need_more_info": False,
            "final_answer": "",
            "sources": [],
            "confidence": 0.0
        }
    
    def process_message(self, message: str) -> Dict[str, Any]:
        """Traite un message de l'utilisateur et renvoie la réponse."""
        # Ajouter le message à l'état
        self.state["messages"].append(HumanMessage(content=message))
        self.state["current_question"] = message
        
        # Exécuter le graphe
        result = self.agent.invoke(self.state)
        
        # Mettre à jour l'état
        self.state = result
        
        # Préparer la réponse
        return {
            "answer": self.state["final_answer"],
            "sources": self.state["sources"],
            "need_more_info": self.state["need_more_info"],
            "confidence": self.state["confidence"]
        }