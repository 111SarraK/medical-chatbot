import os
from typing import List, Dict, Any, Optional
import sqlalchemy
from sqlalchemy import create_engine, Table, Column, String, Float, JSON, MetaData, Integer, DateTime, func, text
from google.cloud.sql.connector import Connector
import pandas as pd
import logging
import json
from datetime import datetime

# Configuration du logger
logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, instance_connection_name: str, db_user: str, db_pass: str, db_name: str):
        """
        Initialize the database manager for Cloud SQL.
        
        Args:
            instance_connection_name: Cloud SQL instance connection name (project:region:instance)
            db_user: Database username
            db_pass: Database password
            db_name: Database name
        """
        self.instance_connection_name = instance_connection_name
        self.db_user = db_user
        self.db_pass = db_pass
        self.db_name = db_name
        self.engine = None
        self.metadata = MetaData()
        self.logger = logger
        
        # Définition des tables
        self.queries = Table(
            'queries', self.metadata,
            Column('id', Integer, primary_key=True, autoincrement=True),
            Column('question', String, nullable=False),
            Column('answer', String, nullable=False),
            Column('sources', JSON),
            Column('similarity_scores', JSON),
            Column('feedback_score', Float, nullable=True),
            Column('timestamp', DateTime, default=func.now()),
            Column('cost', Float, default=0.0),
            Column('language', String, default='Français')
        )
        
        self.documents = Table(
            'documents', self.metadata,
            Column('id', Integer, primary_key=True, autoincrement=True),
            Column('source', String, nullable=False, unique=True),
            Column('focus_area', String, nullable=False),
            Column('content', String, nullable=False),
            Column('processed', Integer, default=0)  # 0=non traité, 1=traité pour vectorisation
        )
        
        self.vector_store_docs = Table(
            'vector_store_docs', self.metadata,
            Column('id', Integer, primary_key=True, autoincrement=True),
            Column('document_id', Integer, nullable=False),
            Column('chunk_id', Integer, nullable=False),
            Column('chunk_content', String, nullable=False),
            Column('metadata', JSON)
        )
    
    def initialize_connection(self):
        """Initialize connection to Cloud SQL."""
        try:
            connector = Connector()
            
            def getconn():
                conn = connector.connect(
                    self.instance_connection_name,
                    "pg8000",
                    user=self.db_user,
                    password=self.db_pass,
                    db=self.db_name
                )
                return conn
                
            self.engine = create_engine(
                "postgresql+pg8000://",
                creator=getconn
            )
            
            # Create tables if they don't exist
            self.metadata.create_all(self.engine)
            self.logger.info("Connexion à la base de données établie avec succès.")
            return True
        except Exception as e:
            self.logger.error(f"Erreur lors de la connexion à la base de données : {e}")
            raise
    
    def check_connection(self) -> bool:
        """Check if the database connection is active."""
        try:
            if not self.engine:
                self.logger.warning("La connexion à la base de données n'a pas été initialisée.")
                return False
                
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                self.logger.info("La connexion à la base de données est active.")
                return True
        except Exception as e:
            self.logger.error(f"La connexion à la base de données a échoué : {e}")
            return False
    
    def store_document(self, source: str, focus_area: str, content: str) -> bool:
        """
        Store a medical document in the database.
        
        Args:
            source: Document source (unique identifier)
            focus_area: Medical focus area of the document
            content: Document content
            
        Returns:
            Boolean indicating success
        """
        try:
            if not self.engine:
                self.initialize_connection()
                
            # Vérifier si le document existe déjà
            with self.engine.connect() as conn:
                result = conn.execute(
                    text("SELECT id FROM documents WHERE source = :source"),
                    {"source": source}
                )
                existing = result.fetchone()
                
                if existing:
                    # Mettre à jour le document existant
                    conn.execute(
                        text("""
                            UPDATE documents 
                            SET focus_area = :focus_area, content = :content, processed = 0 
                            WHERE source = :source
                        """),
                        {"source": source, "focus_area": focus_area, "content": content}
                    )
                    conn.commit()
                    self.logger.info(f"Document mis à jour : {source}")
                else:
                    # Insérer un nouveau document
                    conn.execute(
                        text("""
                            INSERT INTO documents (source, focus_area, content, processed) 
                            VALUES (:source, :focus_area, :content, 0)
                        """),
                        {"source": source, "focus_area": focus_area, "content": content}
                    )
                    conn.commit()
                    self.logger.info(f"Document ajouté : {source}")
                
                return True
        except Exception as e:
            self.logger.error(f"Erreur lors du stockage du document : {e}")
            return False
    
    def bulk_store_documents(self, documents: List[Dict[str, str]]) -> Dict[str, int]:
        """
        Store multiple documents in bulk.
        
        Args:
            documents: List of document dictionaries with 'source', 'focus_area', and 'content' keys
            
        Returns:
            Dictionary with counts of inserted and updated documents
        """
        if not self.engine:
            self.initialize_connection()
            
        inserted = 0
        updated = 0
        failed = 0
        
        for doc in documents:
            try:
                source = doc.get('source')
                focus_area = doc.get('focus_area', 'General')
                content = doc.get('content', '')
                
                if not source or not content:
                    self.logger.warning(f"Document ignoré: source ou contenu manquant - {source}")
                    failed += 1
                    continue
                
                result = self.store_document(source, focus_area, content)
                if result:
                    inserted += 1
                else:
                    failed += 1
            except Exception as e:
                self.logger.error(f"Erreur lors du traitement du document {doc.get('source')}: {e}")
                failed += 1
        
        return {
            "inserted": inserted,
            "updated": updated,
            "failed": failed,
            "total": len(documents)
        }
    
    def load_dataset_from_csv(self, csv_path: str) -> bool:
        """
        Charge un dataset CSV dans la base de données.
        
        Args:
            csv_path: Chemin vers le fichier CSV.
            
        Returns:
            Boolean indiquant si le chargement a réussi.
        """
        try:
            # Charger le CSV dans un DataFrame
            df = pd.read_csv(csv_path)
            
            # Convertir le DataFrame en une liste de dictionnaires
            documents = df.to_dict('records')
            
            # Stocker les documents dans la base de données
            result = self.bulk_store_documents(documents)
            
            logger.info(f"Dataset chargé avec succès : {result}")
            return True
        except Exception as e:
            logger.error(f"Erreur lors du chargement du dataset : {e}")
            return False
    
    def get_all_documents(self) -> List[Dict]:
        """
        Retrieve all medical documents from the database.
        
        Returns:
            List of document dictionaries
        """
        try:
            if not self.engine:
                self.initialize_connection()
                
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT id, source, focus_area, content FROM documents"))
                documents = [
                    {
                        "id": row[0],
                        "source": row[1],
                        "focus_area": row[2],
                        "content": row[3]
                    }
                    for row in result
                ]
                
                self.logger.info(f"{len(documents)} documents récupérés avec succès.")
                return documents
        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération des documents : {e}")
            return []
    
    def get_unprocessed_documents(self, limit: int = 100) -> List[Dict]:
        """
        Retrieve documents that haven't been processed for vectorization yet.
        
        Args:
            limit: Maximum number of documents to retrieve
            
        Returns:
            List of unprocessed document dictionaries
        """
        try:
            if not self.engine:
                self.initialize_connection()
                
            with self.engine.connect() as conn:
                result = conn.execute(
                    text("""
                        SELECT id, source, focus_area, content 
                        FROM documents 
                        WHERE processed = 0
                        LIMIT :limit
                    """),
                    {"limit": limit}
                )
                
                documents = [
                    {
                        "id": row[0],
                        "source": row[1],
                        "focus_area": row[2],
                        "content": row[3]
                    }
                    for row in result
                ]
                
                self.logger.info(f"{len(documents)} documents non traités récupérés.")
                return documents
        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération des documents non traités : {e}")
            return []
    
    def mark_documents_as_processed(self, document_ids: List[int]) -> bool:
        """
        Mark documents as processed for vectorization.
        
        Args:
            document_ids: List of document IDs to mark as processed
            
        Returns:
            Boolean indicating success
        """
        try:
            if not self.engine or not document_ids:
                return False
                
            with self.engine.connect() as conn:
                for doc_id in document_ids:
                    conn.execute(
                        text("UPDATE documents SET processed = 1 WHERE id = :id"),
                        {"id": doc_id}
                    )
                conn.commit()
                
                self.logger.info(f"{len(document_ids)} documents marqués comme traités.")
                return True
        except Exception as e:
            self.logger.error(f"Erreur lors du marquage des documents comme traités : {e}")
            return False
    
    def store_query(self, question: str, answer: str, sources: List[str], 
                   similarity_scores: List[float], cost: float = 0.0, 
                   language: str = "Français") -> bool:
        """
        Store a query and its response in the database.
        
        Args:
            question: User's question
            answer: Generated answer
            sources: List of document sources used for the answer
            similarity_scores: List of similarity scores for each source
            cost: API call cost
            language: Response language
            
        Returns:
            Boolean indicating success
        """
        try:
            if not self.engine:
                self.initialize_connection()
            
            # Préparer les données JSON
            sources_json = json.dumps(sources)
            scores_json = json.dumps(similarity_scores)
            
            with self.engine.connect() as conn:
                conn.execute(
                    text("""
                        INSERT INTO queries 
                        (question, answer, sources, similarity_scores, cost, language, timestamp) 
                        VALUES 
                        (:question, :answer, :sources, :scores, :cost, :language, :timestamp)
                    """),
                    {
                        "question": question,
                        "answer": answer,
                        "sources": sources_json,
                        "scores": scores_json,
                        "cost": cost,
                        "language": language,
                        "timestamp": datetime.now()
                    }
                )
                conn.commit()
                
                self.logger.info(f"Requête stockée avec succès : {question[:50]}...")
                return True
        except Exception as e:
            self.logger.error(f"Erreur lors du stockage de la requête : {e}")
            return False
    
    def store_feedback(self, question: str, feedback_score: float) -> bool:
        """
        Store user feedback for a query.
        
        Args:
            question: The question for which feedback is provided
            feedback_score: Feedback score (0.0 to 1.0)
            
        Returns:
            Boolean indicating success
        """
        try:
            if not self.engine:
                self.initialize_connection()
                
            with self.engine.connect() as conn:
                # Trouver la dernière requête correspondante
                result = conn.execute(
                    text("""
                        SELECT id FROM queries 
                        WHERE question = :question 
                        ORDER BY timestamp DESC 
                        LIMIT 1
                    """),
                    {"question": question}
                )
                query = result.fetchone()
                
                if query:
                    # Mettre à jour le feedback
                    conn.execute(
                        text("UPDATE queries SET feedback_score = :score WHERE id = :id"),
                        {"score": feedback_score, "id": query[0]}
                    )
                    conn.commit()
                    self.logger.info(f"Feedback stocké avec succès pour : {question[:50]}...")
                    return True
                else:
                    self.logger.warning(f"Aucune requête trouvée pour le feedback : {question[:50]}...")
                    return False
        except Exception as e:
            self.logger.error(f"Erreur lors du stockage du feedback : {e}")
            return False
    
    def get_feedback_statistics(self) -> Dict:
        """
        Get statistics about user feedback.
        
        Returns:
            Dictionary with feedback statistics
        """
        try:
            if not self.engine:
                self.initialize_connection()
                
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT 
                        COUNT(*) as total_queries,
                        AVG(feedback_score) as avg_feedback,
                        COUNT(feedback_score) as total_feedback,
                        COUNT(*) - COUNT(feedback_score) as no_feedback_count
                    FROM queries
                """))
                
                stats = dict(zip(result.keys(), result.fetchone()))
                
                # Calcul des pourcentages de satisfaction
                if stats.get("total_feedback", 0) > 0:
                    good_feedback = conn.execute(text("""
                        SELECT COUNT(*) FROM queries 
                        WHERE feedback_score >= 0.7
                    """)).scalar()
                    
                    neutral_feedback = conn.execute(text("""
                        SELECT COUNT(*) FROM queries 
                        WHERE feedback_score >= 0.3 AND feedback_score < 0.7
                    """)).scalar()
                    
                    poor_feedback = conn.execute(text("""
                        SELECT COUNT(*) FROM queries 
                        WHERE feedback_score < 0.3 AND feedback_score IS NOT NULL
                    """)).scalar()
                    
                    stats["good_feedback_percent"] = round(good_feedback / stats["total_feedback"] * 100, 2)
                    stats["neutral_feedback_percent"] = round(neutral_feedback / stats["total_feedback"] * 100, 2)
                    stats["poor_feedback_percent"] = round(poor_feedback / stats["total_feedback"] * 100, 2)
                
                self.logger.info("Statistiques des feedbacks récupérées avec succès.")
                return stats
        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération des statistiques : {e}")
            return {
                "total_queries": 0,
                "avg_feedback": 0,
                "total_feedback": 0,
                "no_feedback_count": 0
            }
    
    def get_recent_queries(self, limit: int = 10) -> List[Dict]:
        """
        Get the most recent queries and their answers.
        
        Args:
            limit: Maximum number of queries to retrieve
            
        Returns:
            List of query dictionaries
        """
        try:
            if not self.engine:
                self.initialize_connection()
                
            with self.engine.connect() as conn:
                result = conn.execute(
                    text("""
                        SELECT id, question, answer, feedback_score, timestamp 
                        FROM queries 
                        ORDER BY timestamp DESC 
                        LIMIT :limit
                    """),
                    {"limit": limit}
                )
                
                queries = [dict(row) for row in result]
                return queries
        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération des requêtes récentes : {e}")
            return []
    
    def export_feedback_to_dataframe(self) -> pd.DataFrame:
        """
        Export feedback data to a pandas DataFrame for analysis.
        
        Returns:
            DataFrame with feedback data
        """
        try:
            if not self.engine:
                self.initialize_connection()
                
            query = """
                SELECT 
                    question, 
                    answer, 
                    feedback_score, 
                    timestamp, 
                    cost,
                    language
                FROM queries 
                WHERE feedback_score IS NOT NULL
                ORDER BY timestamp DESC
            """
            
            df = pd.read_sql(query, self.engine)
            self.logger.info(f"Données de feedback exportées avec succès : {len(df)} entrées.")
            return df
        except Exception as e:
            self.logger.error(f"Erreur lors de l'exportation des données de feedback : {e}")
            return pd.DataFrame()
    
    def close(self):
        """Close the database connection."""
        if hasattr(self, 'engine') and self.engine:
            self.engine.dispose()
            self.logger.info("Connexion à la base de données fermée.")