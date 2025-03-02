import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import sys
import os

# Ajuster le chemin pour importer les modules du projet
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.chatbot.app import app, get_chatbot, get_db_manager

class TestAPI(unittest.TestCase):
    """Tests pour l'API FastAPI."""

    def setUp(self):
        """Configurer le client de test et les mocks."""
        self.client = TestClient(app)
        
        # Patcher les dépendances
        self.api_patches = [
            patch('src.chatbot.app.get_chatbot'),
            patch('src.chatbot.app.get_db_manager')
        ]
        
        # Démarrer les patches
        self.mock_get_chatbot, self.mock_get_db_manager = [p.start() for p in self.api_patches]
        
        # Configurer les mocks
        self.mock_chatbot = MagicMock()
        self.mock_db = MagicMock()
        
        self.mock_get_chatbot.return_value = self.mock_chatbot
        self.mock_get_db_manager.return_value = self.mock_db
        
    def tearDown(self):
        """Arrêter tous les patches."""
        for patch in self.api_patches:
            patch.stop()
    
    def test_read_root(self):
        """Tester l'endpoint racine."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("message", response.json())
        self.assertIn("version", response.json())
    
    def test_get_answer_success(self):
        """Tester l'endpoint /answer avec succès."""
        # Configurer le mock du chatbot
        self.mock_chatbot.get_answer.return_value = (
            "C'est une réponse test",
            [{"source": "source1", "focus_area": "area1", "similarity_score": 0.9}],
            0.05
        )
        
        # Effectuer la requête
        response = self.client.post(
            "/answer",
            json={"question": "Question test?"}
        )
        
        # Vérifications
        self.assertEqual(response.status_code, 200)
        response_data = response.json()
        self.assertEqual(response_data["answer"], "C'est une réponse test")
        self.assertEqual(len(response_data["sources"]), 1)
        self.assertEqual(response_data["sources"][0]["source"], "source1")
        self.assertEqual(response_data["cost"], 0.05)
        
        # Vérifier que la réponse a été stockée
        self.mock_db.store_query.assert_called_once()
    
    def test_get_answer_empty_question(self):
        """Tester l'endpoint /answer avec une question vide."""
        response = self.client.post(
            "/answer",
            json={"question": ""}
        )
        
        # Vérifications
        self.assertEqual(response.status_code, 422)  # Erreur de validation
        self.assertIn("detail", response.json())
    
    def test_get_answer_error(self):
        """Tester l'endpoint /answer avec une erreur."""
        # Configurer le mock pour lever une exception
        self.mock_chatbot.get_answer.side_effect = Exception("Test error")
        
        # Effectuer la requête
        response = self.client.post(
            "/answer",
            json={"question": "Question test?"}
        )
        
        # Vérifications
        self.assertEqual(response.status_code, 500)
        self.assertIn("detail", response.json())
    
    def test_submit_feedback(self):
        """Tester l'endpoint /feedback."""
        # Effectuer la requête
        response = self.client.post(
            "/feedback",
            json={"question": "Question test?", "feedback_score": 4.5}
        )
        
        # Vérifications
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "success")
        
        # Vérifier que le feedback a été stocké
        self.mock_db.store_feedback.assert_called_once_with("Question test?", 4.5)
    
    def test_get_stats(self):
        """Tester l'endpoint /stats."""
        # Configurer le mock de la base de données
        self.mock_db.get_feedback_statistics.return_value = {
            "avg_feedback": 4.2,
            "total_feedback": 10
        }
        
        # Effectuer la requête
        response = self.client.get("/stats")
        
        # Vérifications
        self.assertEqual(response.status_code, 200)
        stats = response.json()
        self.assertEqual(stats["avg_score"], 4.2)
        self.assertEqual(stats["total_feedbacks"], 10)

if __name__ == '__main__':
    unittest.main()


















