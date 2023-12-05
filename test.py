import unittest
from unittest.mock import MagicMock, patch
import pandas as pd  # Ajout de l'importation de pandas
from dashboard import request_prediction

class TestDashboard(unittest.TestCase):

    @patch("dashboard.requests.request")
    def test_request_prediction(self, mock_request):
        # Définir les données de test
        model_uri = "http://example.com/model"
        # Utiliser un DataFrame pour les données
        data = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})

        # Définir la réponse simulée de l'API MLflow
        expected_response = {"predictions": [0.8, 0.2, 0.6]}
        mock_request.return_value.status_code = 200
        mock_request.return_value.json.return_value = expected_response

        # Appeler la fonction à tester
        result = request_prediction(model_uri, data)

        # S'assurer que la requête a été correctement construite
        mock_request.assert_called_with(
            method="POST",
            headers={"Content-Type": "application/json"},
            url=model_uri,
            json={"instances": [{"feature1": 1, "feature2": 4}, {"feature1": 2, "feature2": 5}, {"feature1": 3, "feature2": 6}]}
        )

        # Assurez-vous que la réponse est correcte
        self.assertEqual(result, expected_response)

if __name__ == "__main__":
    unittest.main()
