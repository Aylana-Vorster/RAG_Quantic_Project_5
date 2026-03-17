import pytest
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_index_page(client):
    """Test if the home page loads correctly."""
    response = client.get('/')
    assert response.status_code == 200

def test_health_endpoint(client):
    """Test if the health check endpoint responds correctly."""
    response = client.get('/health')
    assert response.status_code == 200
    data = response.get_json()
    assert 'status' in data
    assert 'database_exists' in data
