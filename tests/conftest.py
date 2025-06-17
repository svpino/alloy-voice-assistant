"""Shared pytest fixtures and configuration for all tests."""

import os
import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import Mock, patch

import pytest


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mock environment variables for testing."""
    def _mock_env(**kwargs):
        for key, value in kwargs.items():
            monkeypatch.setenv(key, value)
    return _mock_env


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    with patch('openai.OpenAI') as mock_client:
        mock_instance = Mock()
        mock_client.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_webcam():
    """Mock webcam stream for testing."""
    with patch('cv2.VideoCapture') as mock_capture:
        mock_instance = Mock()
        mock_instance.read.return_value = (True, Mock())
        mock_capture.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_audio():
    """Mock audio components for testing."""
    with patch('pyaudio.PyAudio') as mock_pyaudio:
        mock_instance = Mock()
        mock_pyaudio.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_speech_recognizer():
    """Mock speech recognition for testing."""
    with patch('speech_recognition.Recognizer') as mock_recognizer:
        mock_instance = Mock()
        mock_recognizer.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def sample_config():
    """Provide sample configuration for testing."""
    return {
        "openai_api_key": "test-api-key",
        "model": "gpt-4-vision-preview",
        "temperature": 0.7,
        "max_tokens": 150,
    }


@pytest.fixture
def mock_langchain_components():
    """Mock LangChain components for testing."""
    with patch('langchain_openai.ChatOpenAI') as mock_chat, \
         patch('langchain_community.chat_message_histories.ChatMessageHistory') as mock_history:
        mock_chat_instance = Mock()
        mock_history_instance = Mock()
        mock_chat.return_value = mock_chat_instance
        mock_history.return_value = mock_history_instance
        yield {
            'chat': mock_chat_instance,
            'history': mock_history_instance
        }


@pytest.fixture(autouse=True)
def cleanup_env():
    """Clean up environment after each test."""
    yield
    # Add any cleanup code here if needed
    pass


@pytest.fixture
def mock_dotenv():
    """Mock dotenv loading for testing."""
    with patch('dotenv.load_dotenv') as mock_load:
        mock_load.return_value = True
        yield mock_load