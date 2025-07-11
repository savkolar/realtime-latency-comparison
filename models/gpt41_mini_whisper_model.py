import time
import requests
import json
import logging
import io
import base64
from typing import Dict, Any, Tuple, Optional

from azure.identity.aio import DefaultAzureCredential

from models.base_model import BaseModel
from utils.exceptions import ModelGenerationError, ModelInitializationError
from utils.audio_utils import get_mp3_duration
from utils.client import create_azure_openai_client
from utils.metrics import calculate_tokens_per_second, create_error_metrics
from config import (
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    GPT41_MINI_DEPLOYMENT,
    TTS_DEPLOYMENT,
    WHISPER_DEPLOYMENT
)

# Configure logger
logger = logging.getLogger(__name__)


class GPT41MiniWhisperModel(BaseModel):
    """
    GPT-4.1-mini with Whisper TTS model implementation.

    Uses the GPT-4.1-mini deployment for text generation and Whisper for text-to-speech.
    """

    def __init__(self):
        """Initialize the GPT-4.1-mini with Whisper TTS model."""
        super().__init__("GPT-4.1-mini + Whisper + TTS")
        self.gpt41_mini_deployment = GPT41_MINI_DEPLOYMENT or "gpt-4.1-mini"
        self.tts_deployment = TTS_DEPLOYMENT or "tts"
        self.whisper_deployment = WHISPER_DEPLOYMENT or "whisper"

    async def initialize(self) -> None:
        """
        Initialize the Azure OpenAI client.

        Raises:
            ModelInitializationError: If client initialization fails
        """
        try:
            self.client = await create_azure_openai_client(AZURE_OPENAI_API_VERSION)
        except Exception as e:
            raise ModelInitializationError(
                f"Failed to initialize GPT-4.1-mini + Whisper model: {e}") from e

    async def _get_tts_headers(self) -> Dict[str, str]:
        """
        Get the headers for the TTS API request.

        Returns:
            Dictionary of headers

        Raises:
            ModelInitializationError: If getting authentication fails
        """
        headers = {'Content-Type': 'application/json'}

        try:
            if AZURE_OPENAI_API_KEY:
                headers['api-key'] = AZURE_OPENAI_API_KEY
            else:
                # Use Microsoft Entra ID authentication for TTS
                credential = DefaultAzureCredential()
                token = await credential.get_token("https://cognitiveservices.azure.com/.default")
                headers['Authorization'] = f'Bearer {token.token}'

            return headers
        except Exception as e:
            raise ModelInitializationError(
                f"Failed to get TTS authentication: {e}") from e

    async def generate_response_from_audio(self, audio_data: bytes, text_prompt: Optional[str] = None) -> Tuple[str, Dict[str, Any], Optional[bytes]]:
        """
        Generate a response for audio input, with Whisper transcription, GPT-4.1-mini processing, and TTS.

        Args:
            audio_data: The input audio data (WAV format)
            text_prompt: Optional text prompt to accompany the audio (will be added to transcript)

        Returns:
            Tuple containing text response, metrics, and audio data

        Raises:
            ModelGenerationError: If response generation fails
        """
        await self.ensure_initialized()
        start_time = time.time()

        try:
            # Step 1: Transcribe audio with Whisper
            logger.debug(f"Transcribing audio with {self.whisper_deployment}")
            whisper_start_time = time.time()

            # Create a buffer for the audio file
            buffer = io.BytesIO(audio_data)
            buffer.name = "audio.wav"

            # Call Whisper for transcription
            transcription_response = await self.client.audio.transcriptions.create(
                model=self.whisper_deployment,
                file=buffer
            )
            buffer.close()

            transcription = transcription_response.text
            whisper_complete_time = time.time()
            whisper_time = whisper_complete_time - whisper_start_time

            logger.debug(
                f"Transcription completed in {whisper_time:.2f}s: {transcription[:50]}...")

            # Combine transcription with any additional text prompt
            if text_prompt:
                full_prompt = f"{text_prompt}\n\nTranscribed audio: {transcription}"
            else:
                full_prompt = f"Respond to this transcribed audio: {transcription}"

            # Step 2: Generate text response with GPT-4.1-mini
            logger.debug(
                f"Generating text response with {self.gpt41_mini_deployment}")
            gpt_start_time = time.time()

            response = await self.client.chat.completions.create(
                model=self.gpt41_mini_deployment,
                messages=[{"role": "user", "content": full_prompt}],
                stream=False
            )

            text_response = response.choices[0].message.content
            gpt_complete_time = time.time()
            gpt_time = gpt_complete_time - gpt_start_time

            logger.debug(
                f"Text response generated in {gpt_time:.2f}s: {text_response[:50]}...")

            # Step 3: Convert text to speech using TTS
            tts_start_time = time.time()
            logger.debug(
                f"Converting text to speech with {self.tts_deployment}")

            # Get TTS headers with authentication
            headers = await self._get_tts_headers()

            # Prepare TTS API request
            tts_url = f"{AZURE_OPENAI_ENDPOINT}openai/deployments/{self.tts_deployment}/audio/speech?api-version={AZURE_OPENAI_API_VERSION}"
            tts_body = {
                "input": text_response,
                "voice": "nova",
                "response_format": "mp3"
            }

            # Make TTS API request
            tts_response = requests.post(
                tts_url, headers=headers, data=json.dumps(tts_body), timeout=30
            )

            if tts_response.status_code != 200:
                raise ModelGenerationError(
                    f"TTS API error: {tts_response.status_code} - {tts_response.text}"
                )

            # Process audio data
            audio_data_output = tts_response.content
            tts_complete_time = time.time()
            tts_generation_time = tts_complete_time - tts_start_time

            logger.debug(f"Audio generated in {tts_generation_time:.2f}s")

            # Calculate audio duration
            audio_duration = get_mp3_duration(
                audio_data_output) if audio_data_output else 0.0

            # Collect metrics with detailed breakdown
            metrics = {
                "model": self.name,
                # Whisper transcription metrics
                "transcribe_time": whisper_time,

                # Text generation metrics (the core GPT-4.1-mini part)
                "text_generation_time": gpt_time,

                # TTS specific metrics
                "tts_time": tts_generation_time,

                # Combined metrics (total pipeline)
                "processing_time": whisper_time + gpt_time + tts_generation_time,

                # Time until audio would start playing (full pipeline)
                "time_to_audio_start": whisper_time + gpt_time + tts_generation_time,

                # Actual duration of the generated audio
                "audio_duration": audio_duration,

                # Other metrics
                "token_count": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
                "audio_size_bytes": len(audio_data_output) if audio_data_output else 0,
                "audio_input_size_bytes": len(audio_data),
                "transcription_length": len(transcription)
            }

            # Calculate tokens per second based on text generation time only
            metrics["tokens_per_second"] = calculate_tokens_per_second(
                metrics["token_count"], metrics["text_generation_time"]
            )

            logger.info(
                f"Generated GPT-4.1-mini + Whisper response in {metrics['processing_time']:.2f}s (from audio input)")

            return text_response, metrics, audio_data_output

        except Exception as e:
            error_msg = f"Error with GPT-4.1-mini + Whisper (audio input): {str(e)}"
            logger.error(error_msg, exc_info=True)

            # Return error metrics
            error_metrics = create_error_metrics(self.name, e, start_time)
            error_metrics["audio_input_size_bytes"] = len(audio_data)

            return error_msg, error_metrics, None
