import os
import torch
import logging
from config import MODEL_NAME, HF_TOKEN
import whisperx
from pyannote.audio import Pipeline

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def download_models():
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.environ["WHISPERX_CACHE_DIR"] = os.path.join(PROJECT_ROOT, "models", "whisper")
    os.environ["PYANNOTE_CACHE"] = os.path.join(PROJECT_ROOT, "models", "pyannote", "pretrained")

    # Create cache directories
    os.makedirs(os.environ["WHISPERX_CACHE_DIR"], exist_ok=True)
    os.makedirs(os.environ["PYANNOTE_CACHE"], exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Downloading Whisper model...")
    whisperx.load_model(MODEL_NAME, device=device)

    logger.info("Downloading alignment model...")
    whisperx.load_align_model("en", device=device)  # Base English model

    logger.info("Downloading pyannote diarization model...")
    Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=HF_TOKEN
    )
    logger.info("All models downloaded successfully!")


if __name__ == "__main__":
    download_models()
