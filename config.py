import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env

# Configuration
MODEL_NAME = "large-v3"
HF_TOKEN = os.getenv("HF_TOKEN", "your_hugging_face_token")  # Use .env for security
BATCH_SIZE = 16
MIN_SPEAKERS = 1
MAX_SPEAKERS = 10
COMPUTE_TYPE = "float16"  # Use "int8" for CPU-only