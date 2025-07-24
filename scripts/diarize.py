import os
import glob
import logging
import torch
import whisperx
from config import MODEL_NAME, HF_TOKEN, BATCH_SIZE, MIN_SPEAKERS, MAX_SPEAKERS, COMPUTE_TYPE

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def diarize_audio(audio_files, output_dir):
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    logger.info(f"Using device: {device} ({torch_dtype})")

    try:
        # Load models
        model = whisperx.load_model(
            MODEL_NAME,
            device=device,
            compute_type=COMPUTE_TYPE,
            asr_options={"batch_size": BATCH_SIZE}
        )

        diarize_model = whisperx.DiarizationPipeline(
            use_auth_token=HF_TOKEN,
            device=device,
            min_speakers=MIN_SPEAKERS,
            max_speakers=MAX_SPEAKERS
        )

        for audio_file in audio_files:
            try:
                logger.info(f"Processing: {os.path.basename(audio_file)}")

                # Transcribe audio
                transcript = model.transcribe(audio_file)

                # Align output
                model_a, metadata = whisperx.load_align_model(
                    language_code=transcript["language"],
                    device=device
                )
                aligned = whisperx.align(
                    transcript["segments"],
                    model_a,
                    metadata,
                    audio_file,
                    device,
                    return_char_alignments=False
                )

                # Perform diarization
                diarization = diarize_model(audio_file)

                # Assign speakers
                result = whisperx.assign_word_speakers(diarization, aligned)

                # Save results
                base_name = os.path.splitext(os.path.basename(audio_file))[0]
                output_path = os.path.join(output_dir, f"{base_name}_diarization.txt")

                with open(output_path, "w", encoding="utf-8") as f:
                    for seg in result["segments"]:
                        speaker = seg.get("speaker", "UNKNOWN")
                        f.write(f"[{seg['start']:.2f}-{seg['end']:.2f}] {speaker}: {seg['text']}\n")

                logger.info(f"Saved results to: {output_path}")

            except Exception as e:
                logger.error(f"Error processing {audio_file}: {str(e)}")

    except Exception as e:
        logger.critical(f"Critical error: {str(e)}")
        raise


if __name__ == "__main__":
    # Set cache locations
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.environ["WHISPERX_CACHE_DIR"] = os.path.join(PROJECT_ROOT, "models", "whisper")
    os.environ["PYANNOTE_CACHE"] = os.path.join(PROJECT_ROOT, "models", "pyannote", "pretrained")

    # Create directories
    input_dir = os.path.join(PROJECT_ROOT, "input")
    output_dir = os.path.join(PROJECT_ROOT, "output")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Find audio files
    audio_files = []
    for ext in ["*.wav", "*.mp3", "*.flac"]:
        audio_files.extend(glob.glob(os.path.join(input_dir, ext)))

    if not audio_files:
        logger.warning(f"No audio files found in {input_dir}")
    else:
        logger.info(f"Found {len(audio_files)} audio files")
        diarize_audio(audio_files, output_dir)
