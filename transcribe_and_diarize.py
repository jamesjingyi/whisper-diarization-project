import whisper
import torch
import os
import json
import pickle
import yaml
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from tqdm import tqdm
from pyannote.audio import Pipeline

class SimpleProgressHook:
    def __init__(self):
        self.progress = 0

    def __enter__(self):
        self.progress = 0
        return self

    def __exit__(self, *args):
        pass

    def __call__(self, step_name, step_artifact, file=None, total=None, completed=None):
        if completed is None:
            completed = total = 1
        progress_percent = int(completed / total * 100)
        print(f'progress {step_name} {progress_percent}', flush=True)

def sanitize_filename(filename):
    """Remove spaces and return a sanitized version of the filename."""
    return filename.replace(" ", "_")

def restore_filename(sanitized_filename, original_filename):
    """Restore the original filename format, adding the appropriate appendices."""
    base, ext = os.path.splitext(original_filename)
    return f"{base}_{sanitized_filename}{ext}"

def convert_audio(input_file, output_file):
    if os.path.exists(output_file):
        print(f"Audio already converted: {output_file}")
        return output_file

    print(f"Converting {input_file} to {output_file}...")
    try:
        subprocess.run(
            ["ffmpeg", "-i", input_file, "-ar", "16000", "-ac", "1", output_file],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
    except subprocess.CalledProcessError as e:
        print(f"Error occurred during ffmpeg conversion: {e.stderr.decode()}")
        raise

    return output_file

def transcribe_audio(audio_file, model_name, language=None, force=False):
    transcription_file = os.path.join("transcriptions/raw_transcripts", sanitize_filename(os.path.basename(audio_file)) + "_transcription.json")
    if not force and os.path.exists(transcription_file):
        print(f"Loading existing transcription from {transcription_file}...")
        with open(transcription_file, "r") as f:
            return json.load(f)

    print(f"Loading Whisper model ({model_name})...")
    model = whisper.load_model(model_name)
    print("Transcribing audio...")
    result = model.transcribe(audio_file, language=language)

    # Save transcription to file
    with open(transcription_file, "w") as f:
        json.dump(result, f)

    return result

def diarize_audio(audio_file, hf_token, num_speakers=None, force=False):
    diarization_file = os.path.join("transcriptions/diarized", sanitize_filename(os.path.basename(audio_file)) + "_diarization.yaml")
    if not force and os.path.exists(diarization_file):
        print(f"Loading existing diarization from {diarization_file}...")
        with open(diarization_file, "r") as f:
            return yaml.safe_load(f)

    print("Performing diarization...")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline.to(device)

    with SimpleProgressHook() as hook:
        diarization = pipeline(audio_file, num_speakers=num_speakers, hook=hook)

    seg_list = []
    for segment, _, label in diarization.itertracks(yield_label=True):
        seg_list.append({'start': int(segment.start * 1000), 
                         'end': int(segment.end * 1000),
                         'label': label})
    
    # Save diarization to file
    with open(diarization_file, "w") as f:
        yaml.safe_dump(seg_list, f)

    return seg_list

def combine_transcription_and_diarization(transcription, diarization):
    segments = []
    for segment in diarization:
        start, end = segment['start'], segment['end']
        speaker = segment['label']

        # Extract the words spoken in this segment
        words = [word for word in transcription["segments"] if word["start"] >= start and word["end"] <= end]
        text = " ".join([word["text"] for word in words])

        segments.append({"start": start, "end": end, "label": speaker, "text": text})

    combined_output = [f"{segment['label']}: {segment['text']}" for segment in segments]
    return combined_output

def process_files_in_data_folder(args):
    data_folder = "data"
    output_folder = "transcriptions"
    raw_transcripts_folder = os.path.join(output_folder, "raw_transcripts")
    diarized_folder = os.path.join(output_folder, "diarized")

    # Ensure the output directories exist
    os.makedirs(raw_transcripts_folder, exist_ok=True)
    os.makedirs(diarized_folder, exist_ok=True)

    with TemporaryDirectory() as tmpdir:
        for file_name in os.listdir(data_folder):
            if not file_name.lower().endswith((".mp3", ".wav", ".mp4", ".m4a", ".flac", ".ogg")):
                continue

            original_file_path = os.path.join(data_folder, file_name)
            sanitized_file_name = sanitize_filename(file_name)
            tmp_file_path = os.path.join(tmpdir, sanitized_file_name)

            # Copy the file to the temp directory
            shutil.copy2(original_file_path, tmp_file_path)

            print(f"Processing file: {tmp_file_path}")

            # Convert the audio to WAV format
            converted_audio_file = convert_audio(tmp_file_path, os.path.join(tmpdir, sanitized_file_name.rsplit(".", 1)[0] + ".wav"))

            # Transcribe the audio
            transcription = transcribe_audio(converted_audio_file, model_name=args.model, language=args.language, force=args.force)

            # Perform speaker diarization
            diarization = diarize_audio(converted_audio_file, hf_token=args.hf_token, num_speakers=args.num_speakers, force=args.force or args.force_diarization)

            # Combine transcription and diarization
            combined_output = combine_transcription_and_diarization(transcription, diarization)

            # Save the combined output to a text file in the diarized folder
            restored_filename = restore_filename(sanitized_file_name, file_name)
            output_path = os.path.join(diarized_folder, f"{restored_filename}_transcription.txt")
            with open(output_path, "w") as f:
                for segment in combined_output:
                    f.write(segment + "\n")

            print(f"Process complete! Transcription saved to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Transcribe and diarize audio files in the data folder.")
    parser.add_argument("--model", type=str, default="base", help="Whisper model to use (tiny, base, small, medium, large).")
    parser.add_argument("--language", type=str, default=None, help="Language spoken in the audio.")
    parser.add_argument("--hf_token", type=str, required=True, help="Hugging Face API token for PyAnnote model.")
    parser.add_argument("--num_speakers", type=int, help="Exact number of speakers expected.")
    parser.add_argument("-f", "--force", action="store_true", help="Force re-processing of each step.")
    parser.add_argument("--force_diarization", action="store_true", help="Force re-processing of diarization step only.")

    args = parser.parse_args()

    # Process all files in the data folder
    process_files_in_data_folder(args)
