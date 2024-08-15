import whisper
from pyannote.audio import Pipeline
import sys
import argparse
import torch
import ffmpeg
import os
import json
import pickle

def convert_audio(input_file):
    output_file = input_file.rsplit(".", 1)[0] + ".wav"
    print(f"Converting {input_file} to {output_file}...")
    ffmpeg.input(input_file).output(output_file).run(overwrite_output=True)
    return output_file

def transcribe_audio(audio_file, model_name, language=None):
    print(f"Loading Whisper model ({model_name})...")
    model = whisper.load_model(model_name)
    print("Transcribing audio...")
    result = model.transcribe(audio_file, language=language)
    return result

def save_transcription(transcription, filepath):
    with open(filepath, "w") as f:
        json.dump(transcription, f)

def load_transcription(filepath):
    with open(filepath, "r") as f:
        return json.load(f)

def diarize_audio(audio_file, hf_token, num_speakers=None, min_speakers=None, max_speakers=None):
    print("Loading PyAnnote model...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token
    )
    pipeline.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    print("Performing diarization...")
    if num_speakers:
        diarization = pipeline(audio_file, num_speakers=num_speakers)
    else:
        diarization = pipeline(audio_file, min_speakers=min_speakers, max_speakers=max_speakers)
    
    return diarization

def save_diarization(diarization, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(diarization, f)

def load_diarization(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)

def combine_transcription_and_diarization(transcription, diarization):
    segments = []
    for segment in diarization.itertracks(yield_label=True):
        start, end = segment[0].start, segment[0].end
        speaker = segment[1]

        # Extract the words spoken in this segment
        words = [word for word in transcription["segments"] if word["start"] >= start and word["end"] <= end]
        text = " ".join([word["text"] for word in words])

        segments.append(f"{speaker}: {text}")

    return segments

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe and diarize audio.")
    parser.add_argument("audio_file", type=str, help="Path to the audio file.")
    parser.add_argument("--model", type=str, default="base", help="Whisper model to use (tiny, base, small, medium, large).")
    parser.add_argument("--language", type=str, default=None, help="Language spoken in the audio.")
    parser.add_argument("--output_dir", type=str, default="./", help="Directory to save the output transcription.")
    parser.add_argument("--hf_token", type=str, required=True, help="Hugging Face API token for PyAnnote model.")
    parser.add_argument("--num_speakers", type=int, help="Exact number of speakers expected.")
    parser.add_argument("--min_speakers", type=int, default=1, help="Minimum number of speakers expected.")
    parser.add_argument("--max_speakers", type=int, default=5, help="Maximum number of speakers expected.")
    parser.add_argument("-f", "--force", action="store_true", help="Force redoing transcription and/or diarization.")

    args = parser.parse_args()

    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Paths for intermediate files
    converted_audio_file = convert_audio(args.audio_file)
    transcription_file = os.path.join(args.output_dir, f"{os.path.basename(args.audio_file)}_transcription.json")
    diarization_file = os.path.join(args.output_dir, f"{os.path.basename(args.audio_file)}_diarization.pkl")

    # Transcription
    if args.force or not os.path.exists(transcription_file):
        print("Transcribing audio...")
        transcription = transcribe_audio(converted_audio_file, model_name=args.model, language=args.language)
        save_transcription(transcription, transcription_file)
    else:
        print("Loading existing transcription...")
        transcription = load_transcription(transcription_file)

    # Diarization
    if args.force or not os.path.exists(diarization_file):
        print("Performing diarization...")
        diarization = diarize_audio(converted_audio_file, hf_token=args.hf_token, num_speakers=args.num_speakers, min_speakers=args.min_speakers, max_speakers=args.max_speakers)
        save_diarization(diarization, diarization_file)
    else:
        print("Loading existing diarization...")
        diarization = load_diarization(diarization_file)

    # Combine transcription and diarization
    combined_output = combine_transcription_and_diarization(transcription, diarization)

    # Save the result to a text file in the specified output directory
    output_path = os.path.join(args.output_dir, f"{os.path.basename(args.audio_file)}_transcription.txt")
    with open(output_path, "w") as f:
        for segment in combined_output:
            f.write(segment + "\n")

    print(f"Process complete! Transcription saved to {output_path}")