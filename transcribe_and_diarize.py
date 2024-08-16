import whisper
from pyannote.audio import Pipeline
from pyannote.core import Segment
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

def transcribe_audio(audio_file, model_name, language=None, force=False):
    transcription_file = audio_file + "_transcription.json"
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
    diarization_file = audio_file + "_diarization.pkl"
    if not force and os.path.exists(diarization_file):
        print(f"Loading existing diarization from {diarization_file}...")
        with open(diarization_file, "rb") as f:
            return pickle.load(f)

    print("Loading PyAnnote model for speaker diarization...")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)

    # Use GPU acceleration if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline.to(device)

    print("Performing diarization...")
    diarization = pipeline(audio_file, num_speakers=num_speakers)

    # Save diarization to file
    with open(diarization_file, "wb") as f:
        pickle.dump(diarization, f)

    return diarization

def combine_transcription_and_diarization(transcription, diarization):
    segments = []
    for segment in diarization.itertracks(yield_label=True):
        start, end = segment[0].start, segment[0].end
        speaker = segment[1]

        # Extract the words spoken in this segment
        words = [word for word in transcription["segments"] if word["start"] >= start and word["end"] <= end]
        text = " ".join([word["text"] for word in words])

        segments.append({"start": start, "end": end, "label": speaker, "text": text})

    combined_output = [f"{segment['label']}: {segment['text']}" for segment in segments]
    return combined_output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe and diarize audio.")
    parser.add_argument("audio_file", type=str, help="Path to the audio file.")
    parser.add_argument("--model", type=str, default="base", help="Whisper model to use (tiny, base, small, medium, large).")
    parser.add_argument("--language", type=str, default=None, help="Language spoken in the audio.")
    parser.add_argument("--output_dir", type=str, default="./", help="Directory to save the output transcription.")
    parser.add_argument("--hf_token", type=str, required=True, help="Hugging Face API token for PyAnnote model.")
    parser.add_argument("--num_speakers", type=int, help="Exact number of speakers expected.")
    parser.add_argument("-f", "--force", action="store_true", help="Force re-processing of each step.")
    parser.add_argument("--force_diarization", action="store_true", help="Force re-processing of diarization step only.")

    args = parser.parse_args()

    # Convert the audio to WAV format
    converted_audio_file = convert_audio(args.audio_file)

    # Transcribe the audio
    transcription = transcribe_audio(converted_audio_file, model_name=args.model, language=args.language, force=args.force)

    # Perform speaker diarization
    diarization = diarize_audio(converted_audio_file, hf_token=args.hf_token, num_speakers=args.num_speakers, force=args.force or args.force_diarization)

    # Combine transcription and diarization
    combined_output = combine_transcription_and_diarization(transcription, diarization)

    # Output the result
    for segment in combined_output:
        print(segment)

    # Save the result to a text file in the specified output directory
    output_path = f"{args.output_dir}/{os.path.basename(args.audio_file)}_transcription.txt"
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w") as f:
        for segment in combined_output:
            f.write(segment + "\n")

    print(f"Process complete! Transcription saved to {output_path}")