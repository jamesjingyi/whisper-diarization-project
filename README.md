# Whisper and PyAnnote Audio Speaker Diarization

This project integrates OpenAI's Whisper model for transcription and PyAnnote.audio for speaker diarization.

## Setup

1. Install ffmpeg
For macOS:
`brew install ffmpeg`

For Ubuntu:
`sudo apt-get install ffmpeg`

For Windows:
Download from the [official website](https://ffmpeg.org/download.html).

1. Clone the repository:
```
git clone 
cd whisper-diarization-project
```

2. Create a virtual environment and install dependencies:
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. Place your audio files in the `data/` directory.

## Usage

To transcribe and diarize an audio file, use the `transcribe_and_diarize.py` script. You can specify the Whisper model, language, and output directory using command-line flags.

### Basic usage

This assumes you have done steps 1-3 in the 'Setup' section above.

1. Place any video or audio files into the `data` folder (these will be converted with ffmpeg)

2. (If not already running) Activate the virtual environment

`source venv/bin/activate`

3. Run the script. At its most basic, it would look like this:

`python transcribe_and_diarize.py`

4. This processes any of the files in `data` then outputs raw transcripts using Whisper in `transcriptions/raw_transcripts` and further diarizations in `transcriptions/diarized`

The script intelligently avoids re-processing if the output files already exist (see flags to override below).

### Advanced Usage with Flags

Model: Specify the Whisper model to use. Options include tiny, base, small, medium, large. Default is base.
`--model medium`

Language: Specify the language spoken in the audio. Default is None, which means Whisper will auto-detect the language.
`--language en`

Output Directory: Specify the directory where the transcription file will be saved. Default is the current directory (./).
`--output_dir ./transcriptions`

Speaker numbers: If you know the specific amount of speakers, you can use `--num_speakers`. You can also set min and max using `--min_speakers` and `--max_speakers`.

By default, the script avoids reprocessing each step (transcription and diarization), so that you can add new files to the `data` folder without worrying about removing old ones. You can override this however using `-f` or `--force` to reprocess both transcription and diarization, or just re-diarization using the `--force_diarization` flag.

### Example Command

This command transcribes and diarizes an audio file using the medium Whisper model, sets the audio to English, saves the output in the `transcriptions` directory, and sets the minimum speakers to 2, and maximum speakers to 4:
`python transcribe_and_diarize.py data/your_audio_file.wav --model medium --language en --output_dir ./transcriptions --min_speakers 2 --max_speakers 4`

## Output

The output will be printed to the console and saved to a text file in the main project folder (unless you specify one). The file will be named using the original audio file’s name with `_transcription.txt` appended.

Example:
If your audio file is `your_audio_file.wav`, the output file will be `your_audio_file_transcription.txt` in the specified output directory.