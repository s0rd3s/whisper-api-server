# Whisper API Service

A local, OpenAI-compatible speech recognition API service using the Whisper model. This service provides a straightforward way to transcribe audio files in various formats with high accuracy and is designed to be compatible with the OpenAI Whisper API.

## Features

- 🔊 High-quality speech recognition using Whisper model
- 🌐 OpenAI-compatible API endpoints
- 🚀 Hardware acceleration support (CUDA, MPS)
- ⚡ Flash Attention 2 for faster transcription on compatible GPUs
- 🎛️ Audio preprocessing for better transcription results
- 🔄 Multiple input formats (file upload, URL, base64, local files)
- 🚪 Easy deployment with Docker or conda environment

## Requirements

- Python 3.10+ (3.11 recommended)
- CUDA-compatible GPU (optional, for faster processing)
- FFmpeg and SoX for audio processing

## Installation

### Using conda (recommended)

1. Clone the repository:
```bash
git clone https://github.com/yourusername/whisper-api-service.git
cd whisper-api-service
```

2. Run the server script with the update flag to create and set up the conda environment:
```bash
chmod +x server.sh
./server.sh --update
```

This will:
- Create a conda environment named "transcribe" with Python 3.11
- Install all required dependencies
- Start the service

### Manual Installation

1. Create and activate a conda environment:
```bash
conda create -n transcribe python=3.11
conda activate transcribe
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Start the service:
```bash
python server.py
```

## Configuration

The service is configured through the `config.json` file:

```json
{
    "service_port": 5042,
    "model_path": "/mnt/cloud/llm/whisper/whisper-large-v3-russian",
    "language": "russian",
    "chunk_length_s": 30,
    "batch_size": 16,
    "max_new_tokens": 256,
    "return_timestamps": false,
    "norm_level": "-0.5",
    "compand_params": "0.3,1 -90,-90,-70,-70,-60,-20,0,0 -5 0 0.2"
}
```

### Configuration Parameters

| Parameter | Description |
|-----------|-------------|
| `service_port` | Port on which the service will run |
| `model_path` | Path to the Whisper model directory |
| `language` | Language for transcription (e.g., "russian", "english") |
| `chunk_length_s` | Length of audio chunks for processing (in seconds) |
| `batch_size` | Batch size for processing |
| `max_new_tokens` | Maximum new tokens for the model output |
| `return_timestamps` | Whether to return timestamps in the transcription |
| `audio_rate` | Audio sampling rate in Hz |
| `norm_level` | Normalization level for audio preprocessing |
| `compand_params` | Parameters for audio compression/expansion |

## API Usage

### Health Check

```bash
curl http://localhost:5042/health
```

### Get Configuration

```bash
curl http://localhost:5042/config
```

### Transcribe an Audio File (OpenAI-compatible)

```bash
curl -X POST http://localhost:5042/v1/audio/transcriptions \
  -F file=@audio.mp3
```

### Transcribe from URL

```bash
curl -X POST http://localhost:5042/v1/audio/transcriptions/url \
  -H "Content-Type: application/json" \
  -d '{"url":"https://example.com/audio.mp3"}'
```

### Transcribe from Base64

```bash
curl -X POST http://localhost:5042/v1/audio/transcriptions/base64 \
  -H "Content-Type: application/json" \
  -d '{"file":"base64_encoded_audio_data"}'
```

### Transcribe a Local File on the Server

```bash
curl -X POST http://localhost:5042/local/transcriptions \
  -H "Content-Type: application/json" \
  -d '{"file_path":"/path/to/audio.mp3"}'
```

## Project Structure

The project consists of the following components:

- `server.py`: Entry point that initializes and starts the service
- `server.sh`: Bash script for launching the server with optional conda environment update
- `config.json`: Service configuration file
- `requirements.txt`: Project dependencies for conda/pip
- `app/`: Main application module
  - `__init__.py`: Contains the `WhisperServiceAPI` class for service initialization
  - `logger.py`: Logging configuration
  - `transcriber.py`: Contains the `WhisperTranscriber` class for speech recognition
  - `audio_processor.py`: Contains the `AudioProcessor` class for audio preprocessing
  - `audio_sources.py`: Contains the `AudioSource` abstract class and implementations
  - `routes.py`: Contains the API route definitions

## Advanced Usage

### Using with Different Models

You can use any Whisper model by changing the `model_path` in the configuration:

1. Download a model from Hugging Face (e.g., `openai/whisper-large-v3`)
2. Update the `model_path` in `config.json`
3. Restart the service

#### Recommended Models

For Russian language transcription, we recommend using the [**whisper-large-v3-russian**](https://huggingface.co/antony66/whisper-large-v3-russian) model from Hugging Face. This model is fine-tuned specifically for Russian speech recognition and delivers high accuracy. For faster transcription with slightly lower accuracy, consider the [**whisper-large-v3-turbo-russian**](https://huggingface.co/dvislobokov/whisper-large-v3-turbo-russian) model, which is optimized for speed.

### Hardware Acceleration

The service automatically selects the best available compute device:
- CUDA GPU (index 1 if available, otherwise index 0)
- Apple Silicon MPS (for Mac with M1/M2/M3 chips)
- CPU (fallback)

For best performance on NVIDIA GPUs, Flash Attention 2 is used when available.

## Troubleshooting

### Audio Processing Issues

If you encounter audio processing errors:
- Ensure that FFmpeg and SoX are installed on your system
- Check that the audio file is not corrupted
- Try different audio preprocessing parameters in the configuration

### Performance Issues

For slow transcription:
- Use a GPU if available
- Adjust `chunk_length_s` and `batch_size` parameters
- Consider using a smaller Whisper model

## Acknowledgements

- OpenAI for the Whisper model
- Hugging Face for model distribution and transformers library
