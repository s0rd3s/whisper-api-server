# Whisper Speech-to-Text API Service

## Overview

This project is a lightweight, OpenAI-compatible API server for transcribing audio to text using the Whisper model. It's designed to run locally, making it easy to set up and use for speech-to-text tasks.

## Features

- **OpenAI API Compatibility**: Fully compatible with OpenAI's `/v1/audio/transcriptions` and `/v1/models` endpoints.
- **Local File Support**: Transcribe audio files stored locally on your machine.
- **Multiple Input Methods**: Supports:
  - Local file paths
  - Files accessible via URL
  - Base64-encoded audio
  - Multipart form uploads
- **Easy Setup**: Designed to run as a local service with minimal configuration.
- **Hardware Optimization**: Utilizes GPU (CUDA, MPS) or CPU for efficient processing.
- **Health Check**: Includes a `/health` endpoint for service monitoring.

## Recommended Model

For Russian language transcription, we recommend using the [**whisper-large-v3-russian**](https://huggingface.co/antony66/whisper-large-v3-russian) model from Hugging Face. This model is fine-tuned specifically for Russian speech recognition and delivers high accuracy. For faster transcription with slightly lower accuracy, consider the [**whisper-large-v3-turbo-russian**](https://huggingface.co/dvislobokov/whisper-large-v3-turbo-russian) model, which is optimized for speed.

Perfect for local development or offline use cases where OpenAI's API isn't accessible.

## Quick Start

1. **Edit the Configuration File (`config.json`)**:
   - Set the path to your Whisper model (`model_path`).
   - Configure other parameters like language (`language`), chunk size (`chunk_length_s`), batch size (`batch_size`), and audio normalization settings.
   ```json
   {
       "service_port": 5042,
       "model_path": "/path/to/your/whisper-model",
       "language": "english",
       "chunk_length_s": 30,
       "batch_size": 16,
       "max_new_tokens": 256,
       "return_timestamps": false,
       "norm_level": "-0.5",
       "compand_params": "0.3,1 -90,-90,-70,-70,-60,-20,0,0 -5 0 0.2"
   }
   ```

2. **Run the Server**:
   - Simply execute the `server.sh` script:
   ```bash
   ./server.sh
   ```
   - If you need to update the environment, use:
   ```bash
   ./server.sh --update
   ```

3. **Use the API**:
   - Once the server is running, you can send transcription requests.
   - Example request (curl):
   ```bash
   curl -X POST -F file=@audio.mp3 http://localhost:5042/v1/audio/transcriptions | jq -r '.text'
   ```

Enjoy seamless audio-to-text transcription with your local Whisper API server!