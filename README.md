# Whisper Speech-to-Text API Service

## Overview

This project is a local API server compatible with OpenAI's API for transcribing audio to text using the Whisper model. It's designed to run as a system service, loading the Whisper model into memory at startup and handling transcription requests via REST API.

## Features

- **Audio Transcription**: Supports various input methods:
  - Local server files
  - Files accessible via URL
  - Base64-encoded files
  - Multipart form data
- **OpenAI API Compatibility**: Works with `/v1/audio/transcriptions` and `/v1/models` endpoints.
- **Audio Preprocessing**: Converts audio to WAV, normalizes, and adds silence.
- **Hardware Support**: Utilizes GPU (CUDA, MPS) or CPU.
- **Logging**: Tracks all operations.
- **Health Check**: Includes a health check endpoint.

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
   curl -X POST -F file=@audio.wav http://localhost:5042/v1/audio/transcriptions
   ```

Enjoy seamless audio-to-text transcription with your local Whisper API server!