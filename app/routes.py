import os
import uuid
import tempfile
import base64
import requests
import time
from flask import request, jsonify
from typing import Dict

from .logger import logger

class FakeFile:
    """Имитирует файловый объект для унификации обработки из разных источников.
    
    Позволяет обрабатывать файлы из различных источников (локальный путь, URL, base64)
    как стандартные файловые объекты Flask, обеспечивая совместимость с существующей 
    логикой обработки файлов.

    Атрибуты:
    - file: Исходный файловый объект или поток
    - filename: Имя файла для метаданных

    Методы эмулируют поведение стандартного файлового объекта:
    - read(): Чтение содержимого файла
    - seek(): Перемещение позиции чтения
    - tell(): Текущая позиция чтения
    - name (property): Возвращает имя файла

    Пример использования:
    >>> with open('audio.wav', 'rb') as f:
    >>>     fake = FakeFile(f, 'audio.wav')
    >>>     fake.save('/tmp/copy.wav')  # Новый метод сохранения
    >>>     processor.handle_file(fake)
    """
    def __init__(self, file, filename):
        self.file = file
        self.filename = filename

    def read(self):
        return self.file.read()

    def seek(self, offset, whence=0):
        self.file.seek(offset, whence)

    def tell(self):
        return self.file.tell()

    def save(self, destination):
        """Сохраняет содержимое файла в указанное место назначения.
        
        Args:
            destination (str): Путь для сохранения файла
            
        Реализует совместимость с Flask FileStorage API. После записи
        сбрасывает позицию чтения в начало файла для последующих операций.
        """
        with open(destination, 'wb') as f:
            content = self.file.read()
            f.write(content)
            self.file.seek(0)  # Reset pointer after reading

    @property
    def name(self):
        return self.filename

class Routes:
    """Класс для регистрации всех эндпоинтов API.

    Этот класс содержит все маршруты (endpoints) для взаимодействия с сервером транскрибации.
    Он предоставляет функциональность для получения списка доступных моделей, информации о конкретной модели,
    а также для транскрибации аудиофайлов, загруженных различными способами.

    Эндпоинты:
    - GET /v1/models:
        Возвращает JSON-список доступных моделей для транскрибации. Каждая модель содержит информацию об ID,
        типе объекта, владельце и разрешениях.

    - GET /v1/models/<model_id>:
        Возвращает JSON-объект с информацией о конкретной модели, идентифицированной по <model_id>.
        Если модель не найдена, возвращает ошибку 404.

    - POST /v1/audio/transcriptions:
        Транскрибирует аудиофайл, загруженный через форму. Ожидает, что файл будет передан в поле 'file'
        multipart формы. Возвращает JSON с транскрибированным текстом и временем обработки.
        Поддерживает параметры language, temperature и prompt, передаваемые также через форму.

    - POST /v1/audio/transcriptions/url:
        Транскрибирует аудиофайл, доступный по указанному URL. Ожидает JSON-запрос с полем 'url',
        содержащим URL аудиофайла. Возвращает JSON с транскрибированным текстом и временем обработки.

    - POST /v1/audio/transcriptions/base64:
        Транскрибирует аудиофайл, закодированный в base64. Ожидает JSON-запрос с полем 'file',
        содержащим base64-encoded представление аудиофайла. Возвращает JSON с транскрибированным текстом
        и временем обработки.

    - POST /v1/audio/transcriptions/multipart:
        Аналогичен /v1/audio/transcriptions, но явно указывает на то, что файл ожидается в multipart форме.
        Используется для транскрибации аудиофайла, загруженного через multipart-форму.
        Возвращает JSON с транскрибированным текстом и временем обработки.
        Поддерживает параметры language, temperature и prompt, передаваемые также через форму.
    """

    def __init__(self, app, transcriber, config: Dict):
        """
        Инициализация маршрутов.

        Args:
            app: Flask-приложение.
            transcriber: Экземпляр транскрайбера.
            config: Словарь с конфигурацией.
        """
        self.app = app
        self.transcriber = transcriber
        self.config = config
        self.max_file_size_mb = self.config.get("max_file_size", 100)  # Default 100MB

        # Регистрация маршрутов
        self._register_routes()

    def _process_audio_file(self, file, request_form=None):
        """
        Общая функция для обработки аудиофайла.

        Args:
            file: Объект файла, полученный из запроса.
            request_form: Объект request.form, если есть параметры из формы.

        Returns:
            jsonify: JSON-ответ с результатом транскрибации.
        """
        if not file:
            return jsonify({"error": "No file part"}), 400

        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        # Проверка размера файла
        file.seek(0, os.SEEK_END)
        file_length = file.tell()
        file.seek(0)  # Reset file pointer after reading for size check

        if file_length > self.max_file_size_mb * 1024 * 1024:
            return jsonify({"error": f"File exceeds maximum size of {self.max_file_size_mb}MB"}), 413

        # Извлекаем параметры из запроса, если они есть
        language = request_form.get('language', self.config.get('language', 'en')) if request_form else self.config.get('language', 'en')  # Default language
        temperature = float(request_form.get('temperature', 0.0)) if request_form else 0.0  # Default temperature
        prompt = request_form.get('prompt', '') if request_form else ''  # Default prompt

        # Сохраняем файл во временный файл
        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, str(uuid.uuid4()) + "_" + os.path.basename(file.filename))
        file.save(temp_file_path)

        try:
            start_time = time.time()
            text = self.transcriber.process_file(temp_file_path)
            processing_time = time.time() - start_time

            # Форматируем ответ в стиле OpenAI
            return jsonify({
                "text": text,
                "processing_time": processing_time,
                "response_size_bytes": len(text.encode('utf-8'))
            }), 200

        except Exception as e:
            logger.error(f"Ошибка при транскрибации: {e}")
            return jsonify({"error": str(e)}), 500

        finally:
            # Очистка временных файлов
            os.remove(temp_file_path)
            os.rmdir(temp_dir)

    def _register_routes(self):
        """Регистрация всех эндпоинтов."""

        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Эндпоинт для проверки статуса сервиса и получения конфигурации."""
            return jsonify({
                "status": "ok",
                "config": self.config
            }), 200

        @self.app.route('/local/transcriptions', methods=['POST'])
        def local_transcribe():
            """Эндпоинт для локальной транскрибации файла по пути на сервере."""
            data = request.json
            
            if not data or "file_path" not in data:
                return jsonify({"error": "No file_path provided"}), 400

            file_path = data["file_path"]
            
            if not os.path.exists(file_path):
                return jsonify({"error": "File not found"}), 400

            try:
                with open(file_path, 'rb') as f:
                    # Создаем объект файла, совместимый с обработчиком
                    fake_file = FakeFile(f, os.path.basename(file_path))
                    return self._process_audio_file(fake_file)

            except Exception as e:
                logger.error(f"Ошибка локальной транскрибации: {e}")
                return jsonify({
                    "error": "Processing error",
                    "details": str(e)
                }), 500

        @self.app.route('/v1/models', methods=['GET'])
        def list_models():
            """Эндпоинт для получения списка доступных моделей."""
            return jsonify({
                "data": [
                    {
                        "id": os.path.basename(self.config["model_path"]),  # Имя модели из конфига
                        "object": "model",
                        "owned_by": "openai",
                        "permissions": []
                    }
                ],
                "object": "list"
            }), 200

        @self.app.route('/v1/models/<model_id>', methods=['GET'])
        def retrieve_model(model_id):
            """Эндпоинт для получения информации о конкретной модели."""
            if model_id == os.path.basename(self.config["model_path"]):
                return jsonify({
                    "id": model_id,
                    "object": "model",
                    "owned_by": "openai",
                    "permissions": []
                }), 200
            else:
                return jsonify({
                    "error": "Model not found",
                    "details": f"Model '{model_id}' does not exist"
                }), 404

        @self.app.route('/v1/audio/transcriptions', methods=['POST'])
        def openai_transcribe_endpoint():
            """Эндпоинт для транскрибации аудиофайла."""
            if 'file' not in request.files:
                return jsonify({"error": "No file part"}), 400

            file = request.files['file']
            return self._process_audio_file(file, request.form)

        @self.app.route('/v1/audio/transcriptions/url', methods=['POST'])
        def transcribe_from_url():
            """Эндпоинт для транскрибации аудиофайла по URL."""
            data = request.json

            if not data or "url" not in data:
                return jsonify({
                    "error": "No URL provided",
                    "details": "Please provide 'url' in the JSON request"
                }), 400

            url = data["url"]

            try:
                # Скачиваем файл по URL
                response = requests.get(url, stream=True)
                response.raise_for_status()

                # Сохраняем файл во временный файл
                temp_dir = tempfile.mkdtemp()
                temp_file_path = os.path.join(temp_dir, str(uuid.uuid4()) + ".wav")
                with open(temp_file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                # Открываем файл для обработки
                with open(temp_file_path, 'rb') as file:
                    # Создаем объект файла, как будто он пришел из request.files
                    fake_file = FakeFile(file, os.path.basename(temp_file_path))
                    result = self._process_audio_file(fake_file)

                return result

            except Exception as e:
                logger.error(f"Ошибка при транскрибации файла по URL {url}: {e}")
                return jsonify({
                    "error": "Transcription error",
                    "details": str(e)
                }), 500

            finally:
                # Очистка временных файлов
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                if os.path.exists(temp_dir):
                    os.rmdir(temp_dir)

        @self.app.route('/v1/audio/transcriptions/base64', methods=['POST'])
        def transcribe_from_base64():
            """Эндпоинт для транскрибации аудио, закодированного в base64."""
            data = request.json

            if not data or "file" not in data:
                return jsonify({
                    "error": "No base64 file provided",
                    "details": "Please provide 'file' in the JSON request"
                }), 400

            base64_data = data["file"]

            try:
                # Декодируем base64
                audio_data = base64.b64decode(base64_data)

                # Сохраняем файл во временный файл
                temp_dir = tempfile.mkdtemp()
                temp_file_path = os.path.join(temp_dir, str(uuid.uuid4()) + ".wav")
                with open(temp_file_path, 'wb') as f:
                    f.write(audio_data)

                # Открываем файл для обработки
                with open(temp_file_path, 'rb') as file:
                    # Создаем объект файла, как будто он пришел из request.files
                    fake_file = FakeFile(file, os.path.basename(temp_file_path))
                    result = self._process_audio_file(fake_file)

                return result

            except Exception as e:
                logger.error(f"Ошибка при транскрибации файла из base64: {e}")
                return jsonify({
                    "error": "Transcription error",
                    "details": str(e)
                }), 500

            finally:
                # Очистка временных файлов
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                if os.path.exists(temp_dir):
                    os.rmdir(temp_dir)

        @self.app.route('/v1/audio/transcriptions/multipart', methods=['POST'])
        def transcribe_multipart():
            """Эндпоинт для транскрибации аудиофайла, загруженного через форму."""
            if 'file' not in request.files:
                return jsonify({"error": "No file part"}), 400

            file = request.files['file']
            return self._process_audio_file(file, request.form)