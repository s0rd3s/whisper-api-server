import argparse
from app import WhisperServiceAPI

def main():
    """
    Локальный, OpenAI совместимый API сервис распознавания речи с использованием модели Whisper.
    """

    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description="Сервис распознавания речи с использованием модели Whisper")
    parser.add_argument("--config", help="Путь к файлу конфигурации", default="config.json")
    
    args = parser.parse_args()
    
    # Запуск сервиса
    service = WhisperServiceAPI(args.config)
    service.run()


if __name__ == "__main__":
    main()
