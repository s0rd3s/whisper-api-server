#!/usr/bin/env python3
"""
Клиент для взаимодействия с сервисом распознавания речи через REST API.
"""

import argparse
import json
import os
import requests
import sys


class TranscribeClient:
    """
    Класс для взаимодействия с сервисом распознавания речи через REST API.
    """
    
    def __init__(self, server_url="http://localhost:5000"):
        """
        Инициализация клиента.
        
        Args:
            server_url: URL сервера распознавания.
        """
        self.server_url = server_url
    
    def health(self):
        """
        Проверка статуса сервиса.
        
        Returns:
            Словарь с информацией о статусе сервиса или None при ошибке.
        """
        endpoint = f"{self.server_url}/health"
        try:
            response = requests.get(endpoint, timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Ошибка health check: {response.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Ошибка при выполнении health check: {e}")
            return None
    
    def transcribe(self, file_path):
        """
        Отправка файла на транскрибацию.
        
        Args:
            file_path: Путь к аудиофайлу.
        
        Returns:
            Словарь с результатом транскрибации или None при ошибке.
        """
        # Проверка существования файла
        if not os.path.exists(file_path):
            print(f"Ошибка: Файл '{file_path}' не найден")
            return None
        
        # Формирование абсолютного пути к файлу
        absolute_path = os.path.abspath(file_path)
        
        # Подготовка данных для запроса
        payload = {"file_path": absolute_path}
        
        # Запрос на транскрибацию
        endpoint = f"{self.server_url}/local/transcriptions"
        
        try:
            # Отправка запроса
            response = requests.post(endpoint, json=payload, timeout=600)  # Таймаут 10 минут
            
            # Обработка ответа
            if response.status_code == 200:
                result = response.json()
                return result
            else:
                print(f"Ошибка запроса: {response.status_code}")
                if response.headers.get('content-type') == 'application/json':
                    print(json.dumps(response.json(), indent=2, ensure_ascii=False))
                else:
                    print(response.text)
                return None
        
        except requests.exceptions.RequestException as e:
            print(f"Ошибка при отправке запроса: {e}")
            return None


def main():
    """Основная функция клиента."""
    parser = argparse.ArgumentParser(description="Клиент для сервиса распознавания речи")
    parser.add_argument("file_path", help="Путь к аудиофайлу для транскрибации")
    parser.add_argument("--server", help="URL сервера распознавания", default="http://localhost:5042")
    
    args = parser.parse_args()
    
    # Создание экземпляра клиента
    client = TranscribeClient(args.server)
    
    # Запуск транскрибации
    result = client.transcribe(args.file_path)
    
    if result:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return result
    
    else:
        print("Что-то пошло не так 🤔")
        return None


if __name__ == "__main__":
    main()
