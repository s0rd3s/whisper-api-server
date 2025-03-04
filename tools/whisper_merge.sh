#!/bin/bash

# Имя окружения conda
CONDA_ENV_NAME="whisper-merge"

# Директории для моделей
MODEL_A="/mnt/cloud/llm/whisper/whisper-large-v3-russian"
MODEL_B="/mnt/cloud/llm/whisper/whisper-large-v3-ru-podlodka"
OUTPUT_DIR="/mnt/cloud/llm/whisper/whisper-large-v3-russian+podlodka"

# Флаг обновления (по умолчанию false)
UPDATE_ENV=false

# Метод слияния (по умолчанию all)
MERGE_METHOD="all"

# Дополнительные параметры (по умолчанию)
SLERP_T=0.5
TIES_DENSITY=0.8
ENCODER_WEIGHTS="0.6,0.4"
DECODER_WEIGHTS="0.4,0.6"
DIRECT_ALPHA=0.5

# Проверка аргументов
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --update) UPDATE_ENV=true ;;
    --method) MERGE_METHOD="$2"; shift ;;
    --slerp-t) SLERP_T="$2"; shift ;;
    --ties-density) TIES_DENSITY="$2"; shift ;;
    --encoder-weights) ENCODER_WEIGHTS="$2"; shift ;;
    --decoder-weights) DECODER_WEIGHTS="$2"; shift ;;
    --direct-alpha) DIRECT_ALPHA="$2"; shift ;;
    *) echo "⚠️ Неизвестный параметр: $1"; exit 1 ;;
  esac
  shift
done

# Проверка наличия conda
if ! command -v conda &> /dev/null; then
    echo "❌ Conda не установлен. Пожалуйста, установите conda и попробуйте снова."
    exit 1
fi

# Создание окружения conda, если оно не существует
if ! conda env list | grep -q "$CONDA_ENV_NAME"; then
    echo "🔹 Создание окружения conda: $CONDA_ENV_NAME"
    conda create -n "$CONDA_ENV_NAME" python=3.12 -y
else
    echo "✅ Окружение conda '$CONDA_ENV_NAME' уже существует."
fi

# Получение пути к conda
CONDA_PATH=$(which conda)

# Проверка, что путь к conda найден
if [ -z "$CONDA_PATH" ]; then
    echo "❌ Не удалось найти путь к conda."
    exit 1
fi

# Активация окружения conda
echo "🔹 Активация окружения conda: $CONDA_ENV_NAME"
source $(dirname "$CONDA_PATH")/../etc/profile.d/conda.sh
conda activate "$CONDA_ENV_NAME"

# Если флаг --update установлен, обновляем зависимости
if [[ "$UPDATE_ENV" == true ]]; then
    echo "🔹 Установка и обновление зависимостей"
    pip install --upgrade pip wheel
    pip install torch torchaudio
    pip install transformers datasets evaluate accelerate
    pip install soundfile librosa tqdm
fi

# Проверка установки torch и transformers
if ! python -c "import torch, transformers" &> /dev/null; then
    echo "❌ PyTorch или Transformers не установлены. Запустите с опцией --update."
    exit 1
fi

# Создание временного Python-скрипта для слияния
cat > merge_models.py << EOL
#!/usr/bin/env python3

import argparse
import os
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from tqdm import tqdm

def direct_merge(model_a_path, model_b_path, output_path, alpha=0.5):
    """Простое прямое слияние весов моделей с коэффициентом alpha"""
    print(f"🔹 Запуск прямого слияния с alpha={alpha}")
    
    print(f"📂 Загрузка модели A из {model_a_path}")
    model_a = WhisperForConditionalGeneration.from_pretrained(model_a_path)
    
    print(f"📂 Загрузка модели B из {model_b_path}")
    model_b = WhisperForConditionalGeneration.from_pretrained(model_b_path)
    
    # Создаем новую модель для хранения объединенных весов
    print("🔄 Объединение весов моделей...")
    merged_model = WhisperForConditionalGeneration.from_pretrained(model_a_path)
    
    # Подсчет общего количества параметров
    param_count = len(list(merged_model.named_parameters()))
    
    # Объединение весов
    with torch.no_grad():
        for i, (param_name, param) in enumerate(tqdm(merged_model.named_parameters(), total=param_count, desc="Слияние параметров")):
            if param_name in model_a.state_dict() and param_name in model_b.state_dict():
                # Объединяем веса с учетом коэффициента альфа
                merged_param = alpha * model_a.state_dict()[param_name] + (1 - alpha) * model_b.state_dict()[param_name]
                param.copy_(merged_param)
    
    # Создаем директорию для выходных данных, если она не существует
    os.makedirs(output_path, exist_ok=True)
    
    # Сохраняем объединенную модель
    print(f"💾 Сохранение объединенной модели в {output_path}")
    merged_model.save_pretrained(output_path)
    
    # Сохраняем процессор из модели A
    print("💾 Сохранение процессора")
    processor = WhisperProcessor.from_pretrained(model_a_path)
    processor.save_pretrained(output_path)
    
    print("✅ Прямое слияние успешно завершено!")
    return merged_model

def slerp_weights(model_a, model_b, t):
    """Выполняет SLERP (Spherical Linear Interpolation) между весами моделей"""
    # Получаем словари состояний моделей
    state_dict_a = model_a.state_dict()
    state_dict_b = model_b.state_dict()
    
    # Создаем новый словарь для смешанных весов
    mixed_weights = {}
    
    for key in tqdm(state_dict_a.keys(), desc="SLERP слияние"):
        if key in state_dict_b:
            # Получаем веса для текущего параметра
            weight_a = state_dict_a[key].float()
            weight_b = state_dict_b[key].float()
            
            # Выполняем SLERP
            # Нормализуем веса
            norm_a = torch.norm(weight_a)
            norm_b = torch.norm(weight_b)
            
            # Если норма близка к нулю, используем линейную интерполяцию
            if norm_a < 1e-6 or norm_b < 1e-6:
                mixed_weights[key] = (1 - t) * weight_a + t * weight_b
                continue
            
            weight_a_normalized = weight_a / norm_a
            weight_b_normalized = weight_b / norm_b
            
            # Вычисляем косинус угла между векторами
            cos_theta = torch.sum(weight_a_normalized * weight_b_normalized)
            cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
            
            # Вычисляем угол
            theta = torch.acos(cos_theta)
            
            # Если угол близок к 0 или π, используем линейную интерполяцию
            if abs(theta) < 1e-6 or abs(theta - torch.pi) < 1e-6:
                mixed_weights[key] = (1 - t) * weight_a + t * weight_b
            else:
                # SLERP формула: (sin((1-t)*theta) / sin(theta)) * weight_a + (sin(t*theta) / sin(theta)) * weight_b
                sin_theta = torch.sin(theta)
                mixed_weights[key] = (torch.sin((1 - t) * theta) / sin_theta) * weight_a + (torch.sin(t * theta) / sin_theta) * weight_b
            
            # Проверка на NaN
            if torch.isnan(mixed_weights[key]).any():
                # Если появились NaN, используем линейную интерполяцию
                mixed_weights[key] = (1 - t) * weight_a + t * weight_b
    
    return mixed_weights

def slerp_merge(model_a_path, model_b_path, output_path, t=0.5):
    """Слияние методом SLERP, реализованное напрямую без использования mergekit"""
    print(f"🔹 Запуск слияния методом SLERP с t={t}")
    
    print(f"📂 Загрузка модели A из {model_a_path}")
    model_a = WhisperForConditionalGeneration.from_pretrained(model_a_path)
    
    print(f"📂 Загрузка модели B из {model_b_path}")
    model_b = WhisperForConditionalGeneration.from_pretrained(model_b_path)
    
    # Создаем новую модель для хранения смешанных весов
    print("🔄 Применение метода SLERP...")
    merged_model = WhisperForConditionalGeneration.from_pretrained(model_a_path)
    
    # Получаем смешанные веса
    mixed_weights = slerp_weights(model_a, model_b, t)
    
    # Загружаем смешанные веса в модель
    merged_model.load_state_dict(mixed_weights)
    
    # Создаем директорию для выходных данных
    os.makedirs(output_path, exist_ok=True)
    
    # Сохраняем объединенную модель
    print(f"💾 Сохранение объединенной модели в {output_path}")
    merged_model.save_pretrained(output_path)
    
    # Сохраняем процессор из модели A
    processor = WhisperProcessor.from_pretrained(model_a_path)
    processor.save_pretrained(output_path)
    
    print("✅ Слияние методом SLERP успешно завершено!")
    return merged_model

def ties_merge(model_a_path, model_b_path, output_path, density=0.8, encoder_weights=(0.6, 0.4), decoder_weights=(0.4, 0.6)):
    """Слияние, реализованное по принципу TIES без использования mergekit"""
    print(f"🔹 Запуск слияния по принципу TIES с плотностью {density}")
    print(f"   Веса кодировщика: {encoder_weights}, веса декодера: {decoder_weights}")
    
    print(f"📂 Загрузка модели A из {model_a_path}")
    model_a = WhisperForConditionalGeneration.from_pretrained(model_a_path)
    
    print(f"📂 Загрузка модели B из {model_b_path}")
    model_b = WhisperForConditionalGeneration.from_pretrained(model_b_path)
    
    # Создаем новую модель для хранения смешанных весов
    print("🔄 Применение метода TIES...")
    merged_model = WhisperForConditionalGeneration.from_pretrained(model_a_path)
    
    # Получаем словари состояний моделей
    state_dict_a = model_a.state_dict()
    state_dict_b = model_b.state_dict()
    
    # Создаем словарь для смешанных весов
    mixed_weights = {}
    
    # Для каждого параметра
    for key in tqdm(state_dict_a.keys(), desc="TIES слияние"):
        if key in state_dict_b:
            # Определение типа слоя (кодировщик или декодер)
            if "encoder" in key:
                w_a, w_b = encoder_weights
            elif "decoder" in key:
                w_a, w_b = decoder_weights
            else:
                # Для других параметров используем среднее
                w_a = (encoder_weights[0] + decoder_weights[0]) / 2
                w_b = (encoder_weights[1] + decoder_weights[1]) / 2
            
            # Получаем веса для текущего параметра
            weight_a = state_dict_a[key]
            weight_b = state_dict_b[key]
            
            # Вычисляем абсолютную разницу между весами
            abs_diff = torch.abs(weight_a - weight_b)
            
            # Создаем маску для топ-k% элементов с наибольшей разницей
            # если density=0.8, то мы сохраняем 80% элементов с наибольшей разницей
            k = int((1 - density) * abs_diff.numel())
            if k > 0:
                threshold = torch.kthvalue(abs_diff.flatten(), k).values
                mask = abs_diff >= threshold
            else:
                mask = torch.ones_like(abs_diff, dtype=torch.bool)
            
            # Создаем смешанные веса:
            # Для элементов с большой разницей используем веса w_a и w_b
            # Для остальных используем среднее арифметическое
            mixed = torch.zeros_like(weight_a)
            mixed[mask] = w_a * weight_a[mask] + w_b * weight_b[mask]
            mixed[~mask] = (weight_a[~mask] + weight_b[~mask]) / 2
            
            mixed_weights[key] = mixed
    
    # Загружаем смешанные веса в модель
    merged_model.load_state_dict(mixed_weights)
    
    # Создаем директорию для выходных данных
    os.makedirs(output_path, exist_ok=True)
    
    # Сохраняем объединенную модель
    print(f"💾 Сохранение объединенной модели в {output_path}")
    merged_model.save_pretrained(output_path)
    
    # Сохраняем процессор из модели A
    processor = WhisperProcessor.from_pretrained(model_a_path)
    processor.save_pretrained(output_path)
    
    print("✅ Слияние по принципу TIES успешно завершено!")
    return merged_model

def main():
    parser = argparse.ArgumentParser(description="Объединение моделей Whisper методами SLERP и TIES")
    parser.add_argument("--model-a", default="$MODEL_A", 
                        help="Путь к первой модели")
    parser.add_argument("--model-b", default="$MODEL_B", 
                        help="Путь к второй модели")
    parser.add_argument("--output-dir", default="$OUTPUT_DIR", 
                        help="Директория для сохранения результатов")
    parser.add_argument("--method", choices=["slerp", "ties", "direct", "all"], default="$MERGE_METHOD", 
                        help="Метод слияния: slerp, ties, direct или all")
    
    # Параметры для SLERP
    parser.add_argument("--slerp-t", type=float, default=$SLERP_T, 
                        help="Коэффициент интерполяции для SLERP (0-1)")
    
    # Параметры для TIES
    parser.add_argument("--ties-density", type=float, default=$TIES_DENSITY, 
                        help="Плотность смешивания для TIES (0-1)")
    parser.add_argument("--encoder-weights", type=str, default="$ENCODER_WEIGHTS", 
                        help="Веса для кодировщика в формате '0.6,0.4'")
    parser.add_argument("--decoder-weights", type=str, default="$DECODER_WEIGHTS", 
                        help="Веса для декодера в формате '0.4,0.6'")
    
    # Параметры для прямого слияния
    parser.add_argument("--direct-alpha", type=float, default=$DIRECT_ALPHA, 
                        help="Коэффициент альфа для прямого слияния (0-1)")
    
    args = parser.parse_args()
    
    # Разбор весов для TIES
    encoder_weights = tuple(map(float, args.encoder_weights.split(',')))
    decoder_weights = tuple(map(float, args.decoder_weights.split(',')))
    
    # Создание выходной директории
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Выполнение слияния выбранным методом
    if args.method == "direct" or args.method == "all":
        direct_output = os.path.join(args.output_dir, "direct")
        direct_merge(args.model_a, args.model_b, direct_output, args.direct_alpha)
    
    if args.method == "slerp" or args.method == "all":
        slerp_output = os.path.join(args.output_dir, "slerp")
        try:
            slerp_merge(args.model_a, args.model_b, slerp_output, args.slerp_t)
        except Exception as e:
            print(f"❌ Ошибка при слиянии методом SLERP: {e}")
            print("⚠️ Используется прямое слияние вместо SLERP.")
            direct_merge(args.model_a, args.model_b, slerp_output, args.slerp_t)
    
    if args.method == "ties" or args.method == "all":
        ties_output = os.path.join(args.output_dir, "ties")
        try:
            ties_merge(args.model_a, args.model_b, ties_output, args.ties_density, encoder_weights, decoder_weights)
        except Exception as e:
            print(f"❌ Ошибка при слиянии методом TIES: {e}")
            print("⚠️ Используется прямое слияние вместо TIES.")
            alpha = (encoder_weights[0] + decoder_weights[0]) / 2
            direct_merge(args.model_a, args.model_b, ties_output, alpha)
    
    print("\n✅ Все операции слияния успешно завершены!")
    print(f"📂 Результаты сохранены в {args.output_dir}")

if __name__ == "__main__":
    main()
EOL

# Запуск Python-скрипта
python merge_models.py

# Удаление временного скрипта
rm merge_models.py

echo ""
echo "✅ Процесс слияния завершен. Результаты сохранены в $OUTPUT_DIR"
