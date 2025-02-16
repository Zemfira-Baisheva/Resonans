import streamlit as st
import sounddevice as sd
import vosk
import queue
import json
import numpy as np
import time
from deep_translator import GoogleTranslator
from gtts import gTTS
from playsound import playsound
import tempfile
import threading

# === Настройки ===
MODEL_PATH = "vosk-model-small-ru-0.22/vosk-model-small-ru-0.22"
SAMPLE_RATE = 16000  # Частота дискретизации
BATCH_TIME = 4  # Время (в сек.), через которое текст отправляется на обработку
SPEECH_QUEUE = queue.Queue()  # Очередь для озвучки

col1, col2, col3 = st.columns([1, 6, 1])  # Пропорции колонок

with col2:  # Центрируем изображение и текст во второй колонке
    # Меньшее изображение
    st.image("resize_photo_2025-02-16_12-40-54.jpg", width=400)  # Указан размер ширины изображения


# === Очередь для аудио ===
q = queue.Queue()

# === Функция для обработки аудиопотока ===
def audio_callback(indata, frames, time, status):
    """Обрабатывает аудио и добавляет его в очередь"""
    if status:
        st.error(f"Ошибка аудио: {status}")
    try:
        if np.all(indata == 0):
            return  # Пропускаем пустой или тихий фрейм
        q.put(bytes(indata))
    except Exception as e:
        st.error(f"Ошибка при добавлении аудио в очередь: {e}")

# === Загрузка модели Vosk ===
model = vosk.Model(MODEL_PATH)
recognizer = vosk.KaldiRecognizer(model, SAMPLE_RATE)

# === Интерфейс Streamlit ===
st.title("Resonance")
st.text("Нажмите кнопку ниже, чтобы начать запись с микрофона.")

# === Выбор языка для перевода ===
language_codes = {
    "Английский": "en",
    "Немецкий": "de",
    "Французский": "fr",
    "Испанский": "es",
    "Китайский": "zh-TW",
    "Японский": "ja",
    "Португальский": "pt",
    "Хинди": "hi",
    "Арабский": "ar"
}
target_language = st.selectbox("Выберите язык для перевода:", list(language_codes.keys()))

# === Включение/выключение озвучки ===
enable_speech = st.checkbox("🔊 Включить озвучку перевода", value=True)

# === Поля для отображения текста ===
subtitle_text = st.empty()  # Текущий распознаваемый текст
translated_text = st.empty()  # Переведенный текст
history_text = st.empty()   # История всех распознанных фраз

# Хранение истории
if "history" not in st.session_state:
    st.session_state.history = ""
if "translated_history" not in st.session_state:
    st.session_state.translated_history = ""

# === Функция перевода текста ===
def translate_text(text, target_lang):
    try:
        translation = GoogleTranslator(source="auto", target=language_codes[target_lang]).translate(text)
        return translation
    except Exception as e:
        st.error(f"Ошибка перевода: {e}")
        return ""

# === Функция синтеза речи (озвучка только перевода) ===
def speak_text():
    """Озвучивает текст из очереди по порядку"""
    while True:
        text, lang = SPEECH_QUEUE.get()  # Берем следующий фрагмент из очереди
        if text.strip():
            try:
                with tempfile.NamedTemporaryFile(delete=True, suffix=".mp3") as temp_audio:
                    tts = gTTS(text=text, lang=lang)
                    tts.save(temp_audio.name)
                    playsound(temp_audio.name)  # Воспроизводим звук
            except Exception as e:
                st.error(f"Ошибка синтеза речи: {e}")
        SPEECH_QUEUE.task_done()  # Помечаем задачу как выполненную

# Запускаем поток для озвучки
speech_thread = threading.Thread(target=speak_text, daemon=True)
speech_thread.start()

# === Функция для обработки аудио ===
def process_audio():
    with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=8000, dtype="int16",
                           channels=1, callback=audio_callback):
        st.write("🎤 Говорите...")
        start_time = time.time()
        
        while True:
            data = q.get()
            if recognizer.AcceptWaveform(data) or (time.time() - start_time > BATCH_TIME):
                result = json.loads(recognizer.Result())
                start_time = time.time()  # Сбрасываем таймер
                
                if result["text"].strip():
                    time.sleep(0.5)  # Задержка перед выводом финального текста
                    recognized_text = result["text"]
                    subtitle_text.text(f"📝 Текущие слова: {recognized_text}")

                    # Перевод
                    translated = translate_text(recognized_text, target_language)
                    translated_text.text(f"🌍 Перевод: {translated}")

                    # Добавляем в историю
                    st.session_state.history += recognized_text + " "
                    st.session_state.translated_history += translated + " "

                    history_text.text(f"📜 Полный текст:\n{st.session_state.history}\n\n"
                                      f"🌍 Полный перевод:\n{st.session_state.translated_history}")

                    # Добавляем текст в очередь на озвучку
                    if enable_speech:
                        SPEECH_QUEUE.put((translated, language_codes[target_language]))

            else:
                partial_result = recognizer.PartialResult()
                partial_data = json.loads(partial_result)
                if partial_data.get("partial", "").strip():
                    subtitle_text.text(f"📝 Текущие слова: {partial_data['partial']}")

# === Кнопки управления ===
col1, col2 = st.columns(2)
with col1:
    if st.button("🎙️ Начать запись"):
        process_audio()
with col2:
    if st.button("🗑️ Очистить историю"):
        st.session_state.history = ""
        st.session_state.translated_history = ""
        history_text.text("📜 История очищена.")