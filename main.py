from fer import FER
from pydub import AudioSegment
from datetime import timedelta
import cv2
import numpy as np
import os
import ffmpeg
from pathlib import Path
import subprocess
import matplotlib.pyplot as plt
import shutil
import datetime
from tqdm import tqdm
from deepface import DeepFace
import streamlit as st
import sys
import time

uploaded_file = st.file_uploader("Choose a file", type=['mp4'])
save_path=None
if uploaded_file is not None:
    save_folder = 'video'
    save_path = Path(save_folder, 'video.mp4')

    with open(save_path, mode='wb') as w:
            w.write(uploaded_file.getvalue())
            st.success(f'File {uploaded_file.name} is successfully saved!')    

import warnings
warnings.filterwarnings("ignore")

SAVING_FRAMES_PER_SECOND = 2

folder_dir_audio = "folder_dir_audio"

def format_timedelta(td):
    """Служебная функция для классного форматирования объектов timedelta (например, 00:00:20.05)
    исключая микросекунды и сохраняя миллисекунды"""
    result = str(td)
    try:
        result, ms = result.split(".")
    except ValueError:
        return "-" + result + ".00".replace(":", "-")
    ms = int(ms)
    ms = round(ms / 1e4)
    return f"-{result}.{ms:02}".replace(":", "-")


def get_saving_frames_durations(cap, saving_fps):
    """Функция, которая возвращает список длительностей сохраняемых кадров"""
    s = []
    # получаем длительность клипа, разделив количество кадров на количество кадров в секунду
    clip_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    # используем np.arange() для выполнения шагов с плавающей запятой
    for i in np.arange(0, clip_duration, 1 / saving_fps):
        s.append(i)
    return s

def main(video_file):
    #filename, _ = os.path.splitext(video_file)
    filename = "frames/"
    
    # создаем папку по названию видео файла
    if not os.path.isdir(filename):
        os.mkdir(filename)
    # читать видео файл    
    cap = cv2.VideoCapture(video_file)
    # получить FPS видео
    fps = cap.get(cv2.CAP_PROP_FPS)
    # если наше SAVING_FRAMES_PER_SECOND больше FPS видео, то установливаем минимальное
    saving_frames_per_second = min(fps, SAVING_FRAMES_PER_SECOND)
    # получить список длительностей кадров для сохранения
    saving_frames_durations = get_saving_frames_durations(cap, saving_frames_per_second)
    # начало цикла
    count = 0
    save_count = 0
    while True:
        is_read, frame = cap.read()
        if not is_read:
            # выйти из цикла, если нет фреймов для чтения
            break
        # получаем длительность, разделив текущее количество кадров на FPS
        frame_duration = count / fps
        try:
            # получить самую первоначальную длительность для сохранения
            closest_duration = saving_frames_durations[0]
        except IndexError:
            # список пуст, все кадры сохранены
            break
        if frame_duration >= closest_duration:
            # если ближайшая длительность меньше или равна длительности текущего кадра,
            # сохраняем фрейм
            frame_duration_formatted = format_timedelta(timedelta(seconds=frame_duration))
            saveframe_name = os.path.join(filename, f"frame{frame_duration_formatted}.jpg")
            cv2.imwrite(saveframe_name, frame)
            save_count += 1
            #print(f"{saveframe_name} сохранён")
            # удалить текущую длительность из списка, так как этот кадр уже сохранен
            try:
                saving_frames_durations.pop(0)
            except IndexError:
                pass
        # увеличить счечик кадров count
        count += 1
        
    print(f"Итого сохранено кадров {save_count}")

begtime = time.perf_counter()
main('video/video.mp4')
endtime = time.perf_counter()
print(f" {endtime - begtime} ")

folder_dir_video = 'frames'
emos=[]
images = Path(folder_dir_video).glob('*.jpg')
emo_detector = FER(mtcnn=True)
for image in images:
  test_image_one = plt.imread(image)
  captured_emotions = emo_detector.detect_emotions(test_image_one)
  dominant_emotion, emotion_score = emo_detector.top_emotion(test_image_one)
  if emotion_score == None:
    emotion_score = 0 
  if float(str(emotion_score)) > 0.5:
    emos.append(dominant_emotion)
  print(dominant_emotion, emotion_score)

def most_frequent(List):
    return max(set(List), key = List.count)


print(emos)
emos = [i for i in emos if i is not None]
emo=most_frequent(emos)
print(emo)

video = cv2.VideoCapture("video/video.mp4")

# Получение общего количества кадров в видео
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

dynamic_frame_count = 0
threshold = 10000

# Чтение первого кадра
ret, frame1 = video.read()
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

dynamic = 'static'
# Инициализация детектора движения
md = cv2.createBackgroundSubtractorMOG2()

with tqdm(total=total_frames) as pbar:
    while True:
        # Чтение следующего кадра
        ret, frame2 = video.read()
        if not ret:
            break

        # Преобразование кадра в серый цвет
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Вычисление разницы между двумя кадрами
        diff = cv2.absdiff(gray1, gray2)

        # Применение детектора движения
        mask = md.apply(diff)

        # Подсчет количества ненулевых пикселей в маске
        non_zero_count = cv2.countNonZero(mask)

        # Если количество ненулевых пикселей больше порога,
        # считать кадр динамическим
        if non_zero_count > threshold:
            dynamic_frame_count += 1

        pbar.update(1)
        # Обновление предыдущего кадра
        gray1 = gray2

if dynamic_frame_count/total_frames > 0.6:
    dynamic = 'dynamic'
    print("Видео динамическое")
else:
    print("Видео не динамическое")
print(dynamic)


chosen_audio=''
audios = Path(folder_dir_audio + '/' + dynamic).glob('*.mp3')
print(folder_dir_audio + '/' + dynamic)
for audio in audios:
  if Path(audio).stem == emo:
    chosen_audio=os.getcwd() +'/'+ folder_dir_audio + '/' + dynamic + '/' + emo + '.mp3'

print(os.getcwd())
print(chosen_audio)

def get_length(input_video):
    result = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', input_video], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return float(result.stdout)

with open(chosen_audio, mode='rb') as file:
            print(file)  
            song = AudioSegment.from_mp3(chosen_audio)
            print(song)
            fin=get_length(uploaded_file)*1000

            extract = song[0:fin]
            extract.export('extract.mp3', format="mp3")

            input_audio = ffmpeg.input('extract.mp3')

input_video = ffmpeg.input('video/video.mp4')


ffmpeg.concat(input_video, input_audio, v=1, a=1).output('finished_video.mp4').run()


with open('finished_video.mp4', 'rb') as f:
   st.download_button('Download', f, file_name='finished_video.mp4')