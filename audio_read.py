# Колаб по какой-то причине в один момент отказался выполнять строки 16-18, поэтому код запускался в анаконде
# Подключаем библиотеки

import librosa
import pandas as pd
import pickle
import numpy as np

#Читаем данные

link = "D:/Mein/Учеба/Университет ИТМО/Анализ акустических событий/"

data_librosa = []
data_set = pd.read_csv(link + 'train.csv')
  
for fname in data_set['fname']': 
  data, _ = librosa.load(link +'audio_train/' + fname, sr=8000)
  data_librosa.append(data)
 
#Сохраняем данные в .pickle
  
with open(link + 'data.pickle', 'wb') as f_d:
    pickle.dump(data_librosa, f_d)
with open(link + 'label.pickle', 'wb') as f_l:
    pickle.dump(data_set['label'], f_l)