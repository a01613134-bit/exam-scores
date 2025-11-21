import numpy as np
import streamlit as st
import pandas as pd
st.write(''' Vamos a predecir la calificación   ''')
st.image("calificaciones.jpg", caption="Predicción de calificaciones.")
st.header('Datos estudiante')
df=pd.read_csv("student_exam_scores.csv")

def user_input_features():
  # Entrada
  horas_estudiando = st.number_input('Horas estudiando',  min_value=0, max_value=1, value = 0, step = 1)
  horas_durmiendo = st.number_input('Horas durmiendo:', min_value=0, max_value=230, value = 0, step = 1)
  asistencia = st.number_input('Asistencia', min_value=0, max_value=140, value = 0, step = 1)
  calificaciones_anteriores = st.number_input('Calificaciones', min_value=0, max_value=50, value = 0, step = 1)

  user_input_data = {'hours_studied': horas_estudiando,
                     'sleep_hours': horas_durmiendo,
                     'attendance_percent': asistencia,
                     'previous_scores':   calificaciones_anteriores,

                     }
  features = pd.DataFrame(user_input_data, index=[0])

  return features


df = user_input_features()
datos =  pd.read_csv('student_exam_scores.csv', encoding='latin-1')
columnas_a_excluir = ['student_id', 'previous_scores', 'exam_score']
X = datos.drop(columns=columnas_a_excluir)
y=datos['exam_score']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1613134)
LR = LinearRegression()
LR.fit(X_train,y_train)

b1 = LR.coef_
b0 = LR.intercept_
prediccion = b0 + b1[0]*df['hours_studied'] + b1[1]*df['sleep_hours'] + b1[2]*df['attendance_percent']
st.subheader('Cálculo del examen')
st.write('la calificación de la persona es', prediccion)
