# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 16:03:56 2022

@author: marco
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import signal
from numpy import linalg
from img_to_binary import np_img_array
import soundfile as sf

# =============================================================================
# Funcion embebido que recibe una señal de audio temporal MONO
# =============================================================================

# Importa audios desde la carpeta
piano, fs = sf.read('Piano.wav')
voz, fs = sf.read('Voz.wav')

def embebido (a, audio):
    """
    Funcion embebido que devuelve la señal de audio en dominio temporal con marca
    de agua

    Parametros:
    -----------
    
    factorfuerza = Factor Fuerza marca de agua (0 < a < 1)
    audio = Señal de audio de voz hablada o musical (Piano.wav o Voz.wav)

    Devuelve:
    ---------

    Señal de audio temporal con marca de agua embebida
    Cálculo de PSNR [dB]
        
    """
     
    b = np_img_array*a
        
    A = audio[:,0] # Toma un solo channel en caso de que sea estéreo
         
    f, t ,A_stft = signal.stft(A)
        
    U, D, V = linalg.svd(A_stft)     # matriz D será embebida con los valores
                                     # del vector binario unidimensional 
    
    D = np.diag(D)
        
    D = np.pad(D,((0,0),(0,A_stft.shape[1]-D.shape[1]))) # Padding
              
    D_1 = D.copy() # Para no modificar la original
                
    
    n = 0
    for i, j in np.ndindex(*D.shape):
        if i != j and n < b.size : # Embeber en los lugares que no sean diag
            D_1[i, j] = b[n]
            n += 1
  
  
    U_w, D_w, V_w = linalg.svd(D_1)
       
    D_w = np.diag(D_w)
            
    D_w = np.pad(D_w,((0,0),(0,D_1.shape[1]-D_w.shape[1]))) # Padding
    
    A_w = U@D_w@V # SVD Proceso inverso
    
    tiempo, A_w_istft = signal.istft(A_w) # Audio con marca de agua
    
# =============================================================================
# Parámetro objetivo (PSNR)
# =============================================================================
             
    A = np.pad(A,(0,abs(A_w_istft.shape[0])-abs(A.shape[0]))) 
    # Padeo para poder realizar el calculo del MSE
            
    mse = (np.square(A - A_w_istft)).mean()
    
    PSNR = 10*np.log10(((max(A))**2)/(mse))
        
    print('Con un factor de fuerza ',a,'se obtuvo un PSNR entre audio y audio con marca de agua: ',round(PSNR,1), 'dB')
      
    return A_w_istft, U, V, D_w, A_w, U_w, V_w, D, D_1, A_stft, b, tiempo, A

# Un test 

ff = input('Ingrese un factor de fuerza (0<a<1): ')

ff = float(ff)

data = input('Seleccione voz o piano: ')

data = str(data.lower())

if data == 'voz':
    data = voz
if data == 'piano':
    data = piano

audio_embebido, U, V, D_w, A_w, U_w, V_w, D, D_a, A_stft, b, eje_tiempo, A = embebido(ff,data)

# =============================================================================
# Varios ataques a audio con marca de agua
# =============================================================================

# Ruido

ruido_gaussiano = (np.random.normal(0,1,audio_embebido.shape))*0.01

audio_ruido = audio_embebido*ruido_gaussiano

# Lowpass filter

fc = 8000

fmax = int(fs/2)

num, den = signal.bessel(3,fc/fmax,btype='low')

t, impulse = signal.impulse((num,den))

audio_lowpass = np.convolve(impulse, audio_embebido,'full')

# Highpass filter

fc2 = 50

num2, den2 = signal.bessel(3,fc2/fmax/fmax,btype='highpass')

t2, impulse2 = signal.impulse((num2,den2))

audio_highpass = np.convolve(impulse2,audio_embebido,'full')

# Combinacion de 3 ataques

audio_3ataques = np.convolve(np.convolve(impulse,audio_ruido),impulse2)

# Pad del audio filtrado por efecto de la convolucion

audio_lowpass = audio_lowpass[:audio_embebido.shape[0]]

audio_highpass = audio_highpass[:audio_embebido.shape[0]]

audio_3ataques = audio_3ataques[:audio_embebido.shape[0]]

# =============================================================================
# Funcion deteccion de marca de agua y comparacion con original
# =============================================================================

def deteccion(audio):
    """
    Funcion deteccion de marca de agua de un audio entrante, mediante razona-
    miento de la referencia "An imperceptible and robust audio watermarking
    algorithm". Se realiza el proceso inverso a la funcion embebido.
    
    Para reconstruir la marca de agua, si un valor de la matriz W1 es menor
    al promedio se entiende que allí fue embebido un '0' y caso contrario 
    un '1' fue embebido

    Parametros:
    -----------
    
    audio = Señal de audio de voz hablada o musical proveniente de la funcion 
    embebido

    Devuelve:
    ---------

    Vector unidimensional binario (Marca de agua "new")
            
    """    
    f, t, A1 = signal.stft(audio) 
    # Toma el audio resultante de funcion anterior, por defecto
           
    U1, D1, V1 = linalg.svd(A1)
    
    D1 = np.diag(D1)
        
    D1 = np.pad(D1,((0,0),(0,A1.shape[1]-D1.shape[1]))) # Padding
    
    W1 = U_w@D1@V_w  # Matrices ortogonales U_w y V_w del proceso de embebido
        
    b1 = []
    
    valores = [0, 1] # Valores de los bits embebidos de la sec binaria
    
    W1_avg = abs(W1.mean())

    k = 0
    for i,j in np.ndindex(*W1.shape):
        if i != j and W1[i,j] <= W1_avg and k < int(b.shape[0]) :
            b1.append(valores[0])
            k += 1
         
        if i != j and W1[i,j] > W1_avg and k < int(b.shape[0]): 
            b1.append(valores[1])
            k += 1
            
    b1 = np.array(b1)
    
    return b1

b_sinataque = deteccion(audio_embebido)    

b_ruido  = deteccion(audio_ruido)

b_lowpass  = deteccion(audio_lowpass)

b_highpass  = deteccion(audio_highpass)

b_3ataques = deteccion(audio_3ataques)

# =============================================================================
# Normalized Correlation
# =============================================================================

def nc(marca_agua):
    """
    Funcion que calcula Normalized Correlation entre dos marca de agua 
    Caso NC=1, las marcas de agua son idénticas y por ende la robustez del
    proceso de embebido es relativamente alta.
    A medida que el valor se aleje de 1, la robustez ira en detrimento

    Parametros:
    -----------
    
    marca_agua = Vector unidimensional marca de agua

    Devuelve:
    ---------
    NC = valor escalar.
    -propio del calculo extraído de la referencia "Hybrid 
    SVD-Based Image Watermarking Schemes: A Review"-
    """
     
    for i in range(len(b)):
        numer = np.sum((b-b.mean())*(marca_agua-marca_agua.mean()))
        denom = np.sum(np.sqrt((b-b.mean())**2)*np.sqrt((marca_agua-marca_agua.mean())**2))
        NC = numer/denom
        return NC   

NC_lowpass = nc(b_lowpass)

NC_highpass = nc(b_highpass)

NC_ruido = nc(b_ruido)

NC_3ataques = nc(b_3ataques)

# =============================================================================
# Bit error rate 
# =============================================================================

BER = []

valores = [0,1]

def BitErrorRate(b_extraida):
    """
    Funcion que calcula el Bit Error Rate, donde a mayor error encontrado
    al momento de comparar los bits extraidos con la secuencia binaria original
    mayor es el BER.

    Parametros:
    -----------
    
    b_extraida = Vector unidimensional marca de agua

    Devuelve:
    ---------
    BER_res = valor escalar.
    
    -propio del calculo extraído de la referencia "An imperceptible and 
    robust audio watermarking algorithm"-
    """
    for j in range(len(b)):
        if b[j] == b_extraida[j]:
            BER.append(valores[1])
        else:
            BER.append(valores[0])
    BER_res = 100*(np.sum(BER)/len(b))
    
    return BER_res

# Calculo relativo de BER, en funcion del valor BER de la marca de agua
# comparada con ella misma

BER_sinataque = BitErrorRate(b)

BER_lowpass = BitErrorRate(b_lowpass)/BER_sinataque 

BER_highpass = BitErrorRate(b_highpass)/BER_sinataque

BER_ruido = BitErrorRate(b_ruido)/BER_sinataque

BER_3ataques = BitErrorRate(b_3ataques)/BER_sinataque

# =============================================================================
# Print por consola de resultados
# =============================================================================
 
Ataque = ['Ruido', 'LF 8kHz', 'HF 50Hz', 'R+HF+LF'] # Cantidad de pruebas

NC = [NC_lowpass, NC_highpass, NC_ruido, NC_3ataques] # Resultados guardados

BER_resultados = [BER_lowpass,BER_highpass,BER_ruido,BER_3ataques] # Resultados guardados

print('')
print('Resultados obtenidos:')
print('')
print('Para factor de fuerza ', ff)
print('')
print('{:^10}{:^10}{:^10}{:^10}{:^10}'.format('Ataque',' |','NC',' |','BER'));
print('------------------------------------------------------')
for i in range(len(Ataque)):
    print('{:^10}{:^10}{:^10}{:^10}{:^10}'.format(Ataque[i],' |',round(NC[i],4),' |',round(BER_resultados[i],2)));

# =============================================================================
# Ploteo
# =============================================================================

# Normaliza para graficos

audio_embebido /= max(audio_embebido)
audio_ruido /= max(audio_ruido)
audio_lowpass /= max(audio_lowpass)
audio_highpass /= max(audio_highpass)
audio_3ataques /= max(audio_3ataques)

# Eje en segundos

eje_tiempo = np.arange(0,len(audio_embebido)/fs,1/fs)

plt.figure(1,figsize=(15,9))
plt.subplot(5,1,1)
plt.title('Audio con marca de agua')
plt.plot(eje_tiempo,audio_embebido)
plt.subplot(5,1,2)
plt.title('Audio con marca de agua + Ruido')
plt.plot(eje_tiempo,audio_ruido)
plt.subplot(5,1,3)
plt.title('Audio con marca de agua + LF 8 kHz')
plt.plot(eje_tiempo,audio_lowpass)
plt.subplot(5,1,4)
plt.title('Audio con marca de agua + HF 50 Hz')
plt.plot(eje_tiempo,audio_highpass)
plt.subplot(5,1,5)
plt.title('Audio con marca de agua + R+HF+LF')
plt.plot(eje_tiempo,audio_3ataques)
plt.tight_layout()
