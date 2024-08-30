import cv2
import numpy as np
import time

# Abre la cámara USB con DirectShow
video = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Cambia el índice según el identificado

if not video.isOpened():
    print("Error al abrir la cámara.")
    exit()

rectangles = []
selected_rect = None

# Marca el tiempo de inicio
start_time = time.time()

# Búsqueda de rectángulos durante 5 segundos
while True:
    ret, frame = video.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # Aplicar desenfoque para reducir el ruido
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:  # Detecta un cuadrilátero
            x1, y1, w, h = cv2.boundingRect(approx)
            x2, y2 = x1 + w, y1 + h
            rectangles.append((x1, y1, x2, y2))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Rectángulos en verde

    cv2.imshow('Frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    elapsed_time = time.time() - start_time
    if elapsed_time >= 5:
        break

# Ordenar rectángulos por tamaño (área) y seleccionar los 3 más grandes
rectangles = sorted(rectangles, key=lambda r: (r[2] - r[0]) * (r[3] - r[1]), reverse=True)
if len(rectangles) > 3:
    rectangles = rectangles[:3]

# Numerar y mostrar los 3 rectángulos más grandes
for i, rect in enumerate(rectangles):
    x1, y1, x2, y2 = rect
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Rectángulos en azul
    cv2.putText(frame, f"{i + 1}", (x1 + 10, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

# Mostrar el frame con los rectángulos numerados
cv2.imshow('Frame', frame)
cv2.waitKey(0)  # Espera hasta que el usuario presione una tecla

# Seleccionar rectángulo basado en la entrada del usuario
user_input = int(input("Seleccione el número del rectángulo que desea (1-3): "))
if 1 <= user_input <= 3:
    selected_rect = rectangles[user_input - 1]

i = 0
start_time = time.time()  # Reiniciar el tiempo de inicio para la lógica de procesamiento de la ROI

while True:
    ret, frame = video.read()
    if not ret:
        break

    if selected_rect:
        x1, y1, x2, y2 = selected_rect
        roi = frame[y1:y2, x1:x2]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        elapsed_time = time.time() - start_time
        if elapsed_time < 5:  # Mostrar mensaje durante los primeros 5 segundos después de seleccionar la ROI
            # Dibuja el rectángulo rojo de la ROI en el frame original
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            # Muestra el mensaje "Área seleccionada" dentro de la ROI
            cv2.putText(frame, "Area seleccionada", (x1 + 10, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        else:
            if i == 20:
                bgGray = gray_roi
            if i > 20:
                dif = cv2.absdiff(gray_roi, bgGray)
                _, th = cv2.threshold(dif, 40, 255, cv2.THRESH_BINARY)
                cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Muestra la imagen umbralizada de la ROI
                cv2.imshow('th', th)

                for c in cnts:
                    area = cv2.contourArea(c)
                    if area > 3000:
                        x, y, w, h = cv2.boundingRect(c)
                        # Dibuja el rectángulo en la ROI y luego ajusta las coordenadas al frame original
                        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        # Ajustar las coordenadas al frame original
                        cv2.rectangle(frame, (x1 + x, y1 + y), (x1 + x + w, y1 + y + h), (0, 255, 0), 2)

            i += 1

        # Dibuja el rectángulo de la ROI en el frame original
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    # Muestra el frame original
    cv2.imshow('Frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
