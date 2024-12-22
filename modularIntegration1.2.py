"""
 Contiene mejoras en cuanto al tiempo de deteccion 
 de movimiento y la eliminacion e los colores para la 
 asignacion del area a detectar
"""

import cv2
import numpy as np
import time
import os
from datetime import datetime

#librerias para prediccion
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array
#Ruta de la imagen no dinamica aun para predicc
#  guardar la imagen
CAPTURE_DELAY = 3  # Tiempo en segundos de movimiento continuo antes de tomar la captura

# Ruta para guardar la imagen capturada
data_dir = r"C:\\Users\\angel\\Desktop\\CUCEI\\INCO 9\\Modular\\codigo\\modular_project\\integration"
file_name = "PuroCV.jpg"
os.makedirs(data_dir, exist_ok=True)
file_path = os.path.join(data_dir, file_name)


def enviar_correo(ahora):
    import os
    import base64
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.mime.base import MIMEBase
    from email import encoders
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError

    # Alcances necesarios para enviar correos
    SCOPES = ['https://www.googleapis.com/auth/gmail.send']

    def autenticar():
        """Autenticar al usuario con OAuth2 usando el archivo credentials.json."""
        creds = None
        
        # Construir ruta dinámica al archivo credentials.json
        script_dir = os.path.dirname(os.path.abspath(__file__))
        credentials_path = os.path.join(script_dir, 'credentials.json')
        
        # Verificar si ya existen credenciales guardadas
        if os.path.exists('token.json'):
            creds = Credentials.from_authorized_user_file('token.json', SCOPES)
        # Si no hay credenciales válidas, se realiza la autenticación
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    credentials_path, SCOPES)
                creds = flow.run_local_server(port=0)
            # Guardar las credenciales para futuras ejecuciones
            with open('token.json', 'w') as token:
                token.write(creds.to_json())
        return creds

    def enviar_correo_con_imagen(destinatario, asunto, mensaje, nombre_imagen):
        """Enviar un correo electrónico con una imagen adjunta utilizando la API de Gmail."""
        try:
            creds = autenticar()
            service = build('gmail', 'v1', credentials=creds)

            # Crear el mensaje MIME
            mensaje_mime = MIMEMultipart()
            mensaje_mime['to'] = destinatario
            mensaje_mime['subject'] = asunto
            mensaje_mime.attach(MIMEText(mensaje, 'plain'))

            # Construir la ruta de la imagen
            script_dir = os.path.dirname(os.path.abspath(__file__))
            ruta_imagen = os.path.join(script_dir, nombre_imagen)

            # Adjuntar la imagen
            with open(ruta_imagen, 'rb') as imagen:
                adjunto = MIMEBase('application', 'octet-stream')
                adjunto.set_payload(imagen.read())
            encoders.encode_base64(adjunto)
            adjunto.add_header(
                'Content-Disposition',
                f'attachment; filename={os.path.basename(ruta_imagen)}'
            )
            mensaje_mime.attach(adjunto)

            # Convertir el mensaje a base64
            mensaje_base64 = base64.urlsafe_b64encode(mensaje_mime.as_bytes()).decode('utf-8')

            # Crear el cuerpo del mensaje para la API
            mensaje_api = {'raw': mensaje_base64}

            # Enviar el correo
            enviado = service.users().messages().send(userId="me", body=mensaje_api).execute()
            print(f"Correo enviado exitosamente con ID: {enviado['id']}")
        except FileNotFoundError:
            print(f"Error: El archivo '{nombre_imagen}' no se encontró en la ruta '{ruta_imagen}'")
        except HttpError as error:
            print(f"Error al enviar el correo: {error}")

    # Llamar a la función para enviar un correo con la imagen usando la ruta completa
    enviar_correo_con_imagen(
        "angel1000sv@gmail.com", 
        "Alerta!, deteccion de movimiento", 
        f"Se detecto a este sujeto merodeando el area en la siguiente fecha {ahora}",
        r"C:\\Users\\angel\\Desktop\\CUCEI\\INCO 9\\Modular\\codigo\\modular_project\\imagenPruebaCVHoy.jpg\\integration\\PuroCV.jpg"  # Ruta completa al archivo
    )


def predecir():
    if file_name:
        model = load_model(r"C:\\Users\\angel\\Desktop\\CUCEI\\INCO 9\\Modular\\codigo\\modular_project\\modelo10_transfer_learning_person.h5")
        classes = {0: 'cat', 1: 'dog', 2: 'person', 3: 'doors'}
        image_path = r"C:\\Users\\angel\\Desktop\\CUCEI\\INCO 9\\Modular\\codigo\\modular_project\\integration\\PuroCV.jpg"
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Not found image {image_path}")
        else:
            #RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Redimensionar y normalizar
            resized_image = cv2.resize(image_rgb, (224, 224))
            image_array = img_to_array(resized_image) / 255.0
            image_array = np.expand_dims(image_array, axis=0) 
            # Realizar predicción
            predictions = model.predict(image_array, verbose=0)[0]
            print(" % category:")
            for idx, score in enumerate(predictions):
                print(f"{classes[idx]}: {score * 100:.2f}%")
            predicted_class = np.argmax(predictions)
            confidence = predictions[predicted_class] * 100
            predicted_label = classes[predicted_class]

            # Imprimir la categoría predicha
            print(f"Image belong to y: {predicted_label} ({confidence:.2f}%)")

            # Mostrar la imagen con matplotlib y el título con la categoría predicha
            plt.imshow(image_rgb)
            plt.title(f"{predicted_label} ({confidence:.2f}%)")
            plt.axis('off')  # Ocultar los ejes
            plt.show()
            print("Finishing prediction...")
            time.sleep(5)
            print("Finished")

    print("Error: Not found image")


# Abre la cámara USB con DirectShow
video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not video.isOpened():
    print("Error: No se puede abrir la cámara.")
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
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            x1, y1, w, h = cv2.boundingRect(approx)
            x2, y2 = x1 + w, y1 + h
            rectangles.append((x1, y1, x2, y2))

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    elapsed_time = time.time() - start_time
    if elapsed_time >= 5:
        break

# Dividir los rectángulos en tercios horizontales
height, width, _ = frame.shape
third_width = width // 3

def sort_rects(rects):
    return sorted(rects, key=lambda r: (r[2] - r[0]) * (r[3] - r[1]), reverse=True)

rects_plane1 = [r for r in rectangles if r[0] < third_width]
rects_plane2 = [r for r in rectangles if third_width <= r[0] < 2 * third_width]
rects_plane3 = [r for r in rectangles if r[0] >= 2 * third_width]

rects_plane1 = sort_rects(rects_plane1)
rects_plane2 = sort_rects(rects_plane2)
rects_plane3 = sort_rects(rects_plane3)

rectangles = []
if rects_plane1: rectangles.append(rects_plane1[0])
if rects_plane2: rectangles.append(rects_plane2[0])
if rects_plane3: rectangles.append(rects_plane3[0])

for i, rect in enumerate(rectangles):
    x1, y1, x2, y2 = rect
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(frame, f"{i + 1}", (x1 + 10, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

cv2.imshow('Frame', frame)
cv2.waitKey(0)

user_input = int(input("Seleccione el número del rectángulo que desea (1-3): "))
if 1 <= user_input <= 3:
    selected_rect = rectangles[user_input - 1]

if selected_rect:
    x1, y1, x2, y2 = selected_rect
    rect_width = x2 - x1
    rect_height = y2 - y1
    print(f"Dimensiones del área seleccionada: {rect_width}x{rect_height} píxeles")
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    # Espera 5 segundos después de pintar el rectángulo
    print("Esperando 5 segundos antes de comenzar la detección...")
    time.sleep(5)
i = 0
start_time = time.time()
motion_start_time = None
ss_taken = False
show_motion = True  # Flag para mostrar cuadros verdes

while True:
    ret, frame = video.read()
    if not ret:
        break

    if selected_rect:
        x1, y1, x2, y2 = selected_rect
        roi = frame[y1:y2, x1:x2]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        if i == 20:
            bgGray = gray_roi
        if i > 20:
            dif = cv2.absdiff(gray_roi, bgGray)
            _, th = cv2.threshold(dif, 40, 255, cv2.THRESH_BINARY)
            cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            cv2.imshow('th', th)

            motion_detected = False
            for c in cnts:
                x, y, w, h = cv2.boundingRect(c)

                min_width = int(0.1 * rect_width)
                min_height = int(0.1 * rect_height)

                if w >= min_width and h >= min_height:
                    if show_motion:
                        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.rectangle(frame, (x1 + x, y1 + y), (x1 + x + w, y1 + y + h), (0, 255, 0), 2)
                    motion_detected = True

            if motion_detected:
                if motion_start_time is None:
                    # Inicio de detección de movimiento
                    motion_start_time = time.time()
                elif time.time() - motion_start_time >= CAPTURE_DELAY:
                    # Movimiento continuo detectado por al menos CAPTURE_DELAY segundos
                    if not ss_taken:
                        # Apagar cuadros verdes
                        show_motion = False
                        cv2.imshow('Frame Limpio', frame)
                        cv2.waitKey(1)

                        # Guardar el screenshot
                        cv2.imwrite(file_path, frame)
                        print(f"Imagen guardada en: {file_path}")
                        ss_taken = True

                        # Detener el uso de la cámara y cerrar el programa
                        video.release()
                        cv2.destroyAllWindows()
                        ahora = datetime.now()
                        predecir()
                        enviar_correo(ahora)
                        print("Guardado con éxito. Cámara detenida.")
                        break
            else:
                # Reiniciar el tiempo si no hay movimiento
                motion_start_time = None

        i += 1

        if show_motion:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video.release()
cv2.destroyAllWindows()


