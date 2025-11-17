import cv2
import numpy as np

def detectar_rectas_hough(
    ruta_imagen,
    mostrar_resultados=True,
    guardar_resultado=True,
    ruta_salida="resultado_hough_lineas.png"
):
 

    # Cargar imagen
    imagen_bgr = cv2.imread(ruta_imagen)
    if imagen_bgr is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen desde: {ruta_imagen}")

    # Copia para dibujar las rectas
    imagen_resultado = imagen_bgr.copy()

    imagen_gray = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY)

    # Suavizado para reducir ruido 
    imagen_blur = cv2.GaussianBlur(imagen_gray, (5, 5), 1)

    #  Detección de bordes con Canny
    bordes = cv2.Canny(imagen_blur, threshold1=50, threshold2=150, apertureSize=3)

    # Transformada de Hough para rectas
    lineas = cv2.HoughLines(
        image=bordes,
        rho=1,
        theta=np.pi / 180,
        threshold=160  # ajustable: menor valor -> detecta más líneas
    )

    # Dibujar rectas detectadas
    if lineas is not None:
        for linea in lineas:
            rho, theta = linea[0]

            # A partir de (rho, theta) reconstruimos dos puntos de la recta
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho

            # Puntos suficientemente alejados para que la recta atraviese la imagen
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            # Dibujar la recta en rojo
            cv2.line(imagen_resultado, (x1, y1), (x2, y2), (0, 0, 255), 2)
    else:
        print("No se detectaron rectas con los parámetros actuales.")

    # Mostrar resultados (opcional)
    if mostrar_resultados:
        cv2.imshow("Rectas detectadas (Hough)", imagen_resultado)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Guardar imagen resultado (opcional)
    if guardar_resultado:
        cv2.imwrite(ruta_salida, imagen_resultado)
        print(f"Resultado guardado en: {ruta_salida}")


if __name__ == "__main__":
    ruta = "block5.png"
    detectar_rectas_hough(ruta)

