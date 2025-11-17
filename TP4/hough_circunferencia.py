import cv2
import numpy as np

def detectar_circulos_hough(
    ruta_imagen,
    mostrar_resultados=True,
    guardar_resultado=True,
    ruta_salida="resultado_hough_circulos.png",
    radio_aproximado=None
):
    #Cargar imagen
    imagen_bgr = cv2.imread(ruta_imagen)
    if imagen_bgr is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen desde: {ruta_imagen}")

    imagen_resultado = imagen_bgr.copy()

    imagen_gray = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY)

    #Suavizado para reducir ruido
    imagen_blur = cv2.medianBlur(imagen_gray, 5)

    # Definir rango de radios
    if radio_aproximado is not None:
        minRadius = int(radio_aproximado * 0.9)
        maxRadius = int(radio_aproximado * 1.0)
    else:
        minRadius = 10     # ajustar según el caso sencillo que uses
        maxRadius = 0      # 0 = sin máximo definido

    # Transformada de Hough para circunferencias
    circulos = cv2.HoughCircles(
        imagen_blur,
        method=cv2.HOUGH_GRADIENT,
        dp=1.2,               # relación de resolución: >1 reduce tamaño del acumulador
        minDist=20,           # distancia mínima entre centros de circunferencias
        param1=100,           # umbral superior de Canny interno
        param2=30,            # umbral de votos en el acumulador (más alto = menos círculos)
        minRadius=minRadius,
        maxRadius=maxRadius
    )

    # Dibujar circunferencias 
    if circulos is not None:
        circulos = np.uint16(np.around(circulos))
        for c in circulos[0, :]:
            x, y, r = c
            # circunferencia
            cv2.circle(imagen_resultado, (x, y), r, (0, 255, 0), 2)
            # centro
            cv2.circle(imagen_resultado, (x, y), 2, (0, 0, 255), 3)
        print(f"Se detectaron {len(circulos[0])} circunferencias.")
    else:
        print("No se detectaron circunferencias con los parámetros actuales.")

    # resultados
    if mostrar_resultados:
        cv2.imshow("Imagen original", imagen_bgr)
        cv2.imshow("Imagen suavizada (gris)", imagen_blur)
        cv2.imshow("Circunferencias detectadas (Hough)", imagen_resultado)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

  
    if guardar_resultado:
        cv2.imwrite(ruta_salida, imagen_resultado)
        print(f"Resultado guardado en: {ruta_salida}")

    return circulos


if __name__ == "__main__":
    # se pueden cambiar las imagenes para probar distintas imagenes
    ruta = "block.png"

    # Si conocés aproximadamente el radio del aro en píxeles, lo pasás acá, por ejemplo 60
    detectar_circulos_hough(
        ruta_imagen=ruta,
        radio_aproximado=100  # o un número, ej: 60
    )
