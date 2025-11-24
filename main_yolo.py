import cv2
from ultralytics import YOLO

# ---------------------------------
# CONFIGURACIÓN GENERAL
# ---------------------------------
VIDEO_PATH = "videos/easy.mp4"
COCO_PATH = "data/coco.txt"
MODEL_PATH = "yolov8n.pt"

FRAME_WIDTH = 1020
FRAME_HEIGHT = 500

# Región de interés vertical (solo la fila de plazas)
ROI_Y_MIN = 260
ROI_Y_MAX = 470

# Umbral mínimo de confianza para contar un auto
CAR_CONF_MIN = 0.5

# Porcentaje de reducción del rectángulo de plaza (ej. 0.2 = 20%)
SPOT_SHRINK_FACTOR = 0.2

# Área mínima de intersección para considerar ocupada una plaza
MIN_INTER_AREA = 1500

# Frames consecutivos para confirmar cambio de estado
STABILITY_FRAMES = 4


# ---------------------------------
# PLAZAS DE ESTACIONAMIENTO
# Coordenadas en el frame redimensionado (1020x500)
# (x1, y1, x2, y2)
# ---------------------------------
PARKING_SPOTS = [
    {"id": 1, "coords": (215, 265, 260, 430)},
    {"id": 2, "coords": (270, 265, 315, 430)},
    {"id": 3, "coords": (325, 265, 370, 430)},
    {"id": 4, "coords": (380, 265, 425, 430)},
    {"id": 5, "coords": (435, 265, 480, 430)},
    {"id": 6, "coords": (490, 265, 535, 430)},
    {"id": 7, "coords": (545, 265, 590, 430)},
    {"id": 8, "coords": (600, 265, 645, 430)},
    {"id": 9, "coords": (655, 265, 700, 430)},
]

# Inicializamos estado para cada plaza
for spot in PARKING_SPOTS:
    spot["state"] = "free"          # "free" o "occupied"
    spot["frames_occupied"] = 0
    spot["frames_free"] = 0


def load_class_list(path: str):
    """Carga la lista de clases (COCO) desde un archivo de texto."""
    with open(path, "r") as f:
        classes = f.read().splitlines()
    return classes


def shrink_rect(box, factor: float):
    """Reduce un rectángulo hacia el centro para evitar falsos solapes."""
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1

    dx = int(w * factor / 2)
    dy = int(h * factor / 2)

    return (
        x1 + dx,
        y1 + dy,
        x2 - dx,
        y2 - dy,
    )


def intersection_area(boxA, boxB):
    """Calcula área de intersección entre dos cajas (x1,y1,x2,y2)."""
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB

    x_left = max(ax1, bx1)
    y_top = max(ay1, by1)
    x_right = min(ax2, bx2)
    y_bottom = min(ay2, by2)

    if x_right <= x_left or y_bottom <= y_top:
        return 0

    return (x_right - x_left) * (y_bottom - y_top)


def main():
    # 1) Cargar clases y modelo
    print("[INFO] Cargando clases COCO...")
    class_list = load_class_list(COCO_PATH)

    print("[INFO] Cargando modelo YOLO...")
    model = YOLO(MODEL_PATH)

    # 2) Abrir video
    print(f"[INFO] Abriendo video: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir el video: {VIDEO_PATH}")
        return

    window_name = "Detección YOLO - Estacionamiento (Refinado)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("[INFO] Presiona 'q' para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        # ---------------------------
        # Detecciones YOLO
        # ---------------------------
        results = model.predict(frame, verbose=False)[0]

        car_boxes = []

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

            conf = float(box.conf[0])
            cls_id = int(box.cls[0])

            if 0 <= cls_id < len(class_list):
                class_name = class_list[cls_id]
            else:
                class_name = f"id_{cls_id}"

            # Dibujar SIEMPRE la caja de detección
            label = f"{class_name} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

            # Guardar solo autos con buena confianza dentro de la ROI
            if (
                class_name == "car"
                and conf >= CAR_CONF_MIN
                and y1 >= ROI_Y_MIN
                and y2 <= ROI_Y_MAX
            ):
                car_boxes.append((x1, y1, x2, y2))

        # ---------------------------
        # Evaluar plazas
        # ---------------------------
        occupied_count = 0

        for spot in PARKING_SPOTS:
            sx1, sy1, sx2, sy2 = spot["coords"]

            # Reducir caja para hacerla más estricta internamente
            inner_box = shrink_rect(spot["coords"], SPOT_SHRINK_FACTOR)

            # Ver si algún auto intersecta suficientemente esa zona
            ocupado_instantaneo = False
            for car_box in car_boxes:
                inter_area = intersection_area(inner_box, car_box)
                if inter_area > MIN_INTER_AREA:
                    ocupado_instantaneo = True
                    break

            # Actualizar contadores de estabilidad
            if ocupado_instantaneo:
                spot["frames_occupied"] += 1
                spot["frames_free"] = 0
            else:
                spot["frames_free"] += 1
                spot["frames_occupied"] = 0

            # Confirmamos cambio de estado solo si pasa el umbral de frames
            if spot["frames_occupied"] >= STABILITY_FRAMES:
                spot["state"] = "occupied"
            elif spot["frames_free"] >= STABILITY_FRAMES:
                spot["state"] = "free"

            # Color según estado estable
            if spot["state"] == "occupied":
                color = (0, 0, 255)  # rojo
                occupied_count += 1
                text = f"{spot['id']}: Ocupado"
            else:
                color = (0, 255, 0)  # verde
                text = f"{spot['id']}: Libre"

            # Dibujar el rectángulo EXTERNO de la plaza (visible)
            cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), color, 1)
            cv2.putText(
                frame,
                text,
                (sx1, sy1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1,
                cv2.LINE_AA,
            )

        # ---------------------------
        # Overlay de resumen
        # ---------------------------
        total_spots = len(PARKING_SPOTS)
        free_spots = total_spots - occupied_count
        info_text = f"Plazas: {total_spots} | Ocupadas: {occupied_count} | Libres: {free_spots}"

        # Fondo negro arriba para que se lea siempre
        cv2.rectangle(frame, (0, 0), (FRAME_WIDTH, 30), (0, 0, 0), -1)
        cv2.putText(
            frame,
            info_text,
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Programa finalizado correctamente.")


if __name__ == "__main__":
    main()
