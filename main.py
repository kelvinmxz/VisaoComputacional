from ultralytics import YOLO
import cv2
import numpy as np
import os

class DetectorObjetos:
    def __init__(self):
        # Carrega o modelo YOLOv8n pré-treinado
        self.model = YOLO('yolov8n.pt')
        
        # Configurações da câmera
        self.cap = None
        self.window_name = "Detector de Objetos - YOLOv8"
        
    def iniciar_camera(self):
        """Inicializa a webcam"""
        print("Procurando câmeras...")
        
        for i in range(10):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    print(f"Câmera {i} conectada!")
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    self.cap = cap
                    return True
                cap.release()
        return False

    def detectar_objetos(self, frame):
        """Detecta objetos no frame"""
        results = self.model(frame, conf=0.5)
        
        # Processa resultados
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Extrai informações
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                name = self.model.names[cls]
                
                # Desenha bbox e label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{name} {conf:.2f}", 
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (0, 255, 0), 2)
        
        return frame

    def executar(self):
        """Executa o detector em tempo real"""
        if not self.iniciar_camera():
            print("Erro: Nenhuma câmera encontrada!")
            return

        print("Detector iniciado. Pressione 'q' para sair.")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                # Detecta e mostra objetos
                frame_processado = self.detectar_objetos(frame)
                
                # Mostra FPS
                cv2.putText(frame_processado, f"FPS: {int(self.cap.get(cv2.CAP_PROP_FPS))}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (0, 255, 0), 2)
                
                # Mostra frame
                cv2.imshow(self.window_name, frame_processado)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Encerrando...")
                    break

        except Exception as e:
            print(f"Erro: {e}")
        
        finally:
            self.cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = DetectorObjetos()
    detector.executar()