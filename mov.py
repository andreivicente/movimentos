"""
Captura de Movimento em Libras
Dependências: pip install mediapipe opencv-python numpy
O arquivo do modelo precisa ser baixado separadamente (veja comentários no código)
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np
from collections import deque
import time

# ==================== ÁREA DE CONFIGURAÇÃO ====================
MODEL_PATH = "pose_landmarker_full.task"  # Baixe em: https://storage.googleapis.com/mediapipe-assets/pose_landmarker_full.task
SMOOTHING_ALPHA = 0.6        # Coeficiente de suavização exponencial (0-1, quanto maior, mais suave)
HISTORY_SIZE = 5             # Número de frames históricos para cálculo de movimento
VISIBILITY_THRESHOLD = 0.7   # Limite de confiabilidade dos pontos-chave
MOTION_THRESHOLD = 0.005     # Limite de detecção de movimento (indica se o sinal está em movimento)

class CapturaMovimentoLibras:
    """Classe principal de captura de movimento em Libras"""
    
    def __init__(self):
        # Inicializa o detector de Pose
        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.pose_landmarker = vision.PoseLandmarker.create_from_options(options)
        
        # Inicializa o detector de Mãos
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Estado de suavização
        self.smoothed_pose = None
        self.smoothed_hands = None
        
        # Histórico de frames para cálculo de movimento
        self.pose_history = deque(maxlen=HISTORY_SIZE)
        self.hand_history = deque(maxlen=HISTORY_SIZE)
        
        # Ferramentas de desenho
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
    
    def _extrair_pontos_pose(self, result):
        """Extrai pontos-chave da Pose, filtrando pontos de baixa confiabilidade"""
        if not result.landmarks or len(result.landmarks) == 0:
            return None
        
        landmarks = result.landmarks[0]  # Pega apenas a primeira pessoa
        arr = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks])
        
        # Pontos de baixa confiabilidade são marcados como NaN
        arr[arr[:, 3] < VISIBILITY_THRESHOLD, :3] = np.nan
        return arr
    
    def _extrair_pontos_maos(self, hand_landmarks_list):
        """Extrai pontos-chave das mãos (21 pontos por mão)"""
        if not hand_landmarks_list:
            return None
        
        hands_data = []
        for hand_landmarks in hand_landmarks_list:
            arr = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
            hands_data.append(arr)
        
        # Garante número fixo (2 mãos), preenche com zeros se faltar
        while len(hands_data) < 2:
            hands_data.append(np.zeros((21, 3)))
        
        return np.array(hands_data)  # shape: (2, 21, 3)
    
    def _suavizacao_exponencial(self, atual, anterior, alpha=SMOOTHING_ALPHA):
        """Suavização exponencial, ignorando valores NaN"""
        if anterior is None:
            return atual.copy()
        
        mask = ~np.isnan(atual[..., 0])
        resultado = anterior.copy()
        resultado[mask] = alpha * atual[mask] + (1 - alpha) * anterior[mask]
        return resultado
    
    def processar_frame(self, frame, timestamp_ms):
        """
        Processa um único frame e retorna:
        - annotated_frame: imagem com anotações
        - motion_data: dicionário com dados de movimento
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # ---- Detecção de Pose ----
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        pose_result = self.pose_landmarker.detect_for_video(mp_image, timestamp_ms)
        pose_landmarks = self._extrair_pontos_pose(pose_result)
        
        if pose_landmarks is not None:
            self.smoothed_pose = self._suavizacao_exponencial(
                pose_landmarks, self.smoothed_pose
            )
            self.pose_history.append(self.smoothed_pose.copy())
        
        # ---- Detecção de Mãos ----
        hand_result = self.hands.process(rgb)
        hand_landmarks = self._extrair_pontos_maos(
            hand_result.multi_hand_landmarks
        )
        
        if hand_landmarks is not None:
            self.smoothed_hands = self._suavizacao_exponencial(
                hand_landmarks, self.smoothed_hands
            )
            self.hand_history.append(self.smoothed_hands.copy())
        
        # ---- Desenho dos pontos-chave ----
        annotated = frame.copy()
        
        # Desenha Pose
        if pose_result.landmarks:
            for landmarks in pose_result.landmarks:
                self.mp_drawing.draw_landmarks(
                    annotated,
                    landmarks,
                    mp.solutions.pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                        color=(0, 255, 0), thickness=2, circle_radius=2
                    ),
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(
                        color=(0, 200, 0), thickness=2
                    )
                )
        
        # Desenha Mãos
        if hand_result.multi_hand_landmarks:
            for hand_landmarks in hand_result.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    annotated,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    connection_drawing_spec=self.mp_drawing_styles.get_default_hand_connections_style()
                )
        
        # ---- Cálculo de movimento ----
        motion_data = self._calcular_movimento()
        
        return annotated, motion_data
    
    def _calcular_movimento(self):
        """Calcula os vetores de movimento das mãos e do corpo"""
        resultado = {
            "is_moving": False,
            "hand_motion": 0.0,
            "body_motion": 0.0,
            "left_hand_moving": False,
            "right_hand_moving": False
        }
        
        # Movimento das mãos
        if len(self.hand_history) >= 2:
            atual = self.hand_history[-1]  # (2, 21, 3)
            anterior = self.hand_history[-2]
            
            # Deslocamento médio de cada mão
            for i, nome_mao in enumerate(["right", "left"]):
                deslocamento = np.nanmean(
                    np.linalg.norm(atual[i] - anterior[i], axis=1)
                )
                if deslocamento > MOTION_THRESHOLD:
                    resultado[f"{nome_mao}_hand_moving"] = True
                    resultado["is_moving"] = True
                resultado["hand_motion"] = max(resultado["hand_motion"], deslocamento)
        
        # Movimento do corpo (braços e tronco)
        if len(self.pose_history) >= 2:
            atual = self.pose_history[-1]
            anterior = self.pose_history[-2]
            
            # Pontos-chave de interesse (índices do MediaPipe Pose)
            indices_superior = [11, 12, 13, 14, 15, 16, 23, 24]  # ombros, cotovelos, pulsos, quadris
            validos = ~np.isnan(atual[indices_superior, 0])
            
            if np.any(validos):
                deslocamento = np.nanmean(
                    np.linalg.norm(
                        atual[indices_superior][validos, :3] - 
                        anterior[indices_superior][validos, :3],
                        axis=1
                    )
                )
                resultado["body_motion"] = deslocamento
                if deslocamento > MOTION_THRESHOLD:
                    resultado["is_moving"] = True
        
        return resultado
    
    def fechar(self):
        """Libera os recursos"""
        self.hands.close()
        self.pose_landmarker.close()


def main():
    """Loop principal: captura vídeo e exibe"""
    captura = CapturaMovimentoLibras()
    cap = cv2.VideoCapture(0)  # Câmera padrão
    
    if not cap.isOpened():
        print("Erro: não foi possível abrir a câmera")
        return
    
    # Define a resolução
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("Captura de Movimento em Libras iniciada")
    print("Pressione 'q' para sair | Pressione 's' para salvar o frame atual")
    
    frame_count = 0
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Timestamp (milissegundos)
        timestamp_ms = int((time.time() - start_time) * 1000)
        
        # Processa o frame
        annotated, motion = captura.processar_frame(frame, timestamp_ms)
        
        # Exibe o estado do movimento
        cor_status = (0, 0, 255) if motion["is_moving"] else (0, 255, 0)
        texto_status = "MOVENDO" if motion["is_moving"] else "ESTATICO"
        cv2.putText(annotated, f"Libras: {texto_status}", (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, cor_status, 3)
        
        # Detalhes do movimento das mãos
        status_esq = "E: MOV" if motion["left_hand_moving"] else "E: ---"
        status_dir = "D: MOV" if motion["right_hand_moving"] else "D: ---"
        cv2.putText(annotated, f"{status_dir}  {status_esq}", (10, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Mostra o frame
        cv2.imshow("Captura de Movimento - Libras", annotated)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Salva o frame anotado atual
            nome_arquivo = f"captura_libras_{frame_count:04d}.jpg"
            cv2.imwrite(nome_arquivo, annotated)
            print(f"Salvo: {nome_arquivo}")
            frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    captura.fechar()


if __name__ == "__main__":
    main()