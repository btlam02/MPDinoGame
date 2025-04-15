import cv2
import mediapipe as mp
from pynput.keyboard import Key, Controller

# Khởi tạo keyboard controller
kb = Controller()
space_pressed = False  # Biến trạng thái phím space

# Khởi tạo mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Mở webcam
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Không nhận được khung hình từ webcam.")
            continue

        # Xử lý ảnh đầu vào
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Xử lý ảnh để hiển thị
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        height, width, _ = image.shape

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Vẽ tay
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                # Lấy vị trí các đốt ngón tay (chỉ lấy ngón trỏ, giữa, áp út)
                fingers_up = (
                    hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y and  # Ngón trỏ
                    hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y and  # Ngón giữa
                    hand_landmarks.landmark[16].y < hand_landmarks.landmark[14].y  # Ngón áp út
                )

                if fingers_up and not space_pressed:
                    kb.press(Key.space)
                    space_pressed = True
                    print("Đã nhấn SPACE")
                elif not fingers_up and space_pressed:
                    kb.release(Key.space)
                    space_pressed = False
                    print("Đã nhả SPACE")

        # Hiển thị hình ảnh
        cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:  # Nhấn ESC để thoát
            break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
