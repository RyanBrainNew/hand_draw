"""
Author：ljl
Date: 2023-04-19
功能：识别手部并绘制手部线条
1、使用OpenCV读取摄像头视频流；
2、识别手掌关键点像素坐标；
"""

# 导入OpenCV
import cv2
# 导入mediapipe
import mediapipe as mp


class HandDraw:
    def __init__(self):
        # 初始化medialpipe
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands

    # 主函数
    def recognize(self):
        # OpenCV读取视频流
        cap = cv2.VideoCapture(0)
        # 视频分辨率
        resize_w = 640
        resize_h = 480

        with self.mp_hands.Hands(min_detection_confidence=0.7,
                                 min_tracking_confidence=0.5,
                                 max_num_hands=2) as hands:
            while cap.isOpened():
                success, image = cap.read()
                image = cv2.resize(image, (resize_w, resize_h))

                if not success:
                    print("空帧.")
                    continue

                # 提高性能
                image.flags.writeable = False
                # 转为RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # 镜像
                image = cv2.flip(image, 1)
                # mediapipe模型处理
                results = hands.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                # 判断是否有手掌
                if results.multi_hand_landmarks:
                    # 遍历每个手掌
                    for hand_landmarks in results.multi_hand_landmarks:
                        # 在画面标注手指
                        self.mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing_styles.get_default_hand_landmarks_style(),
                            self.mp_drawing_styles.get_default_hand_connections_style())

                # 显示画面
                cv2.imshow('MediaPipe Hands', image)
                if cv2.waitKey(5) & 0xFF == 27 or cv2.getWindowProperty('MediaPipe Hands', cv2.WND_PROP_VISIBLE) < 1:
                    break
            cap.release()


# 开始程序
control = HandDraw()
control.recognize()
