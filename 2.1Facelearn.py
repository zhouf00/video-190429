import cv2  # 计算机视觉包


# 认得人脸长什么样子
# "haarcascade_frontalface_default.xml"需要指定路径
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


def detect(gray, frame):
    # 处置黑白画面 用人脸1.3倍大小的框框把脸标记出来，框框的线条粗细是5个像素
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # 在哪里画框框？
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h),(255, 0, 0), 2)
        roi_gray = gray[y: y+h, x:x+w]
        roi_color = frame[y: y+h, x:x+w]

    return frame

# 开启摄像头
video_capture = cv2.VideoCapture(0)

while True:
    _, frame = video_capture.read()  # 读取摄像头采集到的画面
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 把画面转成黑白
    canvas = detect(gray, frame)  # 把结果展示画布上面
    cv2.imshow("Video", canvas)  # 把结果展示出来
    if cv2.waitKey(1)& 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()