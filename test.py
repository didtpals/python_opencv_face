import cv2
import numpy as np
import os

datafile_path = "didtpals_img" # 저장될 이미지의 경로 설정

face_cascade1 = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") # 학습된 정면 얼굴 인식 인공지능 xml파일 호출 
# Load the face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create() # 얼굴 인식기를 호출
recognizer.read("didtpals/training_file.yml") # 얼굴 인식기를 통해 얼굴 이미지가 학습 되어있는 yml파일을 불러와 캠에 얼굴과 비교

save_file_name = "" # 캡쳐 이미지 저장 이름 변수

known_faces = {} # 인식된 얼굴과 그에 상응하는 얼굴 이미지 이름 목록을 정의
i = 0 # i 변수를 0으로 설정
for name in os.listdir(datafile_path): # 지정된 경로의 모든 파일과 디렉토리 목록을 가져옴
    i += 1 # i 변수를 1씩 더해주면서 파일 안에 있는 이미지들을 순차적으로 호출

    known_faces[name] = i # 이미지 파일의 이름을 불러옴


cap = cv2.VideoCapture(0) # 카메라(0) 지정

while True:

    ret, frame = cap.read() # cam 변수에 지정된 카메라 호출

    frame = cv2.flip(frame, 2) # 카메라 화면 좌우 반전

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # 카메라 화면을 흑백으로 설정하여 노이즈를 제거

    faces1 = face_cascade1.detectMultiScale( # 얼굴 위치의 좌표를 반환 해줌
    gray, # 이미지 호출               
    scaleFactor=1.05, # 이미지 확대 크기 제한, 1.3~1.5 (큰값: 인식기회 증가, 속도 감소) 
    minNeighbors=5, # 요구되는 이웃 수(큰값: 품질증가, 검출개수 감소)
    minSize=(150, 150)) # 얼굴 인식 범위 설정

    # 정면 얼굴 인식하기
    for (x, y, w, h) in faces1: # 위 변수에 설정된 좌표를 확인 후 인식 
        face_gray = gray[y:y+h, x:x+w] # 얼굴 부분만 가져오기
        id, confidence = recognizer.predict(face_gray) # 얼굴이 인식된 부분을 변수에 삽입

        # 얼굴 인식 여부를 확인해 카메라 화면에 학습된 이미지와 상응하는 이름을 띄움
        if confidence > 50: 
            name = list(known_faces.keys())[list(known_faces.values()).index(id)]
            save_file_name = name
            confidence = "  {0}%".format(round(confidence)) # 학습된 이미지와 n% 맞는지 표시

            # 얼굴 인식 여부를 확인해 학습된 이미지와 맞지 않다면 Unknown
        else:
            name = "Unknown" # 얼굴 인식 여부를 확인해 학습된 이미지와 얼굴이 50% 이상 맞지 않다면 Unknown
            confidence = "  {0}%".format(round(confidence)) 

            # 프레임에 이름과 얼마나 일치 하는지 그려줌
        cv2.putText(frame, name, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, confidence, (x+5, y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)

        
    cv2.imshow("Webcam", frame) # 위 작업이 완료된 카메라 화면에 표시

    # 특정 키를 지정
    key = cv2.waitKeyEx(10) # 키 입력을 대기, m/s의 단위로 표현

    if key != -1:
        key = chr(key) # 아스키 코드를 문자열로 바꿔줌
            
        if key == "c": #c를 누르면 카메라 화면 캡쳐
            print("캡쳐")
            if save_file_name != "Unknown" and len(save_file_name) != 0: # 인식된 얼굴이 Unknown일 경우 캡쳐 이미지를 저장하지 않음
                cv2.imwrite(f"./didtpals_img/{save_file_name}", frame) # 캡쳐된 이미지를 지정된 경로의 인식된 얼굴 이미지의 이름으로 저장됨

        elif key == "e": # e를 누르면 break
            print("종료")
            break

cap.release() # 카메라 이미지 해제
cv2.destroyAllWindows() # 모든 윈도우 창 제거