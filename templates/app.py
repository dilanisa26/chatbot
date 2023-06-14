from flask import Flask, render_template, Response, request, jsonify
from process import preparation, botResponse, process_video_deadlift, process_video_squat
import cv2
import numpy as np
import mediapipe as mp
import numpy as np #perintah untuk mengimport library numpy as np
import tensorflow as tf

app = Flask(__name__)

preparation()

cap = cv2.VideoCapture(0)

@app.route('/chatbot', methods=["GET", "POST"])
def predict():
    text = request.get_json().get("message")
    response = botResponse(text)
    message = {"answer": response}
    return jsonify(message)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/blog')
def blog():
    return render_template('blog.html')

@app.route('/squat-classes')
def squat_classes():
    return render_template('squat.html')

@app.route('/pu-classes')
def pu_classes():
    return render_template('push_up.html')

@app.route('/su-classes')
def su_classes():
    return render_template('sit_up.html')


def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 


# def gen_frames():
#     # Curl counter variables
#     counter = 0 
#     stage = None
#     mp_drawing = mp.solutions.drawing_utils
#     mp_pose = mp.solutions.pose

#     ## Setup mediapipe instance
#     with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
#         while cap.isOpened():
#             ret, frame = cap.read()
            
#             # Recolor image to RGB
#             image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             image.flags.writeable = False
        
#             # Make detection
#             results = pose.process(image)
        
#             # Recolor back to BGR
#             image.flags.writeable = True
#             image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
#             # Extract landmarks
#             try:
#                 landmarks = results.pose_landmarks.landmark
                
#                 # Get coordinates
#                 hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
#                 knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
#                 ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                
#                 # Calculate angle
#                 angle = calculate_angle(hip, knee, ankle)
                
#                 # Visualize angle
#                 cv2.putText(image, str(angle), 
#                             tuple(np.multiply(knee, [640, 480]).astype(int)), 
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
#                                     )
                
#                 # Curl counter logic
#                 if angle > 170:
#                     stage = "   down"

#                 if angle < 90 and stage =='   down':
#                     stage="   up"
#                     counter +=1
#                     print(counter)

                
#                 # Render curl counter
#                 # Setup status box
#                 cv2.rectangle(image, (0,0), (400,73), (0,0,220), -1)
                
#                 # Rep data
#                 cv2.putText(image, 'REPETISI', (15,12), 
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
#                 cv2.putText(image, str(counter), 
#                             (10,60), 
#                             cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
                
#                 # Stage data
#                 cv2.putText(image, ' PERINTAH', (65,12), 
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
#                 cv2.putText(image, stage, 
#                             (60,60), 
#                             cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
                
                
#                 # Render detections
#                 mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
#                                           mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=2), 
#                                           mp_drawing.DrawingSpec(color=(0,0,220), thickness=2, circle_radius=2) 
#                                          )   
            
#             except Exception as e:
#                 continue

#             finally:
                                    
#                 # cv2.imshow('Mediapipe Feed', image)
#                 if cv2.waitKey(10) & 0xFF == ord('q'):
#                     break

#                 ret, buffer = cv2.imencode('.jpg', image)
#                 frame = buffer.tobytes()
#                 yield (b'--frame\r\n'
#                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


# def gen_frames_2():
#     # Curl counter variables
#     counter = 0 
#     stage = None
#     mp_drawing = mp.solutions.drawing_utils
#     mp_pose = mp.solutions.pose

#     ## Setup mediapipe instance
#     with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
#         while cap.isOpened():
#             ret, frame = cap.read()
            
#             # Recolor image to RGB
#             image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             image.flags.writeable = False
        
#             # Make detection
#             results = pose.process(image)
        
#             # Recolor back to BGR
#             image.flags.writeable = True
#             image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
#             # Extract landmarks
#             try:
#                 landmarks = results.pose_landmarks.landmark
                
#                 # Get coordinates
#                 shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
#                 elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
#                 wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                    
#                 # Calculate angle
#                 angle = calculate_angle(shoulder, elbow, wrist)
                
#                 # Visualize angle
#                 cv2.putText(image, str(angle), 
#                             tuple(np.multiply(elbow, [640, 480]).astype(int)), 
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
#                                     )
                
#                 # Push up counter logic
#                 if angle > 100:
#                     stage = "   up"

#                 if angle < 70 and stage =='   up':
#                     stage="   down"
#                     counter +=1
#                     print(counter)

                
#                 # Render curl counter
#                 # Setup status box
#                 cv2.rectangle(image, (0,0), (400,73), (0,0,220), -1)
                
#                 # Rep data
#                 cv2.putText(image, 'REPETISI', (15,12), 
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
#                 cv2.putText(image, str(counter), 
#                             (10,60), 
#                             cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
                
#                 # Stage data
#                 cv2.putText(image, '                 PERINTAH', (65,12), 
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
#                 cv2.putText(image, stage, 
#                             (60,60), 
#                             cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
                
                
#                 # Render detections
#                 mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
#                                           mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=2), 
#                                           mp_drawing.DrawingSpec(color=(0,0,220), thickness=2, circle_radius=2) 
#                                          )   
            
#             except Exception as e:
#                     continue

#             finally:
                                     
#                 # cv2.imshow('Mediapipe Feed', image)

#                 if cv2.waitKey(10) & 0xFF == ord('q'):
#                     break
                
#                 ret, buffer = cv2.imencode('.jpg', image)
#                 frame = buffer.tobytes()
#                 yield (b'--frame\r\n'
#                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


def gen_frames_3():
            # Curl counter variables
    counter = 0 
    stage = None
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
        
            # Make detection
            results = pose.process(image)
        
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                        
                # Calculate angle
                angle = calculate_angle(shoulder, hip, knee)
                
                # Visualize angle
                cv2.putText(image, str(angle), 
                            tuple(np.multiply(hip, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                
                # Push up counter logic
                if angle > 100:
                    stage = "   up"

                if angle < 70 and stage =='   up':
                    stage="   down"
                    counter +=1
                    print(counter)

                
                # Render curl counter
                # Setup status box
                cv2.rectangle(image, (0,0), (400,73), (0,0,220), -1)
                
                # Rep data
                cv2.putText(image, 'REPETISI', (15,12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(image, str(counter), 
                            (10,60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
                
                # Stage data
                cv2.putText(image, '                 PERINTAH', (65,12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(image, stage, 
                            (60,60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
                
                
                # Render detections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=2), 
                                          mp_drawing.DrawingSpec(color=(0,0,220), thickness=2, circle_radius=2) 
                                         )   
            
            except Exception as e:
                    continue

            finally:
            
                # cv2.imshow('Mediapipe Feed', image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                
                ret, buffer = cv2.imencode('.jpg', image)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


@app.route('/squat')
def squat():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(process_video_squat(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/pu_classes')
def push_up():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(process_video_deadlift(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/su_classes')
def sit_up():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames_3(), mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__ == '__main__':
    app.run(debug=True)


