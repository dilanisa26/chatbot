import json
import random
import nltk
import string
import numpy as np
import pickle
import tensorflow as tf
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences

global responses, lemmatizer, tokenizer, le, model, input_shape
input_shape = 21

# import dataset answer
def load_response():
    global responses
    responses = {}
    with open('dataset/Chatbot.json') as content:
        data = json.load(content)
    for intent in data['intents']:
        responses[intent['tag']]=intent['responses']

# import model dan download nltk file
def preparation():
    load_response()
    global lemmatizer, tokenizer, le, model
    tokenizer = pickle.load(open('model/tokenizer.pkl', 'rb'))
    le = pickle.load(open('model/label_encoder.pkl', 'rb'))
    model = keras.models.load_model('model/model.h5', compile=False)
    lemmatizer = StemmerFactory().create_stemmer()
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

# hapus tanda baca
def remove_punctuation(text):
    texts_p = []
    text = [letters.lower() for letters in text if letters not in string.punctuation]
    text = ''.join(text)
    texts_p.append(text)
    return texts_p

# mengubah text menjadi vector
def vectorization(texts_p):
    vector = tokenizer.texts_to_sequences(texts_p)
    vector = np.array(vector).reshape(-1)
    vector = pad_sequences([vector], input_shape)
    return vector

# klasifikasi pertanyaan user
def predict(vector):
    output = model.predict(vector)
    output = output.argmax()
    response_tag = le.inverse_transform([output])[0]
    return response_tag

# menghasilkan jawaban berdasarkan pertanyaan user
def botResponse(text):
    texts_p = remove_punctuation(text)
    vector = vectorization(texts_p)
    response_tag = predict(vector)
    answer = random.choice(responses[response_tag])
    return answer

def process_video_deadlift():
    # Load the random forest model
    with open('model/random_forest_Deadlift4_good.pkl', 'rb') as f:
        model = pickle.load(f)

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(0)

# Setup mediapipe instance
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

            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                     )
            # Export Coordinate
            try:
                # Take pose landmarks
                pose_landmark = results.pose_landmarks.landmark
                pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose_landmark]).flatten())

                X = pd.DataFrame([pose_row])
                form_class = model.predict(X)[0]
                form_prob = model.predict_proba(X)[0]
                print(form_class, form_prob)

                # Status box
                box_color = (245, 117, 16)  # Default box color
                if form_class == "Rounded" or form_class == "Hip Not Locked":
                    box_color = (0, 0, 255)  # Red box color

                cv2.rectangle(image, (0, 0), (350, 50), box_color, -1)

                # Display Class
                cv2.putText(image, 'FORM', (95, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, form_class, (90, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                # Display Prob
                cv2.putText(image, 'PROB', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(round(form_prob[np.argmax(form_prob)], 2)), (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            except:
                pass

            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        
def process_video_squat():
    with open('random_forest_squat','rb') as f:
        model = pickle.load(f)

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(0)

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

            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )
            # Export Coordinate
            try:
                # take pose landmark
                pose_landmark = results.pose_landmarks.landmark
                pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose_landmark]).flatten())

                X = pd.DataFrame([pose_row])
                form_class = model.predict(X)[0]
                form_prob = model.predict_proba(X)[0]
                print(form_class, form_prob)

                # status box
                box_color = (245, 117, 16)  # Default box color
                if form_class == "not enough depth":
                    box_color = (0, 0, 255)  # Red box color

                cv2.rectangle(image, (0, 0), (400, 50), box_color, -1)


                # Display Class
                cv2.putText(image, 'FORM'
                        , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(image, form_class
                        , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)


                # Display Prob
                cv2.putText(image, 'PROB'
                        , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(image, str(round(form_prob[np.argmax(form_prob)],2))
                        , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            except:

                pass

            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()