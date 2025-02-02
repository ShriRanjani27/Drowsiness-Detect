from flask import Flask, redirect, render_template, Response, request, url_for
from threading import Thread
from flask_login import LoginManager, UserMixin, login_user, logout_user
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import cv2
import imutils
import dlib
from imutils import face_utils
from scipy.spatial import distance
import pygame
import time

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///db.sqlite"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy()
db.init_app(app)

login_manager = LoginManager()
login_manager.init_app(app)

class Users(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(250), unique=True, nullable=False)
    password = db.Column(db.String(250), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return Users.query.get(int(user_id))
    
@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        user = Users.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return render_template('home.html')  # Redirect to the home page after login
    return render_template('login.html')


@app.route('/signup', methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        hashed_password = generate_password_hash(password)
        
        existing_user = Users.query.filter_by(username=username).first()
        if existing_user:
            return "Username already exists. Please choose a different username."
        
        new_user = Users(username=username, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        
        return redirect(url_for("login"))
    
    return render_template("signup.html")

@app.route('/logout', methods=["GET", "POST"])
def logout():
    if request.method == "POST":
        logout_user()
        return redirect(url_for('login'))
    # For GET requests, redirect to the login page
    return redirect(url_for('login'))


# Define the route for the index page
@app.route('/index')
def index():
    return render_template('index.html')

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

thresh = 0.25
frame_check = 20
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

@app.route('/')
def home():
    return render_template('signup.html')

def detect_drowsiness():
    cap = cv2.VideoCapture(0)
    flag_eyes_closed = 0
    flag_face_missing = 0
    alarm_playing = False
    alarm_played = False
    last_detection_time = time.time()
    last_face_detection_time = time.time()
    alarm_start_time = 0
    pygame.mixer.init()
    pygame.mixer.music.load("static/alarm.wav")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame from camera.")
            break

        frame = imutils.resize(frame, width=800)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        subjects = detect(gray, 0)

        if len(subjects) == 0:
            if time.time() - last_face_detection_time >= 3:
                flag_face_missing += 1
                if flag_face_missing >= frame_check:
                    alarm_playing = True
                    alarm_start_time = time.time()
                    cv2.putText(frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            flag_face_missing = 0
            last_face_detection_time = time.time()

        for subject in subjects:
            shape = predict(gray, subject)
            shape = face_utils.shape_to_np(shape)
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            if ear < thresh:
                flag_eyes_closed += 1
                if flag_eyes_closed >= frame_check and time.time() - last_detection_time >= 3:
                    alarm_playing = True
                    alarm_start_time = time.time()
                    cv2.putText(frame, "ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                flag_eyes_closed = 0

        if alarm_playing and not alarm_played:
            pygame.mixer.music.play()
            alarm_played = True

        if time.time() - alarm_start_time >= 3:
            pygame.mixer.music.stop()
            alarm_playing = False
            alarm_played = False

        _, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(detect_drowsiness(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
    with app.app_context():
        db.create_all()  # This line initializes the database
