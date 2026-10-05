# Local face CCTV prototype
![alt text](image.png)

This prototype uses NanoDet and ByteTrack to follow people, recognises people who
have explicitly been enrolled, records their presence locally, and generates an
HTML activity report. Unknown faces are not stored; unknown person tracks are
logged under temporary anonymous IDs such as `Unknown #4`.

Face embeddings are biometric data. Only enrol people with their permission,
protect the `cctv_data` directory, and delete data when it is no longer needed.

https://github.com/FoundationVision/ByteTrack
https://github.com/opencv/opencv_zoo/tree/main/models/object_detection_nanodet


## Setup

TDLR
py -3.13 -m venv .cctv-venv
.\.cctv-venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
.\download_face_models.ps1

python face_cctv.py enroll --name "Your Name"
python face_cctv.py run
python face_cctv.py report

If youre using an external camera plugged into a computer use

python face_cctv.py run --camera 1

python face_cctv.py enroll --name "Your Name" --camera 1

The existing `.venv` is currently broken because its original Microsoft Store
Python installation is inaccessible. Install or repair Python 3.11+ first, then
open PowerShell in this folder and run:

```powershell
py -3.13 -m venv .cctv-venv
.\.cctv-venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
.\download_face_models.ps1
```

## Use

Enrol one consenting person (capture from several head angles):

```powershell
python face_cctv.py enroll --name "Alice"
```

Start recognition and presence logging:

```powershell
python face_cctv.py run
```

Press `Q` or Escape to stop. Then generate the report:

```powershell
python face_cctv.py report
```

Open `cctv_data/report.html` in a browser. Other useful commands are:

```powershell
python face_cctv.py list
python face_cctv.py remove --name "Alice"
```

## How the live pipeline fits together

`NanoDet person detection -> ByteTrack ID -> optional face identity -> CSV/HTML report`

ByteTrack IDs last only for the current run. A face name is attached to a track
after three consistent recognition matches, which reduces one-frame mistakes.

## How it works

Webcam produces lots frames when the python script is started (python face.cctv.py run)

Opencv access the videocapture (capture = cv.videocapture(0)), note wecamera is usally 0 so change here if you are on mobile or any other camera

a video is essentially a rapid sequence of images and each frame is a nympt array here cotaining colour values of every pixel
A typical shape :
1080 rows × 1920 columns × 3 colour channels
This is BGR rather then RGB

Next YuNet finds the face. The ONNX model recieves the complete framem which looks at the face location points. 
For every face YuNEt typically returns 15 values: 
x, y, width, height
right-eye position
left-eye position
nose position
right mouth-corner position
left mouth-corner position
confidence

Next OpenCV aligns the face. People dont have to be looking directely into the camera for recognise to work
This call aligned = self.recognizer.alignCrop(frame, face)
produces a consistently positioned face image

SFace next produces and embedding using the sface ONNX mode with feature = self.recognizer.feature(aligned)
This returns a numeric facial embedding or a vector containing many floating point numbers.
These are patterns SFace uses for comparing faces. Then the embedding is normalised 
feature = feature / np.linalg.norm(feature)
Which changes the vectors overall length to one making comparisions more conaistent across images and lightings

enrolment on local database to recognise someone.
python face_cctv.py enroll --name "Alice"
caprues ten samples by default, aligns each face, converts to an embedding and stores embeddings under name. this is stored in cctv_data/faces.json in a json structure.

During live recognition the application calculates an embedding for the current webcam face and cmapers it with every enrolled sampple
score = np.dot(current_embedding, enrolled_embedding)
then the results meaure how closely the two vectors point in the same direction. higher score = more similar, lower = less. The match threshold is 0.363

The threshold comes from openCVs SFace comparison guidance but it is not a gaurantee. The comparison happens during the indetify() in face_cctv.py

Presence sessions are recorded. for example if alice has been recognised on camera the system will store First seen
Last seen
Best similarity score and as long as Alice continues to appear, the program updates last_seen
If Alice is not recognised for five seconds, the session is considered finished. The program writes it to:cctv_data/events.csv
