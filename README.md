# Trust-net
AI-Powered decentralized identity verification and fraud prevention platform

To run the backend in terminal
cd trustnet
cd backend
    mvn clean install
    mvn spring-boot:run
need to check to download maven...will update on the case of the databse

| Tool                      | Why It's Needed                              |
| ------------------------- | -------------------------------------------- |
| **Java JDK 21**           | Required for running Spring Boot             |
| **Maven (v3.9.x)**        | For building and running the backend project |
| **Git**                   | To pull the code from your GitHub repo       |
| (Optional) MySQL          | Only if that laptop will host the DB         |
| (Optional) Python + Flask | If running microservices locally             |

to run the python microservices 
 first install all the packages of each service with 
    pip install -r requirements.txt
 python app.py


 download pytho 3.10 and run it in a virtual environment to not clash with any other versions present to run the facial recognition part

 dlib installation (make necessary changes like the folder path)

 pip install "C:\Reshma\Trustnet\Dlib\dlib-19.22.99-cp310-cp310-win_amd64.whl"

python -c "import dlib; print(dlib.__version__)"

 py -3.10 -m venv trustnet-face :environment code


for face-match :

1) python -3.10 -m venv trustnet-face
    .\trustnet-face\Scripts\activate
2) pip install -r requirements.txt
3) python app.py
4) Testing :
    curl -X POST http://localhost:5000/face-match \
  -F "selfie=@path/to/selfie.jpg" \
  -F "document=@path/to/aadhaar.jpg" \
  -F "model=Facenet512"
5) pip install tf-keras




General Requirements:

Operating System: Windows 10 / 11

Java Version: Java 21

Build Tool: Apache Maven

Chosen Version: Maven 3.9.x (latest stable)

Download Link: https://maven.apache.org/download.cgi

Environment Setup:
Set MAVEN_HOME and add %MAVEN_HOME%\bin to PATH
Verify: mvn -v
Python Version: Initially Python 3.13 (faced compatibility issues), now using Python 3.10 for face recognition compatibility.

Python Setup:

1) Virtual Environment Setup:
    python -m venv trustnet-face
    .\trustnet-face\Scripts\activate
2) pip install -r requirements.txt
3) pip install deepface
4) pip install fire==0.5.0
5) if directly deepface installation fails...purge the cache and try :
  a) pip cache purge
  b) pip install --no-cache-dir fire

  (.venv) PS C:\Reshma\Trustnet\ai-microservices\Face-match>(for venev) & .\trustnet-face\Scripts\Activate.ps1
  (.venv) PS C:\Reshma\Trustnet\ai-microservices\Liveness-check>  & .\liveness-venv\Scripts\Activate.ps1 