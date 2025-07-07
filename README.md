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