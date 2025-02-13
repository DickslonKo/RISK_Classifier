FROM python:3.9-bullseye
#Expose port 8080
EXPOSE 8080

#Optional - install git to fetch packages directly from github
RUN apt-get update && apt-get install -y git

#Copy Requirements.txt file into app directory
COPY requirements.txt app/requirements.txt

#install all requirements in requirements.txt
RUN pip install -r app/requirements.txt

#Copy all files in current directory into app directory
COPY . /

#Change Working Directory to app directory
WORKDIR /

#Run the application on port 8080
ENTRYPOINT ["streamlit", "run", "risk_classifier.py", "--server.port=8080", "--server.address=0.0.0.0"]
