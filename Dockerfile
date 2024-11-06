FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

EXPOSE 8000

CMD ["chainlit","run","--host","0.0.0.0", "ChatIITK_chainlit.py"]  

