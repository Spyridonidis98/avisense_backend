FROM python:3

RUN pip install fastapi 
RUN pip install uvicorn
RUN pip install python-multipart

WORKDIR /fast_api_tut
COPY . .

EXPOSE 8000

# CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
