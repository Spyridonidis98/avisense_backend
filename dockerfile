FROM python:3

RUN pip install fastapi 
RUN pip install uvicorn

WORKDIR /avisense_backend
COPY . .

EXPOSE 8000

# CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
