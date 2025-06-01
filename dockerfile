FROM python:3.8


COPY requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /avisense_backend
COPY . .

EXPOSE 8000

# CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
