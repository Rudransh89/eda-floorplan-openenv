FROM python:3.10-slim
WORKDIR /app
COPY . /app

# Install standard dependencies and OpenEnv core
RUN pip install --no-cache-dir pydantic openai httpx fastapi uvicorn
RUN pip install --no-cache-dir openenv-core

# Ensure the space listens on the correct port if running as a web service
EXPOSE 7860
CMD ["python", "inference.py"]