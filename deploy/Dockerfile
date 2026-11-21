FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY kuon ./kuon
COPY kuon_agent ./kuon_agent
RUN pip install --no-cache-dir -e .

ENV KUON_ENGINE=mock
EXPOSE 8000
CMD ["python", "-m", "kuon_agent.server", "--bind", "0.0.0.0", "--port", "8000"]
