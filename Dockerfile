FROM node:18-slim AS frontend-builder

WORKDIR /frontend
COPY frontend/package.json .
RUN npm install --legacy-peer-deps
COPY frontend/ .
RUN npm run build

FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
COPY --from=frontend-builder /frontend/build ./frontend/build

EXPOSE 7860

CMD ["python", "app.py"]
