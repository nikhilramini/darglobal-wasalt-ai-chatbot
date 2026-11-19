# ==========================================
# 1. BASE IMAGE
# ==========================================

FROM python:3.12-slim

# ==========================================
# 2. SET WORKING DIRECTORY
# ==========================================

WORKDIR /app

# ==========================================
# 3. PYTHON CONFIGURATION
# ==========================================

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ==========================================
# 4. INSTALL SYSTEM DEPENDENCIES
# ==========================================

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ==========================================
# 5. INSTALL PYTHON DEPENDENCIES
# ==========================================

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ==========================================
# 6. COPY APPLICATION FILES
# ==========================================

COPY app.py .
COPY rag_chatbot.py .
COPY data/ ./data/
COPY chroma_db/ ./chroma_db/

# ==========================================
# 7. EXPOSE STREAMLIT PORT
# ==========================================

EXPOSE 8501

# ==========================================
# 8. START STREAMLIT
# ==========================================

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
