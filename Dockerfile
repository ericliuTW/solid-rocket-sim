# ── 固態火箭推進概念模擬平台 Docker 部署 ──────────────────────────────
# 建構: docker build -t solid-rocket-sim .
# 執行: docker run -p 8501:8501 solid-rocket-sim
# ─────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# 安裝系統字型（CJK 中文支援）
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        fonts-noto-cjk \
        && \
    rm -rf /var/lib/apt/lists/* && \
    fc-cache -fv

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 清除 matplotlib 字型快取，讓它重新偵測 Noto CJK
RUN python -c "import matplotlib; import shutil; shutil.rmtree(matplotlib.get_cachedir(), ignore_errors=True)"

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0", \
            "--server.headless=true"]
