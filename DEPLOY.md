# 固態火箭推進概念模擬平台 — 部署指南

## 所有部署方式總覽

| 方式 | 難度 | 費用 | 適合誰 |
|------|------|------|--------|
| ① Streamlit Community Cloud | ⭐ 最簡單 | 免費 | 最推薦，零設定 |
| ② Railway | ⭐⭐ | 免費 $5/月額度 | 想要自訂域名 |
| ③ Render | ⭐⭐ | 免費方案可用 | 想要免費長期跑 |
| ④ Fly.io | ⭐⭐ | 免費方案可用 | 想要亞洲節點低延遲 |
| ⑤ Docker 自架 | ⭐⭐⭐ | 看主機 | 有自己的伺服器 |
| ⑥ Hugging Face Spaces | ⭐ | 免費 | ML 社群展示 |
| ⑦ Google Cloud Run | ⭐⭐⭐ | 按用量計費 | 企業級需求 |
| ⑧ 本機 (localhost) | ⭐ | 免費 | 自己用/開發 |

---

## ① Streamlit Community Cloud（最推薦）

**最簡單，完全免費，專為 Streamlit 設計。**

### 步驟

1. 把專案推送到 GitHub：
   ```bash
   cd solid-rocket-sim
   git init
   git add .
   git commit -m "initial commit"
   git remote add origin https://github.com/<你的帳號>/solid-rocket-sim.git
   git push -u origin main
   ```

2. 前往 https://share.streamlit.io

3. 登入 GitHub → 選擇 repo → 主檔案填 `app.py`

4. 點 Deploy → 完成！自動取得網址如：
   `https://<你的帳號>-solid-rocket-sim.streamlit.app`

### 優點
- 零設定，不需要 Docker
- 自動偵測 requirements.txt
- 自動 CI/CD：push 到 GitHub 就自動更新
- 完全免費

### 限制
- 資源有限（1GB RAM）
- 閒置 7 天後休眠（有人訪問自動喚醒）
- 不能自訂域名

---

## ② Railway

### 步驟
```bash
# 安裝 CLI
npm install -g @railway/cli

# 登入
railway login

# 初始化並部署
cd solid-rocket-sim
railway init
railway up
```

Railway 會自動偵測 Dockerfile 並建構部署。

### 優點
- 支援自訂域名
- 自動 HTTPS
- 每月 $5 免費額度

---

## ③ Render

### 步驟

1. 推送到 GitHub

2. 前往 https://render.com → New → Web Service

3. 連接 GitHub repo

4. Render 會自動偵測 `render.yaml` 或 `Dockerfile`

5. 部署完成

### 優點
- 免費方案可用
- 自動 HTTPS
- 自動部署

### 注意
- 免費方案閒置 15 分鐘後休眠

---

## ④ Fly.io

### 步驟
```bash
# 安裝 CLI
curl -L https://fly.io/install.sh | sh

# 登入
fly auth login

# 部署（會讀取 fly.toml）
cd solid-rocket-sim
fly launch    # 首次
fly deploy    # 後續更新
```

### 優點
- 亞洲有節點（東京 nrt），延遲低
- 免費方案：3 個 shared-cpu 小機器
- 自動 HTTPS + 自訂域名
- 可設定自動休眠省資源

---

## ⑤ Docker 自架

### 在任何有 Docker 的機器上：

```bash
# 建構
docker build -t solid-rocket-sim .

# 執行
docker run -d -p 8501:8501 --name rocket-sim solid-rocket-sim

# 或用 docker compose
docker compose up -d
```

瀏覽器打開 `http://<你的IP>:8501`

### 搭配 Nginx 反向代理 + HTTPS：
```nginx
server {
    listen 443 ssl;
    server_name rocket.example.com;

    ssl_certificate     /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";  # WebSocket 必要！
        proxy_set_header Host $host;
    }

    location /_stcore/stream {
        proxy_pass http://localhost:8501/_stcore/stream;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

⚠ **Nginx 必須設定 WebSocket 轉發**，否則 Streamlit 無法運作。

---

## ⑥ Hugging Face Spaces

### 步驟

1. 前往 https://huggingface.co/spaces → Create Space

2. 選 SDK = **Streamlit**

3. 上傳所有檔案，或連接 GitHub repo

4. HF 自動建構並部署

### 優點
- 免費
- ML 社群曝光度高
- 支援 GPU（如果需要）

---

## ⑦ Google Cloud Run

### 步驟
```bash
# 建構並推送 Docker image
gcloud builds submit --tag gcr.io/<PROJECT_ID>/solid-rocket-sim

# 部署
gcloud run deploy solid-rocket-sim \
    --image gcr.io/<PROJECT_ID>/solid-rocket-sim \
    --port 8501 \
    --allow-unauthenticated \
    --region asia-east1 \
    --memory 512Mi \
    --session-affinity
```

⚠ **必須加 `--session-affinity`**，確保 WebSocket 黏著到同一個實例。

---

## ⑧ 本機執行

```bash
cd solid-rocket-sim
pip install -r requirements.txt
streamlit run app.py
```

瀏覽器自動開啟 `http://localhost:8501`

---

## ❌ 不能用的平台

| 平台 | 為什麼不行 |
|------|-----------|
| **Vercel** | Serverless 無狀態，不支援 WebSocket 長連線 |
| **Netlify** | 同上，靜態網站 + Serverless Function |
| **GitHub Pages** | 純靜態 HTML，無法執行 Python |
| **Cloudflare Pages** | 純靜態/Workers，不支援長駐 Python 進程 |

### 根本原因

這些平台的共同問題：

```
瀏覽器 ←──WebSocket 持久連線──→ Streamlit Python 進程
         ↑                        ↑
     Vercel 不支援這個        Vercel 不支援這個
     (只支援 HTTP 請求)       (只支援短命函數)
```

Streamlit 的運作模式：
1. 瀏覽器連上後，建立 WebSocket 長連線
2. 使用者拖滑桿 → WebSocket 發送事件 → Python 重新執行腳本 → 回傳新圖表
3. 整個過程需要 Python 進程持續存活

Vercel/Netlify 的運作模式：
1. 收到 HTTP 請求 → 啟動函數 → 回傳結果 → 函數死亡
2. 下一個請求 → 啟動全新函數（無狀態）
3. 不支援 WebSocket，不支援長駐進程
