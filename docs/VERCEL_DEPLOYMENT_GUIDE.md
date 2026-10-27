# Extremis DataSentry: Production Deployment Guide (Vercel & Cloud)

This guide details how to deploy **Extremis DataSentry** to **Vercel** and other production cloud environments.

---

## 1. Deploying to Vercel

Vercel is fundamentally a serverless edge platform. Because Streamlit is built on a persistent Python process with WebSockets, there are two production patterns to deploy Streamlit projects to Vercel:

### Method A: Client-Side Serverless via Stlite (Recommended for Vercel)
**Stlite** runs Streamlit in WebAssembly via Pyodide directly inside the user's browser, allowing Vercel to serve it as a 100% static, serverless web application with **zero backend server costs** and **no execution timeout limits**.

1. **Verify Vercel Configuration:**
   The repository already includes [`vercel.json`](file:///c:/Users/adity/Downloads/Extremis%20DataSentry/vercel.json) and [`public/index.html`](file:///c:/Users/adity/Downloads/Extremis%20DataSentry/public/index.html).

2. **Deploy via Vercel CLI:**
   ```bash
   # Install Vercel CLI (if not installed)
   npm install -g vercel

   # Deploy to preview
   vercel

   # Deploy to production
   vercel --prod
   ```

3. **Deploy via GitHub / Vercel Web Console:**
   - Push this repository to your GitHub account.
   - Go to [vercel.com](https://vercel.com) and click **"Add New Project"**.
   - Select your `Extremis-DataSentry` repository.
   - Set **Framework Preset** to **Other**.
   - Set **Output Directory** to `public` (or leave as default with `vercel.json`).
   - Click **Deploy**.

---

### Method B: Streamlit Community Cloud (Official Free 1-Click Python Hosting)
For compute-intensive ML workloads (such as running multi-fold Out-Of-Fold cross-validation on massive CSV files with 100,000+ rows), **Streamlit Community Cloud** provides dedicated persistent Python backends for free:

1. Push your code to GitHub.
2. Visit [share.streamlit.io](https://share.streamlit.io).
3. Connect your GitHub repository.
4. Set:
   - **Main file path:** `app.py`
   - **Python version:** `3.12`
5. Click **Deploy!** Your app will be live at `https://<your-app-name>.streamlit.app`.

---

### Method C: Docker Container Deployment (AWS, GCP Cloud Run, Render, Railway)
To deploy as a containerized enterprise SaaS service:

Create a `Dockerfile`:
```dockerfile
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run locally with Docker:
```bash
docker build -t extremis-datasentry .
docker run -p 8501:8501 extremis-datasentry
```
