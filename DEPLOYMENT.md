# 🚀 Deployment Guide

This guide provides step-by-step instructions for deploying the Recipe Site Traffic Prediction Streamlit app on various platforms.

## Prerequisites

- Python 3.8+
- Git installed
- GitHub account
- Code pushed to GitHub repository

---

## 🎈 Streamlit Cloud (Recommended)

**Best for**: Free hosting, easiest setup, automatic updates

### Steps:
1. **Push code to GitHub** (if not already done)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **"New app"**
4. Select your repository: `iNSRawat/recipe-site-traffic-prediction`
5. Set **Main file path**: `app.py`
6. Click **"Deploy"**

### Features:
- ✅ Free tier available
- ✅ Automatic redeployment on push
- ✅ Custom subdomain
- ✅ No configuration needed

---

## 🟣 Heroku

**Best for**: Production apps, custom domains

### Files Required:

**Procfile** (create in root):
```
web: sh setup.sh && streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

**setup.sh** (create in root):
```bash
mkdir -p ~/.streamlit/
echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
```

### Deployment Steps:
```bash
# Install Heroku CLI and login
heroku login

# Create app
heroku create recipe-traffic-predictor

# Deploy
git push heroku main

# Open app
heroku open
```

### Features:
- ✅ Free dynos (with limitations)
- ✅ Custom domains
- ✅ Add-ons available
- ⚠️ Sleeps after 30 min inactivity (free tier)

---

## 🚂 Railway

**Best for**: Simple deployment, generous free tier

### Steps:
1. Go to [railway.app](https://railway.app)
2. Click **"New Project"** → **"Deploy from GitHub repo"**
3. Select `recipe-site-traffic-prediction`
4. Add environment variable:
   - `PORT` = `8501`
5. Railway auto-detects Python and deploys

### Start Command (if needed):
```bash
streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

### Features:
- ✅ $5 free credits/month
- ✅ Auto-deploy on push
- ✅ Easy environment variables
- ✅ Built-in monitoring

---

## 🔷 Render

**Best for**: Static sites + web services

### Steps:
1. Go to [render.com](https://render.com)
2. Click **"New"** → **"Web Service"**
3. Connect your GitHub repository
4. Configure:
   - **Name**: `recipe-traffic-predictor`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
5. Click **"Create Web Service"**

### Features:
- ✅ Free tier available
- ✅ Auto-deploy from GitHub
- ✅ SSL certificates included
- ⚠️ Spins down after inactivity (free tier)

---

## 🐳 Docker (Self-hosted)

**Best for**: Custom infrastructure, Kubernetes

### Dockerfile:
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Build and Run:
```bash
# Build image
docker build -t recipe-traffic-app .

# Run container
docker run -p 8501:8501 recipe-traffic-app
```

---

## ☁️ AWS EC2

**Best for**: Full control, scalability

### Steps:
1. Launch EC2 instance (Ubuntu 22.04)
2. SSH into instance
3. Install dependencies:
```bash
sudo apt update
sudo apt install python3-pip
pip3 install -r requirements.txt
```
4. Run with nohup:
```bash
nohup streamlit run app.py --server.port=8501 &
```
5. Configure Security Group to allow port 8501

---

## 📊 Platform Comparison

| Platform | Free Tier | Auto-Deploy | Custom Domain | Setup Difficulty |
|----------|-----------|-------------|---------------|------------------|
| Streamlit Cloud | ✅ | ✅ | ✅ | ⭐ Easy |
| Railway | ✅ ($5/mo) | ✅ | ✅ | ⭐ Easy |
| Render | ✅ | ✅ | ✅ | ⭐⭐ Medium |
| Heroku | ✅ (limited) | ✅ | ✅ | ⭐⭐ Medium |
| Docker | N/A | ❌ | N/A | ⭐⭐⭐ Advanced |
| AWS EC2 | ✅ (1 year) | ❌ | ✅ | ⭐⭐⭐ Advanced |

---

## 🔧 Troubleshooting

### Common Issues:

1. **Port binding error**: Ensure you're using `--server.port=$PORT`
2. **Module not found**: Check `requirements.txt` has all dependencies
3. **File not found**: Verify `workspace/recipe_site_traffic_2212.csv` path
4. **Memory issues**: Upgrade to a paid tier or optimize data loading

### Need Help?
- [Streamlit Docs](https://docs.streamlit.io/streamlit-community-cloud)
- [GitHub Issues](https://github.com/iNSRawat/recipe-site-traffic-prediction/issues)

---

**Author**: Nagendra Singh Rawat  
**GitHub**: [@iNSRawat](https://github.com/iNSRawat)
