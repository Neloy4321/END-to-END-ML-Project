# Sleep Health Prediction System

A complete End-to-End Machine Learning project for predicting sleep-related health conditions using user lifestyle, sleep habits, medical conditions, and behavioral patterns.

This project was developed using:
- FastAPI
- Machine Learning
- MongoDB
- Docker
- AWS EC2
- AWS ECR
- GitHub Actions CI/CD

The system predicts possible sleep-related conditions such as:
- No Condition
- Sleep Respiratory Disorders
- Health Issues
- Mental Health Issues
- Others

---

# Project Architecture

```bash
Sleep Health Prediction System

├── FastAPI Backend
├── HTML Frontend
├── ML Training Pipeline
├── Prediction Pipeline
├── MongoDB Integration
├── Docker Containerization
├── AWS ECR Deployment
├── EC2 Hosting
└── GitHub Actions CI/CD
```

---

# Features

- End-to-End ML Pipeline
- Dynamic Sleep Health Prediction
- FastAPI Web Application
- HTML User Interface
- MongoDB Database Integration
- Dockerized Application
- AWS EC2 Deployment
- AWS ECR Docker Registry
- GitHub Actions CI/CD Automation
- Real-Time Prediction System
- Dataset-Aligned Feature Engineering

---

# Tech Stack

| Technology | Usage |
|---|---|
| Python | Backend & ML |
| FastAPI | API Framework |
| Scikit-Learn | Machine Learning |
| Pandas | Data Processing |
| MongoDB | Database |
| Docker | Containerization |
| AWS EC2 | Deployment Server |
| AWS ECR | Docker Registry |
| GitHub Actions | CI/CD |
| HTML/CSS/Bootstrap | Frontend |

---

# Project Workflow

```bash
Data Collection
        ↓
Data Validation
        ↓
Data Transformation
        ↓
Feature Engineering
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Prediction Pipeline
        ↓
FastAPI Deployment
        ↓
Docker Container
        ↓
AWS EC2 Deployment
```

---

# Project Structure

```bash
sleep_project/

├── artifact/
├── notebook/
├── static/
├── templates/

├── sleep_project/
│   ├── components/
│   ├── pipeline/
│   ├── entity/
│   ├── configuration/
│   ├── constants/
│   ├── logger/
│   └── exception/

├── .github/workflows/
├── app.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
└── README.md
```

---

# Dataset Features

The model uses multiple sleep-related attributes including:

- Age
- Weight
- Height
- Gender
- Occupation
- Bed Time
- Wake Up Time
- Sleep Duration
- Difficulty Falling Asleep
- Breathing Problems
- Restless Legs
- Concentration Problems
- Sleep Environment Comfort
- Medical Conditions
- Sleep Reasons
- Coping Strategies

---

# Machine Learning Pipeline

The project includes:

- Data Ingestion
- Data Validation
- Data Transformation
- Feature Engineering
- Label Encoding
- Multi-label Processing
- Model Training
- Prediction Pipeline

---

# Model Prediction Classes

| Class | Meaning |
|---|---|
| 0 | No Condition |
| 1 | Sleep Respiratory Disorders |
| 2 | Health Issues |
| 3 | Mental Health Issues |
| 4 | Others |

---

# How To Run Locally

## Create Environment

```bash
conda create -n sleep python=3.10 -y
```

## Activate Environment

```bash
conda activate sleep
```

## Install Requirements

```bash
pip install -r requirements.txt
```

## Run Application

```bash
uvicorn app:app --reload
```

## Open Browser

```bash
http://127.0.0.1:8000
```

---

# Docker Setup

## Build Docker Image

```bash
docker build -t sleep-health-app .
```

## Run Docker Container

```bash
docker run -p 8000:8000 sleep-health-app
```

---

# AWS Deployment Architecture

```bash
GitHub Repository
        ↓
GitHub Actions
        ↓
Docker Build
        ↓
AWS ECR
        ↓
EC2 Pulls Docker Image
        ↓
FastAPI Application Runs
```

---

# AWS Services Used

| Service | Purpose |
|---|---|
| EC2 | Hosting |
| ECR | Docker Registry |
| IAM | Deployment Access |
| GitHub Actions | CI/CD Automation |

---

# AWS CI/CD Deployment Steps

## 1. Create IAM User

Required Policies:
- AmazonEC2FullAccess
- AmazonEC2ContainerRegistryFullAccess

---

## 2. Create AWS ECR Repository

Save the ECR URI.

---

## 3. Launch EC2 Instance

Recommended:
- Ubuntu
- t2.medium or higher

---

## 4. Install Docker In EC2

```bash
sudo apt-get update -y

curl -fsSL https://get.docker.com -o get-docker.sh

sudo sh get-docker.sh

sudo usermod -aG docker ubuntu

newgrp docker
```

---

## 5. Configure Self Hosted Runner

GitHub Repository:

```bash
Settings
→ Actions
→ Runners
→ New Self Hosted Runner
```

Run the provided commands inside EC2.

---

## 6. Configure GitHub Secrets

Add these secrets:

```bash
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_DEFAULT_REGION
ECR_REPO
MONGODB_URL
```

---

# Git Commands

## Add Files

```bash
git add .
```

## Commit Changes

```bash
git commit -m "Updated project"
```

## Push Changes

```bash
git push origin main
```

---

# Environment Variables

## Linux/Mac

```bash
export MONGODB_URL="your_mongodb_url"

export AWS_ACCESS_KEY_ID="your_access_key"

export AWS_SECRET_ACCESS_KEY="your_secret_key"
```

---

# Application Access

## Local

```bash
http://127.0.0.1:8000
```

## EC2 Deployment

```bash
http://<EC2_PUBLIC_IP>:8000
```

---

# Key Achievements

- Successfully built a complete ML pipeline
- Implemented dynamic prediction system
- Integrated FastAPI frontend/backend
- Dockerized the application
- Deployed using AWS EC2 & ECR
- Automated deployment using GitHub Actions
- Integrated MongoDB support
- Created real-time prediction workflow

---

# Future Improvements

- Better Feature Engineering
- SHAP Explainability
- User Authentication
- Prediction Confidence Scores
- Real-Time Analytics Dashboard
- Advanced ML Models
- Improved Dataset Balancing
- Batch Prediction System

---

# Author

Neloy Pramanik Supto

Computer Science & Engineering  
AI/ML Research Enthusiast  
FastAPI • Machine Learning • Docker • AWS • CI/CD