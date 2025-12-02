# sps-genai--project

## Running the Backend with Docker
The instructor must be able to clone your repository, build the Docker image, and run the API successfully.

Below are the exact commands needed.

1. Build the Docker image

From the repository root:
```bash
docker build -t med-assistant .
```
2. Run the container

Provide your own Hugging Face API key:
```bash
docker run -e HF_API_KEY=YOUR_HF_API_KEY -p 8000:8000 med-assistant
```

This starts the FastAPI server at:

http://localhost:8000

3. Health Check Endpoint

The server includes a simple route for Docker verification:
```bash
curl http://localhost:8000/health
```

Expected output:

{"status": "ok"}


This confirms the server is alive and running inside Docker.

4. Test the Main API Endpoint

To test the full LLM+Safety pipeline:
```bash
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{
        "question": "I have a mild headache for three days, no fever.",
        "task_type": "general_qa",
        "language": "en",
        "temperature": 0.2,
        "max_tokens": 256
      }'
```

Example JSON response:

{
  "answer": "Here is a general explanation ...",
  "safety": {
    "level": "safe",
    "message": "No emergency features detected."
  },
  "severity": {
    "severity": "low",
    "recommended_action": "self_care",
    "time_window": "1-2 weeks",
    "risk_notes": "Monitor symptoms."
  },
  "used_prompt": "..."
}


This confirms that:

Docker works

FastAPI works

/api/ask works

LLM inference works

Safety layer works


HF_API_KEY	  Your Hugging Face Inference API key

## Running the Frontend (Optional)

The frontend is included under /backend/app/static/.

To run it directly (no Docker needed):
```bash
python -m http.server
```

or simply open the index.html file in your browser.

Development Setup (Non-Docker)
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```
