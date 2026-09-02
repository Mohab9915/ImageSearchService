"""Gunicorn settings.

Two knobs matter here and both come from the model.

Workers: each one loads its own copy of SigLIP, roughly 800MB resident. Four
workers is the chatbot's setting and would be 3GB of duplicated weights for a
service that is almost entirely one forward pass. Two by default, overridable
where the box is bigger.

Timeout: an unindexed tenant's bootstrap runs in a background task, but a
search still has to fetch an image, load the model on the first call and run
inference. 120s leaves room for the cold one without leaving a wedged worker
alive all afternoon.
"""
import os

bind = f"0.0.0.0:{os.environ.get('PORT', '8080')}"
workers = int(os.environ.get("WEB_CONCURRENCY", "2"))
worker_class = "uvicorn.workers.UvicornWorker"

timeout = int(os.environ.get("GUNICORN_TIMEOUT", "120"))
graceful_timeout = 30
keepalive = 5

# Recycle workers periodically: torch's allocator holds on to freed blocks,
# so a long-lived worker's resident size only goes one way.
max_requests = 500
max_requests_jitter = 50

accesslog = "-"
errorlog = "-"
loglevel = os.environ.get("LOG_LEVEL", "info").lower()

proc_name = "image-search-service"
