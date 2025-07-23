import os
import logging
from datetime import datetime
from collections import deque
import threading

import numpy as np
import torch
from torch.optim import AdamW
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_swagger_ui import get_swaggerui_blueprint
from transformers import BertForSequenceClassification, AutoTokenizer

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Flask app & Swagger
# -----------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)

SWAGGER_URL = "/api/docs"
API_URL = "/api/swagger.json"
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL, API_URL, config={"app_name": "Bug Severity Classifier API"}
)
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

# -----------------------------------------------------------------------------
# Globals / State
# -----------------------------------------------------------------------------
MODEL_PATH = "./model.pt"          # torch checkpoint
HF_BACKBONE = "bert-base-uncased"  # HF model/tokenizer id
LOCAL_HF_DIR = "./bert_ckpt"       # optional dir created by save_pretrained
MAX_LEN = 512

model = None
tokenizer = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
init_lock = threading.Lock()

feedback_buffer = deque(maxlen=100)

metrics = {
    "total_predictions": 0,
    "feedback_received": 0,
    "last_update": None,
    "accuracy_history": deque(maxlen=50),
    "confidence_history": deque(maxlen=100),
    "severity_distribution": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
    "api_start_time": datetime.now().isoformat(),
}

SEVERITY_LABELS = {
    1: "SEV-1 Critical",
    2: "SEV-2 Major",
    3: "SEV-3 Minor",
    4: "SEV-4 Low",
    5: "SEV-5 Trivial",
}

# -----------------------------------------------------------------------------
# Initialization
# -----------------------------------------------------------------------------
def _load_from_hf_or_local():
    """
    Load tokenizer/model from a local 'save_pretrained' dir if present,
    otherwise pull from HF hub/cache.
    """
    tok = AutoTokenizer.from_pretrained(LOCAL_HF_DIR if os.path.isdir(LOCAL_HF_DIR) else HF_BACKBONE)
    mdl = BertForSequenceClassification.from_pretrained(
        LOCAL_HF_DIR if os.path.isdir(LOCAL_HF_DIR) else HF_BACKBONE,
        num_labels=5,
    )
    return tok, mdl


def initialize_model():
    global model, tokenizer, device

    with init_lock:
        if model is not None and tokenizer is not None:
            return

        logger.info("Initializing model/tokenizer...")
        if torch.cuda.is_available():
            device = torch.device("cuda")
            torch.cuda.set_device(0)
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Mem: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            device = torch.device("cpu")
            logger.warning("CUDA not available. Using CPU.")

        if not os.path.exists(MODEL_PATH):
            logger.error(f"Model checkpoint not found at {MODEL_PATH}.")
            # try best-effort to at least load a base model
            tokenizer, model = _load_from_hf_or_local()
            model.to(device).eval()
            return

        # Load torch checkpoint
        ckpt = torch.load(MODEL_PATH, map_location=device)

        # 1) tokenizer: try checkpoint object, then LOCAL_HF_DIR, then HF
        tok_from_ckpt = ckpt.get("tokenizer")
        if tok_from_ckpt is not None:
            tokenizer = tok_from_ckpt
            logger.info("Tokenizer loaded from torch checkpoint object.")
        else:
            tokenizer, _ = _load_from_hf_or_local()
            logger.info("Tokenizer loaded from local/HF.")

        # 2) model: recreate arch then load state
        _, base_model = _load_from_hf_or_local()
        base_model.load_state_dict(ckpt["model_state_dict"])
        model = base_model.to(device)
        model.eval()

        if device.type == "cuda":
            torch.cuda.empty_cache()
            logger.info(f"Model on GPU. Allocated: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")
        else:
            logger.info(f"Model on {device}")


def _ensure_ready():
    """Ensure model/tokenizer are available, try initializing if not."""
    if model is None or tokenizer is None:
        initialize_model()
    return model is not None and tokenizer is not None


model_ready = False

@app.before_request
def _ensure_loaded_once():
    global model_ready
    if not model_ready:
        initialize_model()
        model_ready = True

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def calculate_business_metrics():
    avg_time_per_ticket = 5  # minutes saved per prediction
    hourly_rate = 50         # QA hourly rate
    time_saved = metrics["total_predictions"] * avg_time_per_ticket / 60
    cost_saved = time_saved * hourly_rate

    return {
        "tickets_processed": metrics["total_predictions"],
        "hours_saved": round(time_saved, 1),
        "cost_saved_usd": round(cost_saved, 2),
        "avg_confidence": round(np.mean(metrics["confidence_history"]) if metrics["confidence_history"] else 0, 3),
        "accuracy_trend": round(np.mean(metrics["accuracy_history"]) if metrics["accuracy_history"] else 0, 3),
    }


def update_model_with_feedback():
    global model, feedback_buffer, metrics

    if len(feedback_buffer) == 0:
        return

    logger.info(f"Updating model with {len(feedback_buffer)} feedback items...")
    model.train()
    optimizer = AdamW(model.parameters(), lr=1e-6, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    correct_predictions = 0
    feedback_to_process = list(feedback_buffer)
    feedback_buffer.clear()

    for item in feedback_to_process:
        encoding = tokenizer(
            item["description"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(device, non_blocking=True)
        attention_mask = encoding["attention_mask"].to(device, non_blocking=True)
        label = torch.tensor([item["correct_severity"] - 1]).to(device, non_blocking=True)

        optimizer.zero_grad()
        if device.type == "cuda" and scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=label)
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=label)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        with torch.no_grad():
            _, predicted = torch.max(outputs.logits, dim=1)
            if predicted.item() == label.item():
                correct_predictions += 1

    model.eval()

    if feedback_to_process:
        accuracy = correct_predictions / len(feedback_to_process)
        metrics["accuracy_history"].append(accuracy)
        metrics["last_update"] = datetime.now().isoformat()
        logger.info(f"Model updated. Accuracy on feedback: {accuracy:.2%}")

    # Save updated model state
    torch.save({"model_state_dict": model.state_dict(), "tokenizer": tokenizer}, "model_updated.pt")

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.route("/api/swagger.json")
def swagger_spec():
    return jsonify({
        "openapi": "3.0.0",
        "info": {
            "title": "Bug Severity Classifier API",
            "version": "1.0.0",
            "description": "AI-powered bug severity classification with 5-tier Atlassian severity levels (1-5)"
        },
        "paths": {
            "/predict": {"post": {"summary": "Classify bug severity (1-5 scale)"}},
            "/feedback": {"post": {"summary": "Provide feedback to improve model"}},
            "/metrics": {"get": {"summary": "Get model performance metrics"}},
            "/business-value": {"get": {"summary": "Calculate business value and ROI"}}
        }
    })


@app.route("/")
def home():
    return jsonify({
        "service": "Bug Severity Classifier API",
        "version": "1.0",
        "documentation": "/api/docs",
        "endpoints": {
            "/predict": "POST - Classify bug severity",
            "/feedback": "POST - Improve model with corrections",
            "/metrics": "GET - Performance metrics",
            "/business-value": "GET - ROI calculations",
            "/health": "GET - Service health"
        }
    })


@app.route("/predict", methods=["POST"])
def predict():
    if not _ensure_ready():
        return jsonify({"error": "Model/tokenizer not initialized"}), 503

    try:
        data = request.get_json(force=True, silent=True) or {}
        description = data.get("description", "").strip()
        if not description:
            return jsonify({"error": "Description cannot be empty"}), 400

        encoding = tokenizer(
            description,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(device, non_blocking=True)
        attention_mask = encoding["attention_mask"].to(device, non_blocking=True)

        with torch.no_grad():
            if device.type == "cuda":
                with torch.cuda.amp.autocast():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_attentions=True
                    )
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=True
                )

            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            confidence, predicted_class = torch.max(probabilities, dim=1)
            all_probs = probabilities.cpu().numpy()[0]

        severity = predicted_class.item() + 1
        confidence_score = confidence.item()

        metrics["total_predictions"] += 1
        metrics["confidence_history"].append(confidence_score)
        metrics["severity_distribution"][severity] += 1

        result = {
            "severity": severity,
            "severity_label": SEVERITY_LABELS[severity],
            "confidence": round(confidence_score, 3),
            "probabilities": {f"severity_{i+1}": round(float(p), 3) for i, p in enumerate(all_probs)},
            "requires_review": confidence_score < 0.7,
            "timestamp": datetime.now().isoformat(),
            "model_version": metrics.get("last_update", "initial"),
            "business_impact": {
                "time_saved_minutes": 10,
                "priority_level": "P0" if severity == 1 else f"P{severity-1}",
                "atlassian_tier": (
                    "Critical" if severity <= 2 else ("Major" if severity == 3 else ("Minor" if severity == 4 else "Trivial"))
                )
            },
            "device_info": {
                "device": str(device),
                "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None
            }
        }

        return jsonify(result)

    except Exception as e:
        logger.exception("Prediction error")
        return jsonify({"error": str(e)}), 500


@app.route("/feedback", methods=["POST"])
def feedback():
    if not _ensure_ready():
        return jsonify({"error": "Model/tokenizer not initialized"}), 503

    try:
        data = request.get_json(force=True, silent=True) or {}
        feedback_item = {
            "description": data["description"],
            "predicted_severity": data["predicted_severity"],
            "correct_severity": data["correct_severity"],
            "timestamp": datetime.now().isoformat()
        }

        feedback_buffer.append(feedback_item)
        metrics["feedback_received"] += 1

        is_correct = data["predicted_severity"] == data["correct_severity"]
        metrics["accuracy_history"].append(1 if is_correct else 0)

        if len(feedback_buffer) > 0:
            threading.Thread(target=update_model_with_feedback, daemon=True).start()

        return jsonify({
            "status": "success",
            "message": "Feedback recorded",
            "feedback_count": len(feedback_buffer),
            "update_pending": len(feedback_buffer) > 0,
            "current_accuracy": round(np.mean(metrics["accuracy_history"]) if metrics["accuracy_history"] else 0, 3)
        })

    except Exception as e:
        logger.exception("Feedback error")
        return jsonify({"error": str(e)}), 500


@app.route("/metrics", methods=["GET"])
def get_metrics():
    recent_accuracy = np.mean(metrics["accuracy_history"]) if metrics["accuracy_history"] else 0
    avg_confidence = np.mean(metrics["confidence_history"]) if metrics["confidence_history"] else 0

    return jsonify({
        "performance": {
            "total_predictions": metrics["total_predictions"],
            "recent_accuracy": round(recent_accuracy, 3),
            "average_confidence": round(avg_confidence, 3),
            "feedback_received": metrics["feedback_received"]
        },
        "severity_distribution": metrics["severity_distribution"],
        "model_info": {
            "last_update": metrics["last_update"],
            "device": str(device) if "device" in globals() else "not initialized",
            "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "gpu_memory_allocated": f"{torch.cuda.memory_allocated(0)/1024**3:.2f} GB" if device.type == "cuda" else None,
            "uptime": str(datetime.now() - datetime.fromisoformat(metrics["api_start_time"]))
        }
    })


@app.route("/business-value", methods=["GET"])
def business_value():
    return jsonify(calculate_business_metrics())


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "uptime": str(datetime.now() - datetime.fromisoformat(metrics["api_start_time"]))
    })

# -----------------------------------------------------------------------------
# Entry
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Bug Severity Classifier API - RTX 3090 Ti Optimized")
    print("=" * 60)

    with app.app_context():
        initialize_model()

    print("\nAPI Documentation: http://localhost:5000/api/docs")
    print("Business Value Dashboard: http://localhost:5000/business-value")
    if device.type == "cuda":
        print(f"Running on: {torch.cuda.get_device_name(0)}")
    print("\nStarting server...")

    app.run(debug=True, host="0.0.0.0", port=5000)
