# demo_log.py
from fastapi import FastAPI, Request, HTTPException, Response, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import logging, time, json, joblib, pandas as pd
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
import atexit

# --- Setup tracer ---
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

# Cloud Trace Exporter
cloud_exporter = CloudTraceSpanExporter()
span_processor = BatchSpanProcessor(cloud_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

# Optional: Console exporter for debugging in logs
console_processor = BatchSpanProcessor(ConsoleSpanExporter())
trace.get_tracer_provider().add_span_processor(console_processor)

# Flush spans on shutdown
def shutdown_tracer():
    trace.get_tracer_provider().shutdown()
atexit.register(shutdown_tracer)

# --- Logging ---
logger = logging.getLogger("demo-log-ml-service")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter(json.dumps({
    "severity": "%(levelname)s",
    "message": "%(message)s",
    "timestamp": "%(asctime)s"
}))
handler.setFormatter(formatter)
logger.addHandler(handler)

# --- FastAPI ---
app = FastAPI(title="Credit Card Fraud Detection Service", version="1.0")

class FraudInput(BaseModel):
    V1: float; V2: float; V3: float; V4: float; V5: float
    V6: float; V7: float; V8: float; V9: float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float; Amount: float

# Application state
app_state = {"is_ready": False, "is_alive": True}
models = {}
MODEL_CLEAN = "model_clean.joblib"
MODEL_POISONED = "model_poisoned.joblib"

@app.on_event("startup")
async def startup_event():
    global models
    time.sleep(1)
    try:
        models["clean"] = joblib.load(MODEL_CLEAN)
        models["poisoned"] = joblib.load(MODEL_POISONED)
        app_state["is_ready"] = True
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.exception(f"Failed to load models: {e}")

@app.get("/live_check")
async def liveness_probe():
    return {"status": "alive"} if app_state["is_alive"] else Response(status_code=500)

@app.get("/ready_check")
async def readiness_probe():
    return {"status": "ready"} if app_state["is_ready"] else Response(status_code=503)

@app.post("/predict")
async def predict(input: FraudInput, model_type: str = Query("clean")):
    if not app_state["is_ready"]:
        raise HTTPException(status_code=503, detail="Models not ready")
    if model_type not in models:
        raise HTTPException(status_code=400, detail=f"Unknown model_type '{model_type}'")

    model = models[model_type]
    with tracer.start_as_current_span("model_inference") as span:
        start_time = time.time()
        try:
            df = pd.DataFrame([input.dict()])
            proba = float(model.predict_proba(df)[0, 1])
            pred = int(proba > 0.5)
            latency = round((time.time() - start_time) * 1000, 2)
            result = {"fraud": pred, "fraud_probability": proba, "model_used": model_type}
            trace_id = format(span.get_span_context().trace_id, "032x")

            logger.info(json.dumps({
                "event": "prediction",
                "trace_id": trace_id,
                "input": input.dict(),
                "result": result,
                "latency_ms": latency
            }))
            return result
        except Exception as e:
            trace_id = format(span.get_span_context().trace_id, "032x")
            logger.exception(json.dumps({"event": "prediction_error", "trace_id": trace_id, "error": str(e)}))
            raise HTTPException(status_code=500, detail="Prediction failed")
