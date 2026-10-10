"""FastAPI image inference service with API-key auth and defensive validation."""

from contextlib import asynccontextmanager
from io import BytesIO
import hmac
import logging
import os
import warnings

from fastapi import (
    Depends,
    FastAPI,
    File,
    HTTPException,
    Request,
    Security,
    UploadFile
)

from fastapi.concurrency import run_in_threadpool
from fastapi.security import APIKeyHeader

from PIL import (
    Image,
    UnidentifiedImageError
)

from pydantic import BaseModel, Field

from .predictor import Predictor



logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)


logger = logging.getLogger("defect_api")



API_KEY_HEADER = APIKeyHeader(
    name="X-API-Key",
    auto_error=False
)


ALLOWED_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp"
}


ALLOWED_FORMATS = {
    "JPEG",
    "PNG",
    "WEBP"
}


MAX_UPLOAD_BYTES = 10 * 1024 * 1024


Image.MAX_IMAGE_PIXELS = 25_000_000





class PredictionResponse(BaseModel):

    predicted_class: str

    confidence_score: float = Field(
        ge=0,
        le=1
    )

    probabilities: dict[str, float]

    inference_time_ms: float = Field(
        ge=0
    )






@asynccontextmanager
async def lifespan(app: FastAPI):

    key = os.getenv(
        "APP_API_KEY",
        ""
    )


    if len(key) < 24:

        raise RuntimeError(
            "APP_API_KEY must be set to a strong random secret (24+ characters)."
        )


    app.state.api_key = key


    checkpoint_path = os.getenv(
        "MODEL_PATH",
        "models/final_model.pt"
    )


    device = os.getenv(
        "MODEL_DEVICE",
        "cpu"
    )


    app.state.predictor = Predictor(
        checkpoint_path,
        device=device
    )


    logger.info(
        "Loaded EfficientNet-B3 model from %s on %s",
        checkpoint_path,
        device
    )


    yield


    logger.info(
        "Shutting down inference API"
    )







app = FastAPI(

    title="Industrial Defect Detection API",

    description=(
        "EfficientNet-B3: classify casting images "
        "as defective or normal."
    ),

    version="2.0.0",

    lifespan=lifespan
)







async def require_api_key(

    request: Request,

    supplied: str | None = Security(API_KEY_HEADER)

) -> None:


    if supplied is None or not hmac.compare_digest(

        supplied,

        request.app.state.api_key

    ):

        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key"
        )








@app.get("/health")
async def health():

    """
    Liveness probe.
    """

    return {
        "status": "ok"
    }








@app.get("/ready")
async def ready(request: Request):

    """
    Readiness probe.
    """

    if getattr(
        request.app.state,
        "predictor",
        None
    ) is None:

        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )


    return {

        "status": "ready",

        "model": "EfficientNet-B3"

    }








@app.post(
    "/predict",
    response_model=PredictionResponse,
    dependencies=[Depends(require_api_key)]
)

async def predict(

    request: Request,

    file: UploadFile = File(...)

):


    if file.content_type not in ALLOWED_TYPES:

        raise HTTPException(

            status_code=415,

            detail="Only JPEG, PNG and WEBP images are supported."

        )




    try:

        data = await file.read(
            MAX_UPLOAD_BYTES + 1
        )


    finally:

        await file.close()




    if not data:

        raise HTTPException(

            status_code=400,

            detail="Empty image file"

        )




    if len(data) > MAX_UPLOAD_BYTES:

        raise HTTPException(

            status_code=413,

            detail="Image exceeds 10 MiB limit"

        )






    try:


        with warnings.catch_warnings():

            warnings.simplefilter(
                "error",
                Image.DecompressionBombWarning
            )


            with Image.open(
                BytesIO(data)
            ) as img:


                if img.format not in ALLOWED_FORMATS:

                    raise HTTPException(

                        status_code=415,

                        detail="Unsupported encoded image format"

                    )


                img.load()


                image = img.convert(
                    "RGB"
                )



    except (

        UnidentifiedImageError,

        OSError,

        ValueError,

        Image.DecompressionBombWarning,

        Image.DecompressionBombError

    ) as exc:


        logger.warning(

            "Invalid image upload: %s",

            type(exc).__name__

        )


        raise HTTPException(

            status_code=400,

            detail="Invalid or damaged image"

        ) from exc





    try:


        result = await run_in_threadpool(

            request.app.state.predictor.predict,

            image

        )


    except Exception as exc:


        logger.exception(
            "Model inference failed"
        )


        raise HTTPException(

            status_code=500,

            detail="Internal inference error"

        ) from exc





    logger.info(

        "Inference class=%s confidence=%.4f latency_ms=%.2f",

        result["predicted_class"],

        result["confidence_score"],

        result["inference_time_ms"]

    )


    return result