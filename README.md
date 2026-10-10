# Industrial Defect Detection API (EfficientNet-B0)

A production-oriented FastAPI wrapper around the model exported by `defective-parts-detection.ipynb` for the Computer Vision Engineer technical assessment.

## 1. Model contract and results

- Model: EfficientNet-B0, fully fine-tuned on ImageNet-pretrained weights.
- Classes: `defective=0`, `normal=1`.
- Input: RGB image, resized to 224x224; ImageNet mean `[0.485,0.456,0.406]` and std `[0.229,0.224,0.225]`.
- Final checkpoint: **`/kaggle/working/outputs/final_model.pt`** (dictionary with `model_state`, `model_name`, `class_to_idx`, `img_size`, `mean`, `std`). **Do not substitute `best_model.pt`** unless the loading code is changed.
- Original notebook: training 910, validation 195, test 195; best validation defective F1 0.9914; test accuracy 1.0000 on 195 images (117 defective, 78 normal). Caution: small test set, perfect performance is not proof of generalization. A truly independent production dataset is recommended; check near-duplicates or product-level leakage.
- AUC correction: notebook computes `roc_auc_score(all_true, P(defective))` while `all_true` uses `normal=1`; this reverses the positive-class convention. Compute `roc_auc_score((np.array(all_true)==0).astype(int), all_probs)` instead.

## 2. Export your Kaggle checkpoint

The checkpoint is **NOT included** in this package. In your Kaggle notebook, execute:

```python
from IPython.display import display, FileLink
display(FileLink('/kaggle/working/outputs/final_model.pt'))
```

Download it, then copy into this project's `models/final_model.pt` folder:

```text
project/
  models/
    final_model.pt   <-- put your real model here
```

The model is intentionally ignored by Git and Docker build (not committed). On a production server provide it as a mounted private volume.

## 3. Run locally (Windows PowerShell or Mac/Linux)

Install Python 3.11, then in the project directory:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Generate a strong API key:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Then set it in your terminal. **Never upload it to GitHub.**

```powershell
# Windows PowerShell example:
$env:APP_API_KEY="PASTE_YOUR_GENERATED_SECRET_HERE"
$env:MODEL_PATH="models/final_model.pt"
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

```bash
# macOS/Linux equivalent:
export APP_API_KEY="PASTE_YOUR_GENERATED_SECRET_HERE"
export MODEL_PATH="models/final_model.pt"
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Visit **http://127.0.0.1:8000/docs**. Click **Authorize**, supply `X-API-Key`, then **POST /predict**, upload an image and press **Execute**. `GET /health` and `GET /ready` are public operational endpoints.

## 4. Example API request and response

```bash
curl -X POST 'http://127.0.0.1:8000/predict' \
  -H 'X-API-Key: YOUR_SECRET_HERE' \
  -F 'file=@sample_casting.jpg;type=image/jpeg'
```

```json
{
  "predicted_class": "defective",
  "confidence_score": 0.987,
  "probabilities": {"defective": 0.987, "normal": 0.013},
  "inference_time_ms": 85.4
}
```

Values above illustrate the response format; they are **not an actual prediction**. For Windows PowerShell, you can use `curl.exe` or the interactive `/docs` page. `client_example.py` is another option (requires `pip install requests`):

```bash
python client_example.py sample_casting.jpg
```

If using PowerShell, the sample client automatically reads the same `APP_API_KEY` environment variable. The client automatically selects the appropriate JPEG, PNG, or WEBP content type from the file extension.

## 5. Docker (consistent deployment)

```bash
docker build -t defect-detection-api .
```

Run from repository root on macOS/Linux:

```bash
docker run --rm -p 8000:8000 \
  -e APP_API_KEY='PASTE_YOUR_SECRET_HERE' \
  -e MODEL_PATH='/app/models/final_model.pt' \
  -v "$(pwd)/models:/app/models:ro" \
  defect-detection-api
```

On Windows PowerShell:

```powershell
docker run --rm -p 8000:8000 `
  -e APP_API_KEY="PASTE_YOUR_SECRET_HERE" `
  -e MODEL_PATH="/app/models/final_model.pt" `
  -v "${PWD}/models:/app/models:ro" `
  defect-detection-api
```

**Important:** Without a real `models/final_model.pt` and `APP_API_KEY`, the server intentionally fails at startup. Avoid directly exposing this simple demo API to the public internet without HTTPS, rate limiting, traffic controls and secret management.

## 6. Tests and CI/CD

```bash
pip install -r requirements-dev.txt
pytest -q
ruff check app tests --select E4,E7,E9,F
```

`tests/test_api.py` uses a fake predictor to validate API behavior without needing the large trained file. `tests/test_predictor.py` builds a synthetic EfficientNet checkpoint to check the save/load format and tensor inference. Neither test substitutes for an actual end-to-end smoke test with your model and production-like images.

GitHub Actions workflow `.github/workflows/ci-cd.yml`:

1. Runs lint + tests on pushes/PRs to `main`.
2. If the tests pass on a push to `main`, builds the CPU Docker image.
3. Publishes versioned SHA tag + `latest` to GitHub Container Registry (`ghcr.io/OWNER/REPO`).

This is **automated image publishing (CI + CD artifact release)**, not a complete live server deployment. To deploy, pull the published image on a server and run with model-volume mount and `APP_API_KEY`; integrate your hosting provider's deploy hook for fully automated rollout. Package visibility/access must be configured for your server if the GitHub package is private.

## 7. Production hardening / model limitations

- Requires the `X-API-Key` header on predictions; secrets are in environment variables, not source.
- Rejects missing/incorrect key (401), unsupported MIME/format (415), bad files (400), files larger than 10 MiB (413).
- Logs prediction class, confidence and latency, without raw image content.
- Loads the model at startup (not every request), disables gradients at inference, uses CPU by default; set `MODEL_DEVICE=cuda` only with compatible CUDA image/runtime.
- No caching/persistent storage of user images.
- A Docker image does not include the trained checkpoint. Provide and version your weights securely alongside deploys.
- Production gaps: HTTPS must be terminated by a trusted proxy; rate limiting, advanced observability, auth/key rotation, request timeouts, drift checks, calibration, independent holdout data and rollback/versioned deployment need the target infrastructure.
- Main model risk: a visually similar or shifted manufacturing line can differ from the small original casting dataset. Establish false-negative costs and defect recall before any real rejection workflow.

See [architecture.md](architecture.md) for the submission architecture diagram.

## 8. Submission checklist

- [x] Python inference API, `/predict`, confidence score, validation, logs, error handling
- [x] Dockerfile and Docker-run instructions
- [x] GitHub Actions build/test/publish workflow
- [x] README, architecture diagram, example inference request
- [ ] Insert your real checkpoint in `models/final_model.pt`
- [ ] Run a real inference sample and capture screenshots / logs for demo
- [ ] Publish repository to your own GitHub account
- [ ] Deploy to chosen host with HTTPS / secrets, if required for a public URL
- [ ] Record 2-3 minute video explaining data, model, errors, endpoint, CI/CD
