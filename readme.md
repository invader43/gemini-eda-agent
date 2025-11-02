# Run the EDA Langgraph Agent using requirements.txt

Quick steps to install dependencies and run `app.py` (using uv = uvicorn).

## 1. requirements.txt
Create a `requirements.txt` containing at least:
```
chainlit
uvicorn[standard]
```

## 2. Create & activate virtual environment
Unix / macOS:
```bash
python -m venv .venv
source .venv/bin/activate
```
Windows (PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

## 3. Install dependencies
```bash
pip install -r requirements.txt
```

## 4. Ensure your app is ASGI-exposable
uvicorn requires an ASGI callable in your `app.py`. Common pattern:
- Export an ASGI object named `app` (e.g., a FastAPI or Chainlit ASGI app)
- If the callable has a different name, use that name when running uvicorn

Example callable reference: `app:app` (module `app`, variable `app`)

## 5. Run with uvicorn
Basic:
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```
If your callable is named `main`, run `uvicorn app:main`.

Note: Chainlit also provides `chainlit run app.py` as an alternate runner; the uvicorn command above assumes `app.py` exposes an ASGI app compatible with uvicorn.