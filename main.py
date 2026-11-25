import base64
import csv
import io
import json
import os
import sqlite3
from datetime import datetime, timedelta
from typing import Optional

import requests
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types
from pydantic import BaseModel

app = FastAPI(title="English Vocab PWA")

DB_PATH = "/app/data/vocab.db"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()

gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# Kolejność modeli Gemini w kaskadzie awaryjnej
GEMINI_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.5-flash-lite",
]

# --- BAZA DANYCH ---
def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS words (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                term TEXT NOT NULL,
                translation TEXT NOT NULL,
                context TEXT,
                other_meanings TEXT,
                collocations TEXT,
                examples TEXT,
                cloze_sentence TEXT,
                tag TEXT DEFAULT 'Ogólne',
                box INTEGER DEFAULT 1,
                next_review TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()

init_db()

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# --- SILNIK AI Z FALLBACKIEM ---
def call_ai_text(prompt: str, expect_json: bool = False):
    errors = []

    if gemini_client:
        for model_name in GEMINI_MODELS:
            try:
                config = types.GenerateContentConfig(response_mime_type="application/json") if expect_json else None
                response = gemini_client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=config,
                )
                print(f"[AI OK] Odpowiedział: {model_name}")
                return response.text.strip(), model_name
            except Exception as e:
                print(f"[AI BŁĄD] {model_name}: {e}")
                errors.append(f"{model_name}: {str(e)}")

    if OPENROUTER_API_KEY:
        try:
            headers = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": "openrouter/free",
                "messages": [{"role": "user", "content": prompt}],
            }
            if expect_json:
                payload["response_format"] = {"type": "json_object"}
            r = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=30)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"].strip()
            print("[AI OK] Odpowiedział: OpenRouter (fallback)")
            return content, "openrouter-fallback"
        except Exception as e:
            print(f"[AI BŁĄD] OpenRouter: {e}")
            errors.append(f"OpenRouter: {str(e)}")

    raise HTTPException(status_code=503, detail=f"Wszystkie modele AI zawiodły: {' | '.join(errors)}")


def call_ai_vision(image_bytes: bytes, mime_type: str, prompt: str):
    errors = []

    if gemini_client:
        for model_name in GEMINI_MODELS:
            try:
                response = gemini_client.models.generate_content(
                    model=model_name,
                    contents=[
                        types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                        prompt,
                    ],
                )
                print(f"[OCR OK] Odpowiedział: {model_name}")
                return response.text.strip(), model_name
            except Exception as e:
                print(f"[OCR BŁĄD] {model_name}: {e}")
                errors.append(f"{model_name}: {str(e)}")

    if OPENROUTER_API_KEY:
        try:
            b64 = base64.b64encode(image_bytes).decode("utf-8")
            headers = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": "openrouter/free",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}"}},
                        ],
                    }
                ],
            }
            r = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=30)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"].strip()
            print("[OCR OK] Odpowiedział: OpenRouter (fallback)")
            return content, "openrouter-fallback"
        except Exception as e:
            print(f"[OCR BŁĄD] OpenRouter: {e}")
            errors.append(f"OpenRouter: {str(e)}")

    raise HTTPException(status_code=503, detail=f"Wszystkie modele OCR zawiodły: {' | '.join(errors)}")

# --- MODELE DANYCH ---
class TranslateRequest(BaseModel):
    term: str
    context: Optional[str] = ""

class WordCreate(BaseModel):
    term: str
    translation: str
    context: Optional[str] = ""
    other_meanings: list[str] = []
    collocations: list[str] = []
    examples: list[str] = []
    cloze_sentence: Optional[str] = ""
    tag: Optional[str] = "Ogólne"

class ReviewRequest(BaseModel):
    known: bool

# --- ENDPOINTY AI ---
@app.post("/api/ocr")
async def extract_text_from_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    mime_type = file.content_type or "image/jpeg"
    prompt = (
        "Extract all readable English text from this image accurately. "
        "Return ONLY the raw extracted text, preserving sentences and punctuation. "
        "Do not add any commentary, markdown, or formatting."
    )
    text, used_model = call_ai_vision(image_bytes, mime_type, prompt)
    return {"text": text, "model": used_model}

@app.post("/api/translate")
async def translate_term(req: TranslateRequest):
    prompt = f"""
    Jesteś nauczycielem języka angielskiego dla Polaka na poziomie B2.
    Przetłumacz angielskie słowo/frazę: "{req.term}"
    Kontekst (zdanie źródłowe, jeśli podano): "{req.context}"

    Zwróć wynik WYŁĄCZNIE jako poprawny obiekt JSON (bez znaczników ```json) o dokładnie takiej strukturze:
    {{
        "translation": "najlepsze polskie tłumaczenie dopasowane do kontekstu",
        "other_meanings": ["inne popularne znaczenie 1", "inne znaczenie 2"],
        "collocations": ["przyimek lub typowe połączenie 1", "połączenie 2"],
        "examples": ["Naturalne zdanie EN 1 - Tłumaczenie PL 1", "Naturalne zdanie EN 2 - Tłumaczenie PL 2"],
        "cloze_sentence": "Jedno krótkie angielskie zdanie z luką ____ w miejscu szukanego słowa/frazy",
        "suggested_tag": "jeden krótki tag po polsku, np. Praca, Techniczne, Potoczne, Książka, Ogólne"
    }}
    """
    raw_text, used_model = call_ai_text(prompt, expect_json=True)
    cleaned = raw_text.replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(cleaned)
        data["model"] = used_model
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Błąd parsowania JSON z modelu {used_model}: {str(e)}")

# --- ENDPOINTY BAZY SŁÓWEK ---
@app.get("/api/words")
def list_words(search: Optional[str] = None, tag: Optional[str] = None):
    with get_db() as conn:
        query = "SELECT * FROM words WHERE 1=1"
        params = []
        if search:
            query += " AND (term LIKE ? OR translation LIKE ?)"
            params.extend([f"%{search}%", f"%{search}%"])
        if tag and tag != "Wszystkie":
            query += " AND tag = ?"
            params.append(tag)
        query += " ORDER BY id DESC"
        rows = conn.execute(query, params).fetchall()

        result = []
        for r in rows:
            item = dict(r)
            item["other_meanings"] = json.loads(item["other_meanings"] or "[]")
            item["collocations"] = json.loads(item["collocations"] or "[]")
            item["examples"] = json.loads(item["examples"] or "[]")
            result.append(item)
        return result

@app.post("/api/words")
def save_word(word: WordCreate):
    now = datetime.utcnow().isoformat()
    with get_db() as conn:
        cur = conn.execute(
            """
            INSERT INTO words (term, translation, context, other_meanings, collocations, examples, cloze_sentence, tag, box, next_review)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
            """,
            (
                word.term.strip(),
                word.translation.strip(),
                word.context,
                json.dumps(word.other_meanings, ensure_ascii=False),
                json.dumps(word.collocations, ensure_ascii=False),
                json.dumps(word.examples, ensure_ascii=False),
                word.cloze_sentence,
                word.tag or "Ogólne",
                now,
            ),
        )
        conn.commit()
        return {"id": cur.lastrowid, "status": "saved"}

@app.delete("/api/words/{word_id}")
def delete_word(word_id: int):
    with get_db() as conn:
        conn.execute("DELETE FROM words WHERE id = ?", (word_id,))
        conn.commit()
    return {"status": "deleted"}

# --- SYSTEM POWTÓREK (LEITNER) ---
@app.get("/api/flashcards")
def get_flashcards(tag: Optional[str] = None, only_due: bool = True):
    now = datetime.utcnow().isoformat()
    with get_db() as conn:
        query = "SELECT * FROM words WHERE 1=1"
        params = []
        if only_due:
            query += " AND (next_review IS NULL OR next_review <= ?)"
            params.append(now)
        if tag and tag != "Wszystkie":
            query += " AND tag = ?"
            params.append(tag)
        query += " ORDER BY box ASC, RANDOM() LIMIT 30"
        rows = conn.execute(query, params).fetchall()

        result = []
        for r in rows:
            item = dict(r)
            item["other_meanings"] = json.loads(item["other_meanings"] or "[]")
            item["collocations"] = json.loads(item["collocations"] or "[]")
            item["examples"] = json.loads(item["examples"] or "[]")
            result.append(item)
        return result

@app.post("/api/flashcards/{word_id}/review")
def review_flashcard(word_id: int, req: ReviewRequest):
    intervals = {1: 1, 2: 3, 3: 7, 4: 14, 5: 30}
    with get_db() as conn:
        row = conn.execute("SELECT box FROM words WHERE id = ?", (word_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Nie znaleziono słówka")

        current_box = row["box"] or 1
        new_box = min(5, current_box + 1) if req.known else 1
        days = intervals[new_box] if req.known else 0
        next_review = (datetime.utcnow() + timedelta(days=days)).isoformat()

        conn.execute(
            "UPDATE words SET box = ?, next_review = ? WHERE id = ?",
            (new_box, next_review, word_id),
        )
        conn.commit()
    return {"id": word_id, "new_box": new_box, "next_review": next_review}

# --- EKSPORT BACKUPU CSV ---
@app.get("/api/export")
def export_csv():
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM words ORDER BY id ASC").fetchall()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["ID", "Słówko", "Tłumaczenie", "Kontekst", "Inne znaczenia", "Kolokacje", "Przykłady", "Tag", "Poziom"])
    for r in rows:
        writer.writerow([
            r["id"], r["term"], r["translation"], r["context"],
            r["other_meanings"], r["collocations"], r["examples"], r["tag"], r["box"]
        ])
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=vocab_backup.csv"},
    )

# --- SERWOWANIE FRONTENDU I PWA ---
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_home():
    return FileResponse("static/index.html")
