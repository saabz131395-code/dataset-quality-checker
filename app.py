"""
app.py - Flask web interface for the Dataset Quality Checker.
Handles the upload page, reads the CSV, and shows the report.
"""
import csv
import io
import os

import pandas as pd
from flask import Flask, redirect, render_template, request, url_for

from analyzer import analyze

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB upload limit

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_FILE = os.path.join(BASE_DIR, "sample_data.csv")


def _decode(raw_bytes):
    """Decode as UTF-8 first, then fall back to Latin-1 for files saved in older encodings."""
    try:
        return raw_bytes.decode("utf-8-sig"), "utf-8"
    except UnicodeDecodeError:
        return raw_bytes.decode("latin-1"), "latin-1"


def read_csv_safely(raw_bytes):
    """
    Read an uploaded CSV and return (dataframe, notes).
    - rejects binary files (images, PDFs, Excel files renamed to .csv)
    - detects the delimiter: comma, semicolon, tab or pipe
    - warns about duplicate column names
    Raises ValueError with a friendly message if the file cannot be read.
    """
    if b"\x00" in raw_bytes[:8192]:
        raise ValueError("This looks like a binary file (for example an image or an Excel file "
                         "renamed to .csv), not a text CSV file.")

    text, encoding = _decode(raw_bytes)
    sample = text[:64 * 1024]
    if not sample.strip():
        raise ValueError("The file is empty.")

    try:
        delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        delimiter = ","

    notes = []
    if delimiter != ",":
        names = {";": "semicolon", "\t": "tab", "|": "pipe"}
        notes.append(f"This file uses {names[delimiter]}-separated values; it was detected automatically.")

    header = next(csv.reader(io.StringIO(sample), delimiter=delimiter), [])
    dupes = sorted({h for h in header if header.count(h) > 1})
    if dupes:
        notes.append(f"Duplicate column names found: {', '.join(dupes)}. "
                     f"pandas renamed the copies (for example '{dupes[0]}.1').")

    try:
        df = pd.read_csv(io.StringIO(text), sep=delimiter)
    except pd.errors.EmptyDataError:
        raise ValueError("The file is empty.")
    except pd.errors.ParserError as e:
        raise ValueError(f"The rows do not have a consistent number of columns ({e}).")
    return df, notes


def render_report(df, filename, label_col, notes=None):
    warnings = list(notes or [])
    if label_col and label_col not in df.columns:
        available = ", ".join(map(str, df.columns))
        warnings.append(f"Label column '{label_col}' was not found, so class balance was skipped. "
                        f"Available columns: {available}")
        label_col = None

    report = analyze(df, label_col)
    preview = df.head(10).astype(object).where(df.head(10).notna(), None)
    return render_template(
        "report.html",
        report=report,
        filename=filename,
        warnings=warnings,
        preview_columns=[str(c) for c in df.columns],
        preview_rows=[[("" if v is None else str(v)) for v in row]
                      for row in preview.itertuples(index=False)],
    )


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", has_sample=os.path.exists(SAMPLE_FILE))


@app.route("/analyze", methods=["GET"])
def analyze_get():
    """Someone opened /analyze directly (bookmark, shared or copied link): send them to the upload page."""
    return redirect(url_for("index"))


@app.route("/analyze", methods=["POST"])
def analyze_file():
    file = request.files.get("file")
    label_col = request.form.get("label_col", "").strip() or None

    # --- Input validation ---
    if not file or file.filename == "":
        return render_template("index.html", error="Please choose a CSV file.",
                               has_sample=os.path.exists(SAMPLE_FILE))
    if not file.filename.lower().endswith(".csv"):
        return render_template("index.html", error="Only .csv files are supported.",
                               has_sample=os.path.exists(SAMPLE_FILE))

    try:
        df, notes = read_csv_safely(file.read())
    except ValueError as e:
        return render_template("index.html", error=str(e),
                               has_sample=os.path.exists(SAMPLE_FILE))
    except Exception as e:
        return render_template("index.html", error=f"Could not read the file: {e}",
                               has_sample=os.path.exists(SAMPLE_FILE))

    if df.empty:
        return render_template("index.html", error="The file has no data rows.",
                               has_sample=os.path.exists(SAMPLE_FILE))

    return render_report(df, file.filename, label_col, notes)


@app.route("/sample", methods=["GET"])
def sample():
    """Run the checker on the built-in messy sample dataset (demo button)."""
    if not os.path.exists(SAMPLE_FILE):
        return render_template("index.html", error="Sample dataset not found.", has_sample=False)
    df = pd.read_csv(SAMPLE_FILE)
    return render_report(df, "sample_data.csv", "label")


@app.errorhandler(413)
def file_too_large(e):
    return render_template("index.html", error="File is too large. Maximum size is 10 MB.",
                           has_sample=os.path.exists(SAMPLE_FILE)), 413


if __name__ == "__main__":
    app.run(debug=True)
