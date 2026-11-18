# Animated GUI for Offline RAG

This repository includes a simple animated GUI `app_gui.py` built with PySide6.

Quick start:

1. Create and activate your Python environment (recommended):

```powershell
py -3 -m venv .venv
.\.venv\Scripts\activate
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run the GUI:

```powershell
py app_gui.py
```

Usage notes:
- Click "Choose Folder" to select a folder containing documents or audio for ingestion.
- Click "Start Ingestion" to run `run_ingestion` from `ingestion.py` in a subprocess; status messages stream to the log.
- If dependencies required by `ingestion.py` are missing, the ingestion subprocess will print an error.

If you want a custom spinner GIF, place `spinner.gif` next to `app_gui.py`.
