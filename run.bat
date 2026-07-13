@echo off
REM rag-eval-bench — Quick start for Windows
REM Requires Python 3.12+ and pip

echo [rag-eval-bench] Setting up virtual environment...
python -m venv .venv
call .venv\Scripts\activate.bat

echo [rag-eval-bench] Installing dependencies...
pip install -e .

echo.
echo [rag-eval-bench] Done. Run commands:
echo   rag-eval index --help
echo   rag-eval qa-gen --help
echo   rag-eval retrieval --help
echo   rag-eval bench --help
echo.
echo Optional: start Ollama for generation scoring:
echo   ollama serve
echo   ollama pull gemma3:4b
