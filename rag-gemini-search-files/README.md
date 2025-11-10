# rag-gemini-search-files

Protótipo simples de RAG usando Gemini (Search Files) para upload, indexação e consulta de documentos.

## Requisitos

- Python 3.12
- Variável de ambiente `GOOGLE_API_KEY` (ou arquivo `.env` com `GOOGLE_API_KEY=...`)

## Instalação (opcional)

```bash
# ambiente local
python -m venv .venv
source .venv/bin/activate

# instalar o projeto e deps de dev
pip install -e .[dev]
```

## Uso resumido

Este repositório é um esqueleto; adicione seus scripts dentro de `src/` para fazer upload de arquivos e consultar o modelo Gemini.

Exemplos de bibliotecas úteis:

- `google-generativeai` – SDK Gemini
- `python-dotenv` – Carregar `.env`

## Licença

MIT
