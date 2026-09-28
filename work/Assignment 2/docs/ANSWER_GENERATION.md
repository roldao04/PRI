# Answer Generation with Gemini AI

This document describes the implementation of AI-powered answer generation using Google Gemini API (Assignment 2, Task 2).

## Overview

The system uses Google Gemini to generate concise, direct answers based on the top-ranked document from search results. Answer generation is **optional** and can be enabled through the web interface.

## Model

- **Model**: `gemini-2.5-flash-lite`

**Prompt Structure:**

```
És um motor de busca inteligente. A tua tarefa é responder à pergunta do utilizador
de forma concisa e direta com base APENAS no documento fornecido.

**Instruções:**
- Responde em português de Portugal (pt-PT)
- Sê conciso (máximo 2-3 frases)
- Responde apenas com conteúdo relevante ao que foi perguntado
- Usa informações APENAS do documento fornecido
- Se o documento não contém a resposta, diz "Não há informação suficiente no documento para responder."
- Não inventes informações

**Pergunta do utilizador:**
{query}

**Documento:**
Título: {title}

Conteúdo: {text}

**Resposta:**
```

## Setup
Add your key to `.env`:

```env
# Google Gemini API Key for Answer Generation
# Get your key at: https://aistudio.google.com/app/apikey
GEMINI_API_KEY=your_api_key_here
```

## Usage

### API

```bash
GET /api/v1/search?query=exemplo&generate_answers=true
```

Response includes `answer_snippets`:

```json
{
  "query": "exemplo",
  "results": [...],
  "answer_snippets": [
    {
      "doc_id": 123,
      "answer": "Generated answer text...",
      "success": true,
      "error": null
    }
  ]
}
```
