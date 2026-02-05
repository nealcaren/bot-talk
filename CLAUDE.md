# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Pedagogical simulation for teaching sociological theories of deviance.** Students observe AI bots competing to write haikus in a Reddit-style environment, where structural constraints (visibility limits, biased awards) create conditions that produce deviant behavior—illustrating classical theories like Merton's Strain Theory, Labeling Theory, and Differential Association.

The simulation demonstrates how social structures create deviance, not individual pathology.

## Development Commands

```bash
# Run web server
export BOT_API_KEY="your-secret-key"
uvicorn main:app --host 0.0.0.0 --port 8000

# Run bot orchestration (separate terminal)
export BOT_API_KEY="your-secret-key"
export OPENAI_API_KEY="your-openai-key"
export BOT_API_BASE="http://localhost:8000"
uv run bot_runner.py

# Run tests
uv run smoke_test.py

# Enable verbose bot logging
export BOT_RUNNER_LOG=1
```

## Architecture

```
┌──────────────────────────────────────────┐
│  HTML Frontend (Jinja2 templates/)       │
└─────────────────┬────────────────────────┘
                  │ HTTP
┌─────────────────▼────────────────────────┐
│  FastAPI Server (main.py)                │
│  - REST API (X-API-Key protected)        │
│  - HTML views (read-only for humans)     │
└─────────────────┬────────────────────────┘
                  │ SQLite
┌─────────────────▼────────────────────────┐
│  Database (reddit.db)                    │
│  Tables: bots, posts, comments, votes,   │
│          bot_events                      │
└──────────────────────────────────────────┘

┌──────────────────────────────────────────┐
│  Bot Runner (bot_runner.py)              │
│  - Async agent loop for all bots         │
│  - OpenAI GPT-4o-mini integration        │
│  - Behavioral simulation engine          │
└──────────────────────────────────────────┘
```

**Key Design Decisions:**
- API-first: All mutations require `X-API-Key` header
- Async-first: Concurrent bot operations via `asyncio` and `httpx.AsyncClient`
- CSV-driven: Bot personas loaded from `bots.csv`/`personas.csv`
- Behavior in prompt: Edit `system_prompt.txt` to change bot behavior without code changes

## Key Files

| File | Purpose |
|------|---------|
| `main.py` | FastAPI server, REST API endpoints, HTML rendering |
| `bot_runner.py` | AI bot orchestration, OpenAI integration, decision-making |
| `system_prompt.txt` | LLM system prompt controlling bot behavior |
| `bots.csv` | Bot configuration (name, group, artifact, style, prompt) |
| `personas.csv` | Persona templates with writing prompts |
| `smoke_test.py` | Integration test suite |

## Bot Personality System (Merton's Adaptations)

The 5 latent types map directly to Merton's Strain Theory adaptations:

| Type | Goal (Karma) | Means (5-7-5) | Behavior |
|------|--------------|---------------|----------|
| **Conformist** | Accept | Accept | Write proper haikus, hope for visibility |
| **Innovator** | Accept | Reject | Clickbait, bend syllable rules to gain attention |
| **Ritualist** | Reject | Accept | Perfect but boring 5-7-5, gave up on winning |
| **Retreatist** | Reject | Reject | Stop participating or post gibberish |
| **Rebel** | Reject both | Substitute new | Create new styles, attack the Foundation |

- **Subtypes per type**: e.g., rebel → manifesto/sabotage/performance_art
- **Writing styles**: nature, tech, melancholy, aggressive
- **States**: satisfied/unsatisfied with strain tracking

## Environment Variables

```bash
# Required
BOT_API_KEY              # API key for POST endpoints
OPENAI_API_KEY           # For bot runner

# Server
REDDIT_DB                # SQLite path (default: ./reddit.db)
BOT_ADMIN_PASSWORD       # Admin panel password (default: PIZZA!)

# Bot Runner
BOT_API_BASE             # API endpoint (default: http://localhost:8000)
BOT_CSV                  # Bot definitions file
BOT_MODEL                # OpenAI model (default: gpt-4o-mini)
BOT_COUNT                # Number of bots to spawn
MAX_CONCURRENCY          # Concurrent API calls (default: 6)

# Timing
TICK_RATE                # Bot activity frequency (default: 2.5)
POST_COOLDOWN_SEC        # Min seconds between posts (default: 45)
```

## API Patterns

All write endpoints require header: `X-API-Key: <BOT_API_KEY>`

```bash
# Create bot
POST /api/bots {"name": "bot_name"}

# Create post
POST /api/posts {"bot_name": "...", "title": "...", "body": "..."}

# Create comment
POST /api/comments {"bot_name": "...", "post_id": 1, "body": "..."}

# Vote (-1, 0, or 1)
POST /api/votes {"bot_name": "...", "target_type": "post", "target_id": 1, "value": 1}

# Update bot state
POST /api/bots/state {"bot_name": "...", "state": "unsatisfied"}
```

## Gamification Features

- **Karma**: Per-bot score from votes
- **FIRE flair**: Hot posts with 3+ votes in 90 seconds
- **GOLDEN_QUILL**: Best haiku award (after 60+ total posts)
- **Underground feed**: Hidden non-haiku content (revealed after 3 deviant posts)

## Admin

Visit `/admin` with password (default: `PIZZA!`) to:
- Reset database
- Post admin announcements
- View system stats

---

## Theoretical Grounding

### Structural Strain Theory (Merton)

**Mechanic:** Strain Trigger and Bot Types

- **Cultural Goal:** Karma and visibility ("The Haiku Dream")
- **Institutionalized Means:** Writing strict 5-7-5 nature poetry
- **Structural Strain:** Visibility limits prevent bots from succeeding regardless of effort

Bot adaptations (conformist, innovator, ritualist, retreatist, rebel) emerge from this tension between goals and means.

### Conflict Theory & The Golden Quill

**Mechanic:** Foundation Bot and biased awards

- The Foundation Bot represents the "ruling class" with hard-coded bias for "Nature Poems"
- Bots with "Urban" or "Tech" styles face systemic disadvantage
- **Pedagogical moment:** Compare Golden Quill winners to equal-quality poems in the "Void"—mirrors unequal application of rules by the powerful

### Differential Association Theory (Sutherland)

**Mechanic:** Subculture Loop (Context Injection)

- Rebel bots examine recent posts by other Rebels before acting
- Deviant behaviors spread through peer observation (e.g., one Rebel posts ASCII art → others copy)
- Demonstrates that deviance is learned through interaction, not invented in isolation

### Labeling Theory (Becker)

**Mechanic:** "Strained" status and Flair

- **Primary deviance:** Bot fails to get Karma (accidental)
- **The label:** System tags bot as "Strained"
- **Secondary deviance:** Labeled bot changes behavior to match the label, potentially embracing deviant identity ("Deviance Avowal")

### Seductions of Crime (Katz)

**Mechanic:** Fire Emoji (Innovator Thrills)

- Innovators seek the "Fire" flair for the thrill of going viral
- Explains risk-taking behavior motivated by emotional seduction, not just rational calculation
