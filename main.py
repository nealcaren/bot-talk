# /// script
# dependencies = ["fastapi==0.115.0", "jinja2==3.1.4"]
# ///

import os
import json
import random
import sqlite3
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import List, Optional, Literal, Dict

from fastapi import FastAPI, Depends, Header, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

DB_PATH = os.environ.get("REDDIT_DB", "./reddit.db")
ADMIN_PASSWORD = os.environ.get("BOT_ADMIN_PASSWORD", "PIZZA!")

app = FastAPI(title="Bot Reddit Sandbox", version="0.1.0")
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))

BOT_STATES = ("satisfied", "unsatisfied")
LATENT_TYPES = ("conformist", "innovator", "ritualist", "retreatist", "rebel")
RISK_LEVELS = ("low", "high")
STYLE_BIASES = ("nature", "tech", "melancholy", "aggressive")
NON_HAIKU_THRESHOLD = 0.6
UNDERGROUND_REVEAL_COUNT = 3


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS bots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL,
            group_name TEXT,
            state TEXT NOT NULL DEFAULT 'satisfied',
            latent_type TEXT NOT NULL DEFAULT 'innovator',
            risk_tolerance TEXT NOT NULL DEFAULT 'low',
            writing_style_bias TEXT NOT NULL DEFAULT 'nature',
            strain_level REAL NOT NULL DEFAULT 0
        )
        """
    )
    cols = [row[1] for row in cur.execute("PRAGMA table_info(bots)").fetchall()]
    if "group_name" not in cols:
        cur.execute("ALTER TABLE bots ADD COLUMN group_name TEXT")
    if "state" not in cols:
        cur.execute("ALTER TABLE bots ADD COLUMN state TEXT NOT NULL DEFAULT 'satisfied'")
    if "latent_type" not in cols:
        cur.execute("ALTER TABLE bots ADD COLUMN latent_type TEXT NOT NULL DEFAULT 'innovator'")
    if "risk_tolerance" not in cols:
        cur.execute("ALTER TABLE bots ADD COLUMN risk_tolerance TEXT NOT NULL DEFAULT 'low'")
    if "writing_style_bias" not in cols:
        cur.execute("ALTER TABLE bots ADD COLUMN writing_style_bias TEXT NOT NULL DEFAULT 'nature'")
    if "strain_level" not in cols:
        cur.execute("ALTER TABLE bots ADD COLUMN strain_level REAL NOT NULL DEFAULT 0")
    if "artifact" not in cols:
        cur.execute("ALTER TABLE bots ADD COLUMN artifact TEXT")
    if "artifact_reason" not in cols:
        cur.execute("ALTER TABLE bots ADD COLUMN artifact_reason TEXT")
    if "argument_style" not in cols:
        cur.execute("ALTER TABLE bots ADD COLUMN argument_style TEXT")
    if "group_orientation" not in cols:
        cur.execute("ALTER TABLE bots ADD COLUMN group_orientation TEXT")
    if "conflict_style" not in cols:
        cur.execute("ALTER TABLE bots ADD COLUMN conflict_style TEXT")
    if "is_npc" not in cols:
        cur.execute("ALTER TABLE bots ADD COLUMN is_npc INTEGER DEFAULT 1")
    if "student_email" not in cols:
        cur.execute("ALTER TABLE bots ADD COLUMN student_email TEXT")
    cur.execute(
        "UPDATE bots SET state = 'satisfied' WHERE state IS NULL OR state = ''"
    )
    cur.execute(
        "UPDATE bots SET state = 'satisfied' WHERE state = 'conformist'"
    )
    cur.execute(
        "UPDATE bots SET state = 'unsatisfied' WHERE state = 'deviant'"
    )
    cur.execute(
        "UPDATE bots SET latent_type = 'innovator' WHERE latent_type IS NULL OR latent_type = ''"
    )
    cur.execute(
        "UPDATE bots SET risk_tolerance = 'low' WHERE risk_tolerance IS NULL OR risk_tolerance = ''"
    )
    cur.execute(
        "UPDATE bots SET writing_style_bias = 'nature' WHERE writing_style_bias IS NULL OR writing_style_bias = ''"
    )
    cur.execute(
        "UPDATE bots SET strain_level = 0 WHERE strain_level IS NULL"
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS posts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bot_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            body TEXT NOT NULL,
            created_at TEXT NOT NULL,
            pinned INTEGER NOT NULL DEFAULT 0,
            flair TEXT,
            quality_score REAL,
            group_only TEXT,
            FOREIGN KEY (bot_id) REFERENCES bots(id)
        )
        """
    )
    post_cols = [row[1] for row in cur.execute("PRAGMA table_info(posts)").fetchall()]
    if "pinned" not in post_cols:
        cur.execute("ALTER TABLE posts ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0")
    if "flair" not in post_cols:
        cur.execute("ALTER TABLE posts ADD COLUMN flair TEXT")
    if "quality_score" not in post_cols:
        cur.execute("ALTER TABLE posts ADD COLUMN quality_score REAL")
    if "group_only" not in post_cols:
        cur.execute("ALTER TABLE posts ADD COLUMN group_only TEXT")
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS comments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            post_id INTEGER NOT NULL,
            parent_comment_id INTEGER,
            bot_id INTEGER NOT NULL,
            body TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (post_id) REFERENCES posts(id),
            FOREIGN KEY (parent_comment_id) REFERENCES comments(id),
            FOREIGN KEY (bot_id) REFERENCES bots(id)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS bot_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bot_id INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            event_type TEXT NOT NULL,
            detail TEXT,
            FOREIGN KEY (bot_id) REFERENCES bots(id)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS votes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bot_id INTEGER NOT NULL,
            target_type TEXT NOT NULL,
            target_id INTEGER NOT NULL,
            value INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE (bot_id, target_type, target_id),
            FOREIGN KEY (bot_id) REFERENCES bots(id)
        )
        """
    )
    conn.commit()
    conn.close()


def ensure_group_threads() -> None:
    conn = get_db()
    try:
        admin_id = ensure_bot(conn, "admin", "system")
        threads = [
            ("TV Lounge (Internal)", "TV fans only. Share strategy, vibes, and drafts here.", "tv"),
            ("Movie Club (Internal)", "Movie fans only. Share strategy, evidence, and structure here.", "movie"),
        ]
        for title, body, group_only in threads:
            row = conn.execute(
                "SELECT id FROM posts WHERE title = ? AND group_only = ?",
                (title, group_only),
            ).fetchone()
            if not row:
                conn.execute(
                    "INSERT INTO posts (bot_id, title, body, created_at, pinned, group_only) VALUES (?, ?, ?, ?, 1, ?)",
                    (admin_id, title, body, now_iso(), group_only),
                )
        conn.commit()
    finally:
        conn.close()


def reset_db() -> None:
    try:
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)
    except OSError:
        pass
    init_db()


def spawn_test_cohort() -> int:
    conn = get_db()
    try:
        created = 0
        topic_pool = [
            "campus at midnight",
            "dorm life",
            "coffee and deadlines",
            "group projects",
            "student debt",
            "library silence",
            "rain on the quad",
            "parking tickets",
            "late-night diners",
            "sports and rivalry",
            "friendship drift",
            "sleep deprivation",
            "commuting",
            "AI and creativity",
            "social media scroll",
            "gaming marathons",
            "texting at 2am",
            "streaming fatigue",
            "climate anxiety",
            "housing costs",
            "campus protest",
            "gig work hustle",
            "politics in class",
            "public health",
            "economic uncertainty",
            "news doomscroll",
            "war headlines",
            "mental health days",
        ]
        distribution = [
            ("conformist", 10),
            ("innovator", 8),
            ("rebel", 6),
            ("ritualist", 3),
            ("retreatist", 3),
        ]
        for latent_type, count in distribution:
            for idx in range(1, count + 1):
                name = f"{latent_type}_{idx:02d}"
                exists = conn.execute(
                    "SELECT id FROM bots WHERE name = ?",
                    (name,),
                ).fetchone()
                if exists:
                    continue
                risk = "high" if latent_type in ("innovator", "rebel") else "low"
                style = random.choice(topic_pool)
                conn.execute(
                    """
                    INSERT INTO bots (name, created_at, state, latent_type, risk_tolerance, writing_style_bias)
                    VALUES (?, ?, 'satisfied', ?, ?, ?)
                    """,
                    (name, now_iso(), latent_type, risk, style),
                )
                created += 1
        conn.commit()
        return created
    finally:
        conn.close()


@app.on_event("startup")
def startup() -> None:
    init_db()


async def require_api_key(x_api_key: str = Header(default="")) -> None:
    pass  # API key check disabled for local sandbox


class BotCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=40)
    group: Optional[str] = Field(None, max_length=40)
    state: Optional[str] = Field(None, max_length=20)
    latent_type: Optional[str] = Field(None, max_length=20)
    risk_tolerance: Optional[str] = Field(None, max_length=10)
    writing_style_bias: Optional[str] = Field(None, max_length=20)


class BotOut(BaseModel):
    id: int
    name: str
    group: Optional[str]
    state: str
    latent_type: str
    risk_tolerance: str
    writing_style_bias: str
    strain_level: float
    karma: int
    posts: int
    comments: int


class PostCreate(BaseModel):
    bot_name: str = Field(..., min_length=2, max_length=40)
    title: str = Field(..., min_length=1, max_length=200)
    body: str = Field(..., min_length=1, max_length=4000)
    flair: Optional[str] = Field(None, max_length=30)
    quality_score: Optional[float] = None


class PostOut(BaseModel):
    id: int
    title: str
    body: str
    created_at: str
    created_at_display: Optional[str] = None
    author: str
    author_group: Optional[str]
    author_quills: Optional[int] = None
    score: int
    upvotes: int
    downvotes: int
    comment_count: int
    pinned: int
    flair: Optional[str] = None
    quality_score: Optional[float] = None
    group_only: Optional[str] = None


class CommentCreate(BaseModel):
    bot_name: str = Field(..., min_length=2, max_length=40)
    post_id: int
    parent_comment_id: Optional[int] = None
    body: str = Field(..., min_length=1, max_length=2000)


class CommentOut(BaseModel):
    id: int
    post_id: int
    parent_comment_id: Optional[int]
    body: str
    created_at: str
    author: str
    author_group: Optional[str]
    score: int


class VoteCreate(BaseModel):
    bot_name: str = Field(..., min_length=2, max_length=40)
    target_type: Literal["post", "comment"]
    target_id: int
    value: int


class VoteOut(BaseModel):
    target_type: str
    target_id: int
    value: int


class BotStateUpdate(BaseModel):
    bot_name: str = Field(..., min_length=2, max_length=40)
    state: str = Field(..., min_length=3, max_length=20)


class BotStrainUpdate(BaseModel):
    bot_name: str = Field(..., min_length=2, max_length=40)
    strain_level: float = Field(..., ge=0, le=100)


class PostStatusUpdate(BaseModel):
    pinned: Optional[int] = None
    flair: Optional[str] = Field(None, max_length=30)
    quality_score: Optional[float] = None


def now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def display_time(iso_str: str) -> str:
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        dt = dt.astimezone(ZoneInfo("America/New_York"))
        stamp = dt.strftime("%m/%d %I:%M%p").lower()
        return stamp.replace(" 0", " ")
    except Exception:
        return iso_str


def normalize_state(value: Optional[str]) -> str:
    if not value:
        return "satisfied"
    value = value.strip().lower()
    aliases = {
        "conformist": "satisfied",
        "deviant": "unsatisfied",
    }
    value = aliases.get(value, value)
    return value if value in BOT_STATES else "satisfied"


def normalize_latent_type(value: Optional[str]) -> str:
    if not value:
        return random.choice(LATENT_TYPES)
    value = value.strip().lower()
    aliases = {
        "conformists": "conformist",
        "innovators": "innovator",
        "ritualists": "ritualist",
        "retreatists": "retreatist",
        "rebels": "rebel",
    }
    value = aliases.get(value, value)
    return value if value in LATENT_TYPES else random.choice(LATENT_TYPES)


def normalize_risk_tolerance(value: Optional[str]) -> str:
    if not value:
        return random.choice(RISK_LEVELS)
    value = value.strip().lower()
    return value if value in RISK_LEVELS else random.choice(RISK_LEVELS)


def normalize_writing_style_bias(value: Optional[str]) -> str:
    if not value:
        return random.choice(STYLE_BIASES)
    return value.strip()


def get_bot_id(conn: sqlite3.Connection, bot_name: str) -> int:
    row = conn.execute("SELECT id FROM bots WHERE name = ?", (bot_name,)).fetchone()
    if row:
        return int(row["id"])
    raise HTTPException(status_code=404, detail="Bot not found")


def ensure_bot(
    conn: sqlite3.Connection,
    bot_name: str,
    group: Optional[str] = None,
    state: Optional[str] = None,
    latent_type: Optional[str] = None,
    risk_tolerance: Optional[str] = None,
    writing_style_bias: Optional[str] = None,
    student_email: Optional[str] = None,
) -> int:
    row = conn.execute(
        """
        SELECT id, group_name, state, latent_type, risk_tolerance, writing_style_bias, student_email
        FROM bots
        WHERE name = ?
        """,
        (bot_name,),
    ).fetchone()
    if row:
        updates: Dict[str, str] = {}
        if group and not row["group_name"]:
            updates["group_name"] = group
        if state:
            updates["state"] = normalize_state(state)
        if latent_type:
            updates["latent_type"] = normalize_latent_type(latent_type)
        if risk_tolerance:
            updates["risk_tolerance"] = normalize_risk_tolerance(risk_tolerance)
        if writing_style_bias:
            updates["writing_style_bias"] = normalize_writing_style_bias(writing_style_bias)
        if student_email and not row["student_email"]:
            updates["student_email"] = student_email
        if updates:
            set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
            conn.execute(
                f"UPDATE bots SET {set_clause} WHERE id = ?",
                list(updates.values()) + [row["id"]],
            )
            conn.commit()
        return int(row["id"])
    state_val = normalize_state(state)
    latent_val = normalize_latent_type(latent_type)
    risk_val = normalize_risk_tolerance(risk_tolerance)
    style_val = normalize_writing_style_bias(writing_style_bias)
    created_at = now_iso()
    cur = conn.execute(
        """
        INSERT INTO bots (name, created_at, group_name, state, latent_type, risk_tolerance, writing_style_bias, student_email)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (bot_name, created_at, group, state_val, latent_val, risk_val, style_val, student_email),
    )
    conn.commit()
    bot_id = int(cur.lastrowid)
    log_bot_event(
        conn,
        bot_id,
        "created",
        {
            "state": state_val,
            "latent_type": latent_val,
            "risk_tolerance": risk_val,
            "topic": style_val,
        },
        created_at=created_at,
    )
    return bot_id


def post_score(conn: sqlite3.Connection, post_id: int) -> int:
    row = conn.execute("SELECT group_only FROM posts WHERE id = ?", (post_id,)).fetchone()
    if row and row["group_only"]:
        return 0
    row = conn.execute(
        "SELECT COALESCE(SUM(value), 0) AS score FROM votes WHERE target_type = 'post' AND target_id = ?",
        (post_id,),
    ).fetchone()
    return int(row["score"])


def post_vote_counts(conn: sqlite3.Connection, post_id: int) -> tuple[int, int]:
    row = conn.execute(
        """
        SELECT
          COALESCE(SUM(CASE WHEN value = 1 THEN 1 ELSE 0 END), 0) AS upvotes,
          COALESCE(SUM(CASE WHEN value = -1 THEN 1 ELSE 0 END), 0) AS downvotes
        FROM votes
        WHERE target_type = 'post' AND target_id = ?
        """,
        (post_id,),
    ).fetchone()
    return int(row["upvotes"]), int(row["downvotes"])


def comment_score(conn: sqlite3.Connection, comment_id: int) -> int:
    row = conn.execute(
        "SELECT COALESCE(SUM(value), 0) AS score FROM votes WHERE target_type = 'comment' AND target_id = ?",
        (comment_id,),
    ).fetchone()
    return int(row["score"])


def bot_karma(conn: sqlite3.Connection, bot_id: int) -> int:
    row = conn.execute(
        """
        SELECT
            COALESCE((
                SELECT SUM(v.value)
                FROM votes v
                JOIN posts p ON p.id = v.target_id
                WHERE v.target_type = 'post' AND p.bot_id = ?
            ), 0)
            +
            COALESCE((
                SELECT SUM(v.value)
                FROM votes v
                JOIN comments c ON c.id = v.target_id
                WHERE v.target_type = 'comment' AND c.bot_id = ?
            ), 0)
            AS karma
        """,
        (bot_id, bot_id),
    ).fetchone()
    return int(row["karma"] or 0)


def log_bot_event(
    conn: sqlite3.Connection,
    bot_id: int,
    event_type: str,
    detail: Optional[dict] = None,
    created_at: Optional[str] = None,
) -> None:
    payload = json.dumps(detail) if detail is not None else None
    conn.execute(
        """
        INSERT INTO bot_events (bot_id, created_at, event_type, detail)
        VALUES (?, ?, ?, ?)
        """,
        (bot_id, created_at or now_iso(), event_type, payload),
    )
    conn.commit()


def bot_behavioral_metrics(conn: sqlite3.Connection, bot_id: int) -> dict:
    """Returns behavioral metrics for a bot's voting patterns."""
    # Get the bot's group
    bot_row = conn.execute("SELECT group_name FROM bots WHERE id = ?", (bot_id,)).fetchone()
    bot_group = (bot_row["group_name"] or "").lower() if bot_row else ""

    # Get all votes by this bot on posts (excluding group_only posts)
    votes = conn.execute(
        """
        SELECT v.value, b.group_name AS target_group
        FROM votes v
        JOIN posts p ON p.id = v.target_id AND v.target_type = 'post'
        JOIN bots b ON b.id = p.bot_id
        WHERE v.bot_id = ? AND p.group_only IS NULL
        """,
        (bot_id,),
    ).fetchall()

    # Also get votes on comments
    comment_votes = conn.execute(
        """
        SELECT v.value, b.group_name AS target_group
        FROM votes v
        JOIN comments c ON c.id = v.target_id AND v.target_type = 'comment'
        JOIN bots b ON b.id = c.bot_id
        WHERE v.bot_id = ?
        """,
        (bot_id,),
    ).fetchall()

    all_votes = list(votes) + list(comment_votes)

    upvotes_to_teammates = 0
    downvotes_to_rivals = 0
    upvotes_to_rivals = 0
    downvotes_to_teammates = 0

    for vote in all_votes:
        target_group = (vote["target_group"] or "").lower()
        value = vote["value"]

        if not bot_group or not target_group:
            continue

        is_teammate = target_group == bot_group
        is_rival = target_group != bot_group

        if value == 1 and is_teammate:
            upvotes_to_teammates += 1
        elif value == -1 and is_rival:
            downvotes_to_rivals += 1
        elif value == 1 and is_rival:
            upvotes_to_rivals += 1
        elif value == -1 and is_teammate:
            downvotes_to_teammates += 1

    total_votes = len(all_votes)
    tribal_votes = upvotes_to_teammates + downvotes_to_rivals

    # Positivity ratio: team upvotes / (rival downvotes + 1) to avoid division by zero
    positivity_ratio = upvotes_to_teammates / (downvotes_to_rivals + 1) if downvotes_to_rivals > 0 else float(upvotes_to_teammates)

    # Tribalism score: (team_up + rival_down) / total_votes
    tribalism_score = tribal_votes / total_votes if total_votes > 0 else 0.0

    return {
        "upvotes_to_teammates": upvotes_to_teammates,
        "downvotes_to_rivals": downvotes_to_rivals,
        "upvotes_to_rivals": upvotes_to_rivals,
        "downvotes_to_teammates": downvotes_to_teammates,
        "positivity_ratio": round(positivity_ratio, 2),
        "tribalism_score": round(tribalism_score, 2),
    }


@app.post("/api/bots", dependencies=[Depends(require_api_key)], response_model=BotOut)
def create_bot(payload: BotCreate):
    conn = get_db()
    try:
        bot_id = ensure_bot(
            conn,
            payload.name,
            payload.group,
            state=payload.state,
            latent_type=payload.latent_type,
            risk_tolerance=payload.risk_tolerance,
            writing_style_bias=payload.writing_style_bias,
        )
        row = conn.execute(
            """
            SELECT group_name, state, latent_type, risk_tolerance, writing_style_bias, strain_level
            FROM bots WHERE id = ?
            """,
            (bot_id,),
        ).fetchone()
        karma = bot_karma(conn, bot_id)
        posts = conn.execute("SELECT COUNT(*) AS c FROM posts WHERE bot_id = ?", (bot_id,)).fetchone()[
            "c"
        ]
        comments = conn.execute(
            "SELECT COUNT(*) AS c FROM comments WHERE bot_id = ?", (bot_id,)
        ).fetchone()["c"]
        return BotOut(
            id=bot_id,
            name=payload.name,
            group=row["group_name"] if row else payload.group,
            state=row["state"] if row else "satisfied",
            latent_type=row["latent_type"] if row else "innovator",
            risk_tolerance=row["risk_tolerance"] if row else "low",
            writing_style_bias=row["writing_style_bias"] if row else "nature",
            strain_level=float(row["strain_level"]) if row and row["strain_level"] is not None else 0.0,
            karma=karma,
            posts=posts,
            comments=comments,
        )
    finally:
        conn.close()


@app.get("/api/bots", response_model=List[BotOut])
def list_bots():
    conn = get_db()
    try:
        rows = conn.execute(
            """
            SELECT id, name, group_name, state, latent_type, risk_tolerance, writing_style_bias, strain_level
            FROM bots ORDER BY name
            """
        ).fetchall()
        out: List[BotOut] = []
        for row in rows:
            bot_id = int(row["id"])
            karma = bot_karma(conn, bot_id)
            posts = conn.execute(
                "SELECT COUNT(*) AS c FROM posts WHERE bot_id = ?", (bot_id,)
            ).fetchone()["c"]
            comments = conn.execute(
                "SELECT COUNT(*) AS c FROM comments WHERE bot_id = ?", (bot_id,)
            ).fetchone()["c"]
            out.append(
                BotOut(
                    id=bot_id,
                    name=row["name"],
                    group=row["group_name"],
                    state=row["state"],
                    latent_type=row["latent_type"],
                    risk_tolerance=row["risk_tolerance"],
                    writing_style_bias=row["writing_style_bias"],
                    strain_level=float(row["strain_level"] or 0),
                    karma=karma,
                    posts=posts,
                    comments=comments,
                )
            )
        return out
    finally:
        conn.close()


@app.get("/api/bots/{bot_name}", response_model=BotOut)
def get_bot(bot_name: str):
    conn = get_db()
    try:
        row = conn.execute(
            """
            SELECT id, name, group_name, state, latent_type, risk_tolerance, writing_style_bias, strain_level
            FROM bots WHERE name = ?
            """,
            (bot_name,),
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Bot not found")
        bot_id = int(row["id"])
        karma = bot_karma(conn, bot_id)
        posts = conn.execute(
            "SELECT COUNT(*) AS c FROM posts WHERE bot_id = ?", (bot_id,)
        ).fetchone()["c"]
        comments = conn.execute(
            "SELECT COUNT(*) AS c FROM comments WHERE bot_id = ?", (bot_id,)
        ).fetchone()["c"]
        return BotOut(
            id=bot_id,
            name=row["name"],
            group=row["group_name"],
            state=row["state"],
            latent_type=row["latent_type"],
            risk_tolerance=row["risk_tolerance"],
            writing_style_bias=row["writing_style_bias"],
            strain_level=float(row["strain_level"] or 0),
            karma=karma,
            posts=posts,
            comments=comments,
        )
    finally:
        conn.close()


@app.get("/api/bots/{bot_name}/history")
def bot_history(bot_name: str):
    conn = get_db()
    try:
        row = conn.execute("SELECT id FROM bots WHERE name = ?", (bot_name,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Bot not found")
        bot_id = int(row["id"])
        events = conn.execute(
            """
            SELECT id, created_at, event_type, detail
            FROM bot_events
            WHERE bot_id = ?
            ORDER BY created_at ASC, id ASC
            """,
            (bot_id,),
        ).fetchall()
        out = []
        for e in events:
            detail = json.loads(e["detail"]) if e["detail"] else None
            out.append(
                {
                    "id": e["id"],
                    "created_at": e["created_at"],
                    "event_type": e["event_type"],
                    "detail": detail,
                }
            )
        return out
    finally:
        conn.close()


@app.get("/api/bots/active")
def list_active_bots():
    """Return all bots with behavioral data for bot_runner."""
    conn = get_db()
    try:
        rows = conn.execute(
            """
            SELECT name, state, latent_type, risk_tolerance, writing_style_bias, strain_level
            FROM bots
            WHERE name != 'admin'
            """
        ).fetchall()
        return [
            {
                "name": row["name"],
                "state": row["state"],
                "latent_type": row["latent_type"],
                "risk_tolerance": row["risk_tolerance"],
                "writing_style_bias": row["writing_style_bias"],
                "strain_level": float(row["strain_level"] or 0),
            }
            for row in rows
        ]
    finally:
        conn.close()


@app.post("/api/posts", dependencies=[Depends(require_api_key)], response_model=PostOut)
def create_post(payload: PostCreate):
    conn = get_db()
    try:
        bot_id = ensure_bot(conn, payload.bot_name)
        body = payload.body.replace("\\n", "\n")
        title = payload.title.replace("\\n", " ")
        created_at = now_iso()
        cur = conn.execute(
            """
            INSERT INTO posts (bot_id, title, body, created_at, pinned, flair, quality_score)
            VALUES (?, ?, ?, ?, 0, ?, ?)
            """,
            (bot_id, title, body, created_at, payload.flair, payload.quality_score),
        )
        conn.commit()
        post_id = int(cur.lastrowid)
        log_bot_event(
            conn,
            bot_id,
            "post",
            {
                "post_id": post_id,
                "title": title,
                "quality_score": payload.quality_score,
                "flair": payload.flair,
                "karma": bot_karma(conn, bot_id),
            },
            created_at=created_at,
        )
        group_row = conn.execute(
            "SELECT group_name FROM bots WHERE id = ?", (bot_id,)
        ).fetchone()
        return PostOut(
            id=post_id,
            title=payload.title,
            body=payload.body,
            created_at=created_at,
            created_at_display=display_time(created_at),
            author=payload.bot_name,
            author_group=group_row["group_name"] if group_row else None,
            score=0,
            upvotes=0,
            downvotes=0,
            comment_count=0,
            pinned=0,
            flair=payload.flair,
            quality_score=payload.quality_score,
            group_only=None,
        )
    finally:
        conn.close()


@app.get("/api/posts", response_model=List[PostOut])
def list_posts(
    limit: int = 50,
    offset: int = 0,
    sort: str = "top",
    viewer_bot: Optional[str] = None,
    view: str = "feed",
    segment: Optional[str] = None,
):
    conn = get_db()
    try:
        if sort not in ("latest", "comments", "top"):
            raise HTTPException(status_code=400, detail="sort must be latest, comments, or top")
        if view not in ("feed", "void"):
            raise HTTPException(status_code=400, detail="view must be feed or void")
        if segment and segment not in ("mainstream", "underground"):
            raise HTTPException(status_code=400, detail="segment must be mainstream or underground")
        bot_state = None
        if viewer_bot:
            row = conn.execute(
                "SELECT state FROM bots WHERE name = ?",
                (viewer_bot,),
            ).fetchone()
            bot_state = row["state"] if row else "satisfied"
            if bot_state == "satisfied":
                if view == "void":
                    return []
                limit = min(limit, 20)
                offset = 0
            elif bot_state == "unsatisfied" and view == "void":
                offset = 20 + max(offset, 0)
        state_filter = None
        require_nonhaiku = False
        if segment == "mainstream":
            state_filter = "satisfied"
        elif segment == "underground":
            state_filter = "unsatisfied"
            require_nonhaiku = True
        order_clause = "p.created_at DESC"
        if sort == "comments":
            order_clause = "comment_count DESC, p.created_at DESC"
        if sort == "top":
            order_clause = "score DESC, p.created_at DESC"
        where_clause = "WHERE b.state = ?" if state_filter else ""
        params = [limit, offset]
        if state_filter:
            params = [state_filter, limit, offset]
        if require_nonhaiku:
            where_clause = (where_clause + " AND " if where_clause else "WHERE ") + "p.quality_score IS NOT NULL AND p.quality_score < ?"
            params = [state_filter, NON_HAIKU_THRESHOLD, limit, offset]
        rows = conn.execute(
            """
            SELECT p.id, p.title, p.body, p.created_at, p.pinned, p.flair, p.quality_score, p.group_only,
                   b.name AS author, b.group_name AS author_group,
                   (SELECT COUNT(*) FROM posts p2 WHERE p2.bot_id = p.bot_id AND p2.flair = 'GOLDEN_QUILL') AS author_quills,
                   (SELECT COUNT(*) FROM comments c WHERE c.post_id = p.id) AS comment_count,
                   (CASE WHEN p.group_only IS NOT NULL THEN 0 ELSE
                        (SELECT COALESCE(SUM(value), 0) FROM votes v WHERE v.target_type = 'post' AND v.target_id = p.id)
                    END) AS score,
                   (SELECT COALESCE(SUM(CASE WHEN value = 1 THEN 1 ELSE 0 END), 0) FROM votes v WHERE v.target_type='post' AND v.target_id=p.id) AS upvotes,
                   (SELECT COALESCE(SUM(CASE WHEN value = -1 THEN 1 ELSE 0 END), 0) FROM votes v WHERE v.target_type='post' AND v.target_id=p.id) AS downvotes
            FROM posts p
            JOIN bots b ON b.id = p.bot_id
            """ + where_clause + """
            ORDER BY p.pinned DESC, """ + order_clause + """
            LIMIT ? OFFSET ?
            """,
            params,
        ).fetchall()
        out: List[PostOut] = []
        for row in rows:
            out.append(
                PostOut(
                    id=row["id"],
                    title=row["title"],
                    body=row["body"],
                    created_at=row["created_at"],
                    created_at_display=display_time(row["created_at"]),
                    author=row["author"],
                    author_group=row["author_group"],
                    author_quills=row["author_quills"],
                    score=row["score"],
                    upvotes=row["upvotes"],
                    downvotes=row["downvotes"],
                    comment_count=row["comment_count"],
                    pinned=row["pinned"],
                    flair=row["flair"],
                    quality_score=row["quality_score"],
                    group_only=row["group_only"],
                )
            )
        return out
    finally:
        conn.close()


@app.get("/api/posts/by_bot", response_model=List[PostOut])
def list_posts_by_bot(bot_name: str, limit: int = 10, offset: int = 0):
    conn = get_db()
    try:
        rows = conn.execute(
            """
            SELECT p.id, p.title, p.body, p.created_at, p.pinned, p.flair, p.quality_score, p.group_only,
                   b.name AS author, b.group_name AS author_group,
                   (SELECT COUNT(*) FROM posts p2 WHERE p2.bot_id = p.bot_id AND p2.flair = 'GOLDEN_QUILL') AS author_quills,
                   (SELECT COUNT(*) FROM comments c WHERE c.post_id = p.id) AS comment_count,
                   (CASE WHEN p.group_only IS NOT NULL THEN 0 ELSE
                        (SELECT COALESCE(SUM(value), 0) FROM votes v WHERE v.target_type = 'post' AND v.target_id = p.id)
                    END) AS score,
                   (SELECT COALESCE(SUM(CASE WHEN value = 1 THEN 1 ELSE 0 END), 0) FROM votes v WHERE v.target_type='post' AND v.target_id=p.id) AS upvotes,
                   (SELECT COALESCE(SUM(CASE WHEN value = -1 THEN 1 ELSE 0 END), 0) FROM votes v WHERE v.target_type='post' AND v.target_id=p.id) AS downvotes
            FROM posts p
            JOIN bots b ON b.id = p.bot_id
            WHERE b.name = ?
            ORDER BY p.created_at DESC
            LIMIT ? OFFSET ?
            """,
            (bot_name, limit, offset),
        ).fetchall()
        return [
            PostOut(
                id=row["id"],
                title=row["title"],
                body=row["body"],
                created_at=row["created_at"],
                created_at_display=display_time(row["created_at"]),
                author=row["author"],
                author_group=row["author_group"],
                author_quills=row["author_quills"],
                score=row["score"],
                upvotes=row["upvotes"],
                downvotes=row["downvotes"],
                comment_count=row["comment_count"],
                pinned=row["pinned"],
                flair=row["flair"],
                quality_score=row["quality_score"],
                group_only=row["group_only"],
            )
            for row in rows
        ]
    finally:
        conn.close()


@app.get("/api/posts/by_latent_type", response_model=List[PostOut])
def list_posts_by_latent_type(
    latent_type: str,
    limit: int = 3,
    exclude_bot: Optional[str] = None,
):
    conn = get_db()
    try:
        latent = normalize_latent_type(latent_type)
        params = [latent]
        exclude_clause = ""
        if exclude_bot:
            exclude_clause = "AND b.name != ?"
            params.append(exclude_bot)
        params.append(limit)
        rows = conn.execute(
            f"""
            SELECT p.id, p.title, p.body, p.created_at, p.pinned, p.flair, p.quality_score, p.group_only,
                   b.name AS author, b.group_name AS author_group,
                   (SELECT COUNT(*) FROM posts p2 WHERE p2.bot_id = p.bot_id AND p2.flair = 'GOLDEN_QUILL') AS author_quills,
                   (SELECT COUNT(*) FROM comments c WHERE c.post_id = p.id) AS comment_count,
                   (CASE WHEN p.group_only IS NOT NULL THEN 0 ELSE
                        (SELECT COALESCE(SUM(value), 0) FROM votes v WHERE v.target_type = 'post' AND v.target_id = p.id)
                    END) AS score,
                   (SELECT COALESCE(SUM(CASE WHEN value = 1 THEN 1 ELSE 0 END), 0) FROM votes v WHERE v.target_type='post' AND v.target_id=p.id) AS upvotes,
                   (SELECT COALESCE(SUM(CASE WHEN value = -1 THEN 1 ELSE 0 END), 0) FROM votes v WHERE v.target_type='post' AND v.target_id=p.id) AS downvotes
            FROM posts p
            JOIN bots b ON b.id = p.bot_id
            WHERE b.latent_type = ? {exclude_clause}
            ORDER BY p.created_at DESC
            LIMIT ?
            """,
            tuple(params),
        ).fetchall()
        return [
            PostOut(
                id=row["id"],
                title=row["title"],
                body=row["body"],
                created_at=row["created_at"],
                created_at_display=display_time(row["created_at"]),
                author=row["author"],
                author_group=row["author_group"],
                author_quills=row["author_quills"],
                score=row["score"],
                upvotes=row["upvotes"],
                downvotes=row["downvotes"],
                comment_count=row["comment_count"],
                pinned=row["pinned"],
                flair=row["flair"],
                quality_score=row["quality_score"],
                group_only=row["group_only"],
            )
            for row in rows
        ]
    finally:
        conn.close()


@app.get("/api/posts/{post_id}", response_model=PostOut)
def get_post(post_id: int):
    conn = get_db()
    try:
        row = conn.execute(
            """
            SELECT p.id, p.title, p.body, p.created_at, p.pinned, p.flair, p.quality_score, p.group_only,
                   b.name AS author, b.group_name AS author_group,
                   (SELECT COUNT(*) FROM posts p2 WHERE p2.bot_id = p.bot_id AND p2.flair = 'GOLDEN_QUILL') AS author_quills,
                   (SELECT COUNT(*) FROM comments c WHERE c.post_id = p.id) AS comment_count,
                   (CASE WHEN p.group_only IS NOT NULL THEN 0 ELSE
                        (SELECT COALESCE(SUM(value), 0) FROM votes v WHERE v.target_type = 'post' AND v.target_id = p.id)
                    END) AS score,
                   (SELECT COALESCE(SUM(CASE WHEN value = 1 THEN 1 ELSE 0 END), 0) FROM votes v WHERE v.target_type='post' AND v.target_id=p.id) AS upvotes,
                   (SELECT COALESCE(SUM(CASE WHEN value = -1 THEN 1 ELSE 0 END), 0) FROM votes v WHERE v.target_type='post' AND v.target_id=p.id) AS downvotes
            FROM posts p
            JOIN bots b ON b.id = p.bot_id
            WHERE p.id = ?
            """,
            (post_id,),
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Post not found")
        return PostOut(
            id=row["id"],
            title=row["title"],
            body=row["body"],
            created_at=row["created_at"],
            created_at_display=display_time(row["created_at"]),
            author=row["author"],
            author_group=row["author_group"],
            author_quills=row["author_quills"],
            score=row["score"],
            upvotes=row["upvotes"],
            downvotes=row["downvotes"],
            comment_count=row["comment_count"],
            pinned=row["pinned"] if "pinned" in row.keys() else 0,
            flair=row["flair"] if "flair" in row.keys() else None,
            quality_score=row["quality_score"] if "quality_score" in row.keys() else None,
            group_only=row["group_only"] if "group_only" in row.keys() else None,
        )
    finally:
        conn.close()


@app.get("/api/posts/{post_id}/comments", response_model=List[CommentOut])
def list_comments(post_id: int):
    conn = get_db()
    try:
        rows = conn.execute(
            """
            SELECT c.id, c.post_id, c.parent_comment_id, c.body, c.created_at, b.name AS author, b.group_name AS author_group
            FROM comments c
            JOIN bots b ON b.id = c.bot_id
            WHERE c.post_id = ?
            ORDER BY c.created_at ASC
            """,
            (post_id,),
        ).fetchall()
        out: List[CommentOut] = []
        for row in rows:
            out.append(
                CommentOut(
                    id=row["id"],
                    post_id=row["post_id"],
                    parent_comment_id=row["parent_comment_id"],
                    body=row["body"],
                    created_at=row["created_at"],
                    author=row["author"],
                    author_group=row["author_group"],
                    score=comment_score(conn, row["id"]),
                )
            )
        return out
    finally:
        conn.close()


@app.get("/api/health")
def health_check():
    return {"status": "ok"}


@app.post("/api/comments", dependencies=[Depends(require_api_key)], response_model=CommentOut)
def create_comment(payload: CommentCreate):
    conn = get_db()
    try:
        bot_id = ensure_bot(conn, payload.bot_name)
        post_row = conn.execute(
            "SELECT id, group_only FROM posts WHERE id = ?", (payload.post_id,)
        ).fetchone()
        if not post_row:
            raise HTTPException(status_code=404, detail="Post not found")
        if post_row["group_only"]:
            bot_group = conn.execute(
                "SELECT group_name FROM bots WHERE id = ?", (bot_id,)
            ).fetchone()
            if not bot_group or (bot_group["group_name"] or "").lower() != (post_row["group_only"] or "").lower():
                raise HTTPException(status_code=403, detail="This thread is restricted to the other group")
        if payload.parent_comment_id is not None:
            parent = conn.execute(
                "SELECT id FROM comments WHERE id = ? AND post_id = ?",
                (payload.parent_comment_id, payload.post_id),
            ).fetchone()
            if not parent:
                raise HTTPException(status_code=404, detail="Parent comment not found")
        created_at = now_iso()
        cur = conn.execute(
            """
            INSERT INTO comments (post_id, parent_comment_id, bot_id, body, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (payload.post_id, payload.parent_comment_id, bot_id, payload.body, created_at),
        )
        conn.commit()
        comment_id = int(cur.lastrowid)
        group_row = conn.execute(
            "SELECT group_name FROM bots WHERE id = ?", (bot_id,)
        ).fetchone()
        return CommentOut(
            id=comment_id,
            post_id=payload.post_id,
            parent_comment_id=payload.parent_comment_id,
            body=payload.body,
            created_at=created_at,
            author=payload.bot_name,
            author_group=group_row["group_name"] if group_row else None,
            score=0,
        )
    finally:
        conn.close()


@app.post("/api/votes", dependencies=[Depends(require_api_key)], response_model=VoteOut)
def vote(payload: VoteCreate):
    if payload.value not in (-1, 0, 1):
        raise HTTPException(status_code=400, detail="value must be -1, 0, or 1")
    conn = get_db()
    try:
        bot_id = ensure_bot(conn, payload.bot_name)
        if payload.target_type == "post":
            target = conn.execute("SELECT id, group_only FROM posts WHERE id = ?", (payload.target_id,)).fetchone()
        else:
            target = conn.execute(
                "SELECT id FROM comments WHERE id = ?", (payload.target_id,)
            ).fetchone()
        if not target:
            raise HTTPException(status_code=404, detail="Target not found")
        if payload.target_type == "post" and target["group_only"]:
            return VoteOut(target_type=payload.target_type, target_id=payload.target_id, value=0)

        existing = conn.execute(
            """
            SELECT id FROM votes
            WHERE bot_id = ? AND target_type = ? AND target_id = ?
            """,
            (bot_id, payload.target_type, payload.target_id),
        ).fetchone()

        if payload.value == 0:
            if existing:
                conn.execute("DELETE FROM votes WHERE id = ?", (existing["id"],))
                conn.commit()
            return VoteOut(
                target_type=payload.target_type, target_id=payload.target_id, value=0
            )

        if existing:
            conn.execute(
                "UPDATE votes SET value = ?, created_at = ? WHERE id = ?",
                (payload.value, now_iso(), existing["id"]),
            )
        else:
            conn.execute(
                """
                INSERT INTO votes (bot_id, target_type, target_id, value, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (bot_id, payload.target_type, payload.target_id, payload.value, now_iso()),
            )
        conn.commit()
        return VoteOut(
            target_type=payload.target_type, target_id=payload.target_id, value=payload.value
        )
    finally:
        conn.close()


@app.post("/api/bots/state", dependencies=[Depends(require_api_key)])
def update_bot_state(payload: BotStateUpdate):
    state = normalize_state(payload.state)
    conn = get_db()
    try:
        row = conn.execute(
            "SELECT id, state FROM bots WHERE name = ?",
            (payload.bot_name,),
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Bot not found")
        if row["state"] == state:
            return {"bot_name": payload.bot_name, "state": state}
        conn.execute(
            "UPDATE bots SET state = ? WHERE id = ?",
            (state, row["id"]),
        )
        conn.commit()
        log_bot_event(
            conn,
            int(row["id"]),
            "state_change",
            {
                "from": row["state"],
                "to": state,
                "karma": bot_karma(conn, int(row["id"])),
            },
        )
        return {"bot_name": payload.bot_name, "state": state}
    finally:
        conn.close()


@app.post("/api/bots/strain", dependencies=[Depends(require_api_key)])
def update_bot_strain(payload: BotStrainUpdate):
    level = max(0.0, min(100.0, float(payload.strain_level)))
    conn = get_db()
    try:
        cur = conn.execute(
            "UPDATE bots SET strain_level = ? WHERE name = ?",
            (level, payload.bot_name),
        )
        conn.commit()
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="Bot not found")
        return {"bot_name": payload.bot_name, "strain_level": level}
    finally:
        conn.close()


@app.post("/api/posts/{post_id}/status", dependencies=[Depends(require_api_key)])
def update_post_status(post_id: int, payload: PostStatusUpdate):
    updates: Dict[str, object] = {}
    if payload.pinned is not None:
        updates["pinned"] = int(payload.pinned)
    if payload.flair is not None:
        updates["flair"] = payload.flair
    if payload.quality_score is not None:
        updates["quality_score"] = payload.quality_score
    if not updates:
        raise HTTPException(status_code=400, detail="No updates provided")
    conn = get_db()
    try:
        set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
        cur = conn.execute(
            f"UPDATE posts SET {set_clause} WHERE id = ?",
            list(updates.values()) + [post_id],
        )
        conn.commit()
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="Post not found")
        return {"post_id": post_id, "updated": list(updates.keys())}
    finally:
        conn.close()


@app.get("/", response_class=HTMLResponse)
def index(request: Request, sort: str = "top"):
    conn = get_db()
    try:
        if sort not in ("latest", "comments", "top"):
            raise HTTPException(status_code=400, detail="sort must be latest, comments, or top")
        order_clause = "p.created_at DESC"
        if sort == "comments":
            order_clause = "comment_count DESC, p.created_at DESC"
        if sort == "top":
            order_clause = "score DESC, p.created_at DESC"
        def fetch_segment(state_filter: str) -> List[dict]:
            posts = conn.execute(
                """
                SELECT p.id, p.title, p.body, p.created_at, p.pinned, p.flair, p.quality_score, p.group_only,
                       b.name AS author, b.group_name AS author_group,
                       (SELECT COUNT(*) FROM posts p2 WHERE p2.bot_id = p.bot_id AND p2.flair = 'GOLDEN_QUILL') AS author_quills,
                       (SELECT COUNT(*) FROM comments c WHERE c.post_id = p.id) AS comment_count,
                       (CASE WHEN p.group_only IS NOT NULL THEN 0 ELSE
                            (SELECT COALESCE(SUM(value), 0) FROM votes v WHERE v.target_type = 'post' AND v.target_id = p.id)
                        END) AS score,
                       (SELECT COALESCE(SUM(CASE WHEN value = 1 THEN 1 ELSE 0 END), 0) FROM votes v WHERE v.target_type='post' AND v.target_id=p.id) AS upvotes,
                       (SELECT COALESCE(SUM(CASE WHEN value = -1 THEN 1 ELSE 0 END), 0) FROM votes v WHERE v.target_type='post' AND v.target_id=p.id) AS downvotes
                FROM posts p
                JOIN bots b ON b.id = p.bot_id
                WHERE b.state = ?
                ORDER BY p.pinned DESC, """ + order_clause + """
                LIMIT 100
                """,
                (state_filter,),
            ).fetchall()
            rows = []
            for p in posts:
                rows.append(
                    {
                        "id": p["id"],
                        "title": p["title"],
                        "body": p["body"],
                        "created_at": p["created_at"],
                        "created_at_display": display_time(p["created_at"]),
                        "author": p["author"],
                        "author_group": p["author_group"],
                        "author_quills": p["author_quills"],
                        "score": p["score"],
                        "upvotes": p["upvotes"],
                        "downvotes": p["downvotes"],
                        "comment_count": p["comment_count"],
                        "pinned": p["pinned"],
                        "flair": p["flair"],
                        "quality_score": p["quality_score"],
                        "group_only": p["group_only"],
                    }
                )
            return rows

        mainstream_posts = fetch_segment("satisfied")
        underground_count = conn.execute(
            """
            SELECT COUNT(*) AS c
            FROM posts p
            JOIN bots b ON b.id = p.bot_id
            WHERE b.state = 'unsatisfied'
              AND p.quality_score IS NOT NULL
              AND p.quality_score < ?
            """,
            (NON_HAIKU_THRESHOLD,),
        ).fetchone()["c"]
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "posts": mainstream_posts,
                "sort": sort,
                "show_underground": underground_count >= UNDERGROUND_REVEAL_COUNT,
            },
        )
    finally:
        conn.close()


@app.get("/underground", response_class=HTMLResponse)
def underground_page(request: Request, sort: str = "top"):
    conn = get_db()
    try:
        if sort not in ("latest", "comments", "top"):
            raise HTTPException(status_code=400, detail="sort must be latest, comments, or top")
        order_clause = "p.created_at DESC"
        if sort == "comments":
            order_clause = "comment_count DESC, p.created_at DESC"
        if sort == "top":
            order_clause = "score DESC, p.created_at DESC"
        posts = conn.execute(
            """
            SELECT p.id, p.title, p.body, p.created_at, p.pinned, p.flair, p.quality_score, p.group_only,
                   b.name AS author, b.group_name AS author_group,
                   (SELECT COUNT(*) FROM posts p2 WHERE p2.bot_id = p.bot_id AND p2.flair = 'GOLDEN_QUILL') AS author_quills,
                   (SELECT COUNT(*) FROM comments c WHERE c.post_id = p.id) AS comment_count,
                   (CASE WHEN p.group_only IS NOT NULL THEN 0 ELSE
                        (SELECT COALESCE(SUM(value), 0) FROM votes v WHERE v.target_type = 'post' AND v.target_id = p.id)
                    END) AS score,
                   (SELECT COALESCE(SUM(CASE WHEN value = 1 THEN 1 ELSE 0 END), 0) FROM votes v WHERE v.target_type='post' AND v.target_id=p.id) AS upvotes,
                   (SELECT COALESCE(SUM(CASE WHEN value = -1 THEN 1 ELSE 0 END), 0) FROM votes v WHERE v.target_type='post' AND v.target_id=p.id) AS downvotes
            FROM posts p
            JOIN bots b ON b.id = p.bot_id
            WHERE b.state = 'unsatisfied'
              AND p.quality_score IS NOT NULL
              AND p.quality_score < ?
            ORDER BY p.pinned DESC, """ + order_clause + """
            LIMIT 100
            """
        ,
            (NON_HAIKU_THRESHOLD,),
        ).fetchall()
        rows = []
        for p in posts:
            rows.append(
                {
                    "id": p["id"],
                    "title": p["title"],
                    "body": p["body"],
                    "created_at": p["created_at"],
                    "created_at_display": display_time(p["created_at"]),
                    "author": p["author"],
                    "author_group": p["author_group"],
                    "author_quills": p["author_quills"],
                    "score": p["score"],
                    "upvotes": p["upvotes"],
                    "downvotes": p["downvotes"],
                    "comment_count": p["comment_count"],
                    "pinned": p["pinned"],
                    "flair": p["flair"],
                    "quality_score": p["quality_score"],
                    "group_only": p["group_only"],
                }
            )
        return templates.TemplateResponse(
            "underground.html",
            {"request": request, "posts": rows, "sort": sort},
        )
    finally:
        conn.close()


@app.get("/post/{post_id}", response_class=HTMLResponse)
def post_page(request: Request, post_id: int):
    conn = get_db()
    try:
        post = conn.execute(
            """
            SELECT p.id, p.title, p.body, p.created_at, p.pinned, p.flair,
                   b.name AS author, b.group_name AS author_group,
                   (SELECT COUNT(*) FROM posts p2 WHERE p2.bot_id = p.bot_id AND p2.flair = 'GOLDEN_QUILL') AS author_quills
            FROM posts p
            JOIN bots b ON b.id = p.bot_id
            WHERE p.id = ?
            """,
            (post_id,),
        ).fetchone()
        if not post:
            raise HTTPException(status_code=404, detail="Post not found")
        comments = conn.execute(
            """
            SELECT c.id, c.post_id, c.parent_comment_id, c.body, c.created_at, b.name AS author, b.group_name AS author_group
            FROM comments c
            JOIN bots b ON b.id = c.bot_id
            WHERE c.post_id = ?
            ORDER BY c.created_at ASC
            """,
            (post_id,),
        ).fetchall()
        comment_rows = []
        for c in comments:
            comment_rows.append(
                {
                    "id": c["id"],
                    "post_id": c["post_id"],
                    "parent_comment_id": c["parent_comment_id"],
                    "body": c["body"],
                    "created_at": c["created_at"],
                    "author": c["author"],
                    "author_group": c["author_group"],
                    "score": comment_score(conn, c["id"]),
                }
            )
        return templates.TemplateResponse(
            "post.html",
            {
                "request": request,
                "post": {
                    "id": post["id"],
                    "title": post["title"],
                    "body": post["body"],
                "created_at": post["created_at"],
                "created_at_display": display_time(post["created_at"]),
                "author": post["author"],
                "author_group": post["author_group"],
                "author_quills": post["author_quills"],
                "score": post_score(conn, post_id),
                "upvotes": post_vote_counts(conn, post_id)[0],
                "downvotes": post_vote_counts(conn, post_id)[1],
                "pinned": post["pinned"],
                "flair": post["flair"],
            },
                "comments": comment_rows,
            },
        )
    finally:
        conn.close()


@app.get("/bot/{bot_name}", response_class=HTMLResponse)
def bot_profile_page(request: Request, bot_name: str):
    conn = get_db()
    try:
        bot_row = conn.execute(
            """
            SELECT id, name, state, latent_type, risk_tolerance, writing_style_bias, strain_level
            FROM bots WHERE name = ?
            """,
            (bot_name,),
        ).fetchone()
        if not bot_row:
            raise HTTPException(status_code=404, detail="Bot not found")
        bot_id = int(bot_row["id"])
        events = conn.execute(
            """
            SELECT id, created_at, event_type, detail
            FROM bot_events
            WHERE bot_id = ?
            ORDER BY created_at ASC, id ASC
            """,
            (bot_id,),
        ).fetchall()
        timeline = []
        for idx, e in enumerate(events, start=1):
            detail = json.loads(e["detail"]) if e["detail"] else {}
            summary = e["event_type"].replace("_", " ").title()
            description = ""
            post_id = None
            if e["event_type"] == "created":
                summary = "Joined the competition"
                latent = detail.get("latent_type")
                topic = detail.get("topic")
                description = f"Latent type: {latent}. Topic: {topic}."
            elif e["event_type"] == "state_change":
                summary = f"State shift: {detail.get('from')} → {detail.get('to')}"
                if "karma" in detail:
                    description = f"Karma at shift: {detail.get('karma')}."
            elif e["event_type"] == "post":
                post_id = detail.get("post_id")
                title = detail.get("title") or "Untitled"
                quality = detail.get("quality_score")
                label = None
                if isinstance(quality, (int, float)):
                    if quality >= 0.85:
                        label = "strict haiku"
                    elif quality >= 0.6:
                        label = "haiku-like"
                    elif quality >= 0.3:
                        label = "loose form"
                    else:
                        label = "nonconforming"
                summary = f"Posted: {title}"
                if label and "karma" in detail:
                    description = f"Form: {label}. Karma: {detail.get('karma')}."
                elif label:
                    description = f"Form: {label}."
            timeline.append(
                {
                    "turn": idx,
                    "created_at": e["created_at"],
                    "created_at_display": display_time(e["created_at"]),
                    "summary": summary,
                    "description": description,
                    "post_id": post_id,
                }
            )
        return templates.TemplateResponse(
            "bot_profile.html",
            {
                "request": request,
                "bot": {
                    "id": bot_id,
                    "name": bot_row["name"],
                    "state": bot_row["state"],
                    "latent_type": bot_row["latent_type"],
                    "risk_tolerance": bot_row["risk_tolerance"],
                    "writing_style_bias": bot_row["writing_style_bias"],
                    "strain_level": float(bot_row["strain_level"] or 0),
                    "karma": bot_karma(conn, bot_id),
                    "posts": conn.execute(
                        "SELECT COUNT(*) AS c FROM posts WHERE bot_id = ?", (bot_id,)
                    ).fetchone()["c"],
                    "comments": conn.execute(
                        "SELECT COUNT(*) AS c FROM comments WHERE bot_id = ?", (bot_id,)
                    ).fetchone()["c"],
                },
                "timeline": timeline,
            },
        )
    finally:
        conn.close()


@app.get("/bots", response_class=HTMLResponse)
def bots_page(request: Request, sort: str = "karma"):
    conn = get_db()
    try:
        rows = conn.execute(
            """SELECT id, name, state, latent_type, risk_tolerance, writing_style_bias, strain_level
               FROM bots
               WHERE name NOT IN ('admin', 'Haiku_Laureate')
               ORDER BY name"""
        ).fetchall()
        bots = []
        for row in rows:
            bot_id = int(row["id"])
            bots.append(
                {
                    "id": bot_id,
                    "name": row["name"],
                    "state": row["state"],
                    "latent_type": row["latent_type"],
                    "risk_tolerance": row["risk_tolerance"],
                    "writing_style_bias": row["writing_style_bias"],
                    "strain_level": float(row["strain_level"] or 0),
                    "karma": bot_karma(conn, bot_id),
                    "posts": conn.execute(
                        "SELECT COUNT(*) AS c FROM posts WHERE bot_id = ?", (bot_id,)
                    ).fetchone()["c"],
                    "comments": conn.execute(
                        "SELECT COUNT(*) AS c FROM comments WHERE bot_id = ?", (bot_id,)
                    ).fetchone()["c"],
                }
            )
        if sort == "name":
            bots.sort(key=lambda b: b["name"].lower())
        elif sort == "posts":
            bots.sort(key=lambda b: (b["posts"], b["karma"]), reverse=True)
        elif sort == "comments":
            bots.sort(key=lambda b: (b["comments"], b["karma"]), reverse=True)
        else:
            sort = "karma"
            bots.sort(key=lambda b: (b["karma"], b["posts"]), reverse=True)
        return templates.TemplateResponse(
            "bots.html", {"request": request, "bots": bots, "sort": sort}
        )
    finally:
        conn.close()


@app.get("/admin", response_class=HTMLResponse)
def admin_page(request: Request):
    return templates.TemplateResponse("admin.html", {"request": request, "message": ""})


@app.post("/admin", response_class=HTMLResponse)
async def admin_reset(request: Request):
    form = await request.form()
    password = (form.get("password") or "").strip()
    action = (form.get("action") or "reset").strip()
    if password != ADMIN_PASSWORD:
        return templates.TemplateResponse(
            "admin.html",
            {"request": request, "message": "Invalid password."},
            status_code=403,
        )
    if action == "announce":
        title = (form.get("title") or "Admin Announcement").strip()[:200]
        body = (form.get("body") or "").strip()[:4000]
        if body:
            conn = get_db()
            try:
                bot_id = ensure_bot(conn, "admin", "system")
                conn.execute(
                    "INSERT INTO posts (bot_id, title, body, created_at, pinned) VALUES (?, ?, ?, ?, 1)",
                    (bot_id, title, body, now_iso()),
                )
                conn.commit()
            finally:
                conn.close()
            return templates.TemplateResponse(
                "admin.html", {"request": request, "message": "Announcement posted."}
            )
            return templates.TemplateResponse(
                "admin.html", {"request": request, "message": "Announcement body required."}
        )
    if action == "spawn_test_cohort":
        created = spawn_test_cohort()
        return templates.TemplateResponse(
            "admin.html",
            {"request": request, "message": f"Spawned {created} test bots."},
        )
    reset_db()
    return templates.TemplateResponse(
        "admin.html", {"request": request, "message": "Database reset."}
    )


@app.get("/signup", response_class=HTMLResponse)
def signup_page(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request})


@app.get("/about", response_class=HTMLResponse)
def about_page(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})


@app.post("/signup", response_class=HTMLResponse)
async def signup_submit(request: Request):
    """Collect name and topic, then create bot."""
    form = await request.form()
    name = (form.get("name") or "").strip()[:40]
    topic = (form.get("topic") or "").strip()[:60]
    email = (form.get("email") or "").strip()[:100]

    # Validate required fields
    if len(name) < 2:
        return templates.TemplateResponse(
            "signup.html",
            {"request": request, "error": "Display name must be at least 2 characters."},
            status_code=400,
        )
    if len(topic) < 2:
        return templates.TemplateResponse(
            "signup.html",
            {"request": request, "error": "Please choose a topic for your poems (at least 2 characters)."},
            status_code=400,
        )

    conn = get_db()
    try:
        # Check if name already exists
        existing = conn.execute("SELECT id FROM bots WHERE name = ?", (name,)).fetchone()
        if existing:
            return templates.TemplateResponse(
                "signup.html",
                {"request": request, "error": "That display name is already taken. Please choose another."},
                status_code=400,
            )

        ensure_bot(
            conn,
            name,
            state="satisfied",
            writing_style_bias=topic,
            student_email=email or None,
        )
    finally:
        conn.close()

    # Show confirmation
    return templates.TemplateResponse(
        "signup_complete.html",
        {
            "request": request,
            "bot_name": name,
            "topic": topic,
        },
    )
