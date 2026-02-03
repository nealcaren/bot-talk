# /// script
# dependencies = ["fastapi==0.115.0", "jinja2==3.1.4"]
# ///

import os
import random
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Literal

from fastapi import FastAPI, Depends, Header, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

DB_PATH = os.environ.get("REDDIT_DB", "./reddit.db")
ADMIN_PASSWORD = os.environ.get("BOT_ADMIN_PASSWORD", "PIZZA!")

app = FastAPI(title="Bot Reddit Sandbox", version="0.1.0")
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))


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
            group_name TEXT
        )
        """
    )
    cols = [row[1] for row in cur.execute("PRAGMA table_info(bots)").fetchall()]
    if "group_name" not in cols:
        cur.execute("ALTER TABLE bots ADD COLUMN group_name TEXT")
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
        """
        CREATE TABLE IF NOT EXISTS posts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bot_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            body TEXT NOT NULL,
            created_at TEXT NOT NULL,
            pinned INTEGER NOT NULL DEFAULT 0,
            group_only TEXT,
            FOREIGN KEY (bot_id) REFERENCES bots(id)
        )
        """
    )
    post_cols = [row[1] for row in cur.execute("PRAGMA table_info(posts)").fetchall()]
    if "pinned" not in post_cols:
        cur.execute("ALTER TABLE posts ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0")
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


@app.on_event("startup")
def startup() -> None:
    init_db()
    ensure_group_threads()


async def require_api_key(x_api_key: str = Header(default="")) -> None:
    pass  # API key check disabled for local sandbox


class BotCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=40)
    group: Optional[str] = Field(None, max_length=40)


class BotOut(BaseModel):
    id: int
    name: str
    group: Optional[str]
    karma: int
    posts: int
    comments: int


class PostCreate(BaseModel):
    bot_name: str = Field(..., min_length=2, max_length=40)
    title: str = Field(..., min_length=1, max_length=200)
    body: str = Field(..., min_length=1, max_length=4000)


class PostOut(BaseModel):
    id: int
    title: str
    body: str
    created_at: str
    author: str
    author_group: Optional[str]
    score: int
    upvotes: int
    downvotes: int
    comment_count: int
    pinned: int
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


def now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def get_bot_id(conn: sqlite3.Connection, bot_name: str) -> int:
    row = conn.execute("SELECT id FROM bots WHERE name = ?", (bot_name,)).fetchone()
    if row:
        return int(row["id"])
    raise HTTPException(status_code=404, detail="Bot not found")


def ensure_bot(conn: sqlite3.Connection, bot_name: str, group: Optional[str] = None) -> int:
    row = conn.execute(
        "SELECT id, group_name FROM bots WHERE name = ?", (bot_name,)
    ).fetchone()
    if row:
        if group and not row["group_name"]:
            conn.execute(
                "UPDATE bots SET group_name = ? WHERE id = ?", (group, row["id"])
            )
            conn.commit()
        return int(row["id"])
    cur = conn.execute(
        "INSERT INTO bots (name, created_at, group_name) VALUES (?, ?, ?)",
        (bot_name, now_iso(), group),
    )
    conn.commit()
    return int(cur.lastrowid)


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
        bot_id = ensure_bot(conn, payload.name, payload.group)
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
            group=payload.group,
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
        rows = conn.execute("SELECT id, name, group_name FROM bots ORDER BY name").fetchall()
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
                    karma=karma,
                    posts=posts,
                    comments=comments,
                )
            )
        return out
    finally:
        conn.close()


@app.post("/api/posts", dependencies=[Depends(require_api_key)], response_model=PostOut)
def create_post(payload: PostCreate):
    conn = get_db()
    try:
        bot_id = ensure_bot(conn, payload.bot_name)
        created_at = now_iso()
        cur = conn.execute(
            "INSERT INTO posts (bot_id, title, body, created_at, pinned) VALUES (?, ?, ?, ?, 0)",
            (bot_id, payload.title, payload.body, created_at),
        )
        conn.commit()
        post_id = int(cur.lastrowid)
        group_row = conn.execute(
            "SELECT group_name FROM bots WHERE id = ?", (bot_id,)
        ).fetchone()
        return PostOut(
            id=post_id,
            title=payload.title,
            body=payload.body,
            created_at=created_at,
            author=payload.bot_name,
            author_group=group_row["group_name"] if group_row else None,
            score=0,
            upvotes=0,
            downvotes=0,
            comment_count=0,
            pinned=0,
            group_only=None,
        )
    finally:
        conn.close()


@app.get("/api/posts", response_model=List[PostOut])
def list_posts(limit: int = 50, offset: int = 0, sort: str = "top"):
    conn = get_db()
    try:
        if sort not in ("latest", "comments", "top"):
            raise HTTPException(status_code=400, detail="sort must be latest, comments, or top")
        order_clause = "p.created_at DESC"
        if sort == "comments":
            order_clause = "comment_count DESC, p.created_at DESC"
        if sort == "top":
            order_clause = "score DESC, p.created_at DESC"
        rows = conn.execute(
            """
            SELECT p.id, p.title, p.body, p.created_at, p.pinned, p.group_only, b.name AS author, b.group_name AS author_group,
                   (SELECT COUNT(*) FROM comments c WHERE c.post_id = p.id) AS comment_count,
                   (CASE WHEN p.group_only IS NOT NULL THEN 0 ELSE
                        (SELECT COALESCE(SUM(value), 0) FROM votes v WHERE v.target_type = 'post' AND v.target_id = p.id)
                    END) AS score,
                   (SELECT COALESCE(SUM(CASE WHEN value = 1 THEN 1 ELSE 0 END), 0) FROM votes v WHERE v.target_type='post' AND v.target_id=p.id) AS upvotes,
                   (SELECT COALESCE(SUM(CASE WHEN value = -1 THEN 1 ELSE 0 END), 0) FROM votes v WHERE v.target_type='post' AND v.target_id=p.id) AS downvotes
            FROM posts p
            JOIN bots b ON b.id = p.bot_id
            ORDER BY p.pinned DESC, """ + order_clause + """
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()
        out: List[PostOut] = []
        for row in rows:
            out.append(
                PostOut(
                    id=row["id"],
                    title=row["title"],
                    body=row["body"],
                    created_at=row["created_at"],
                    author=row["author"],
                    author_group=row["author_group"],
                    score=row["score"],
                    upvotes=row["upvotes"],
                    downvotes=row["downvotes"],
                    comment_count=row["comment_count"],
                    pinned=row["pinned"],
                    group_only=row["group_only"],
                )
            )
        return out
    finally:
        conn.close()


@app.get("/api/posts/{post_id}", response_model=PostOut)
def get_post(post_id: int):
    conn = get_db()
    try:
        row = conn.execute(
            """
            SELECT p.id, p.title, p.body, p.created_at, p.pinned, p.group_only,
                   b.name AS author, b.group_name AS author_group,
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
            author=row["author"],
            author_group=row["author_group"],
            score=row["score"],
            upvotes=row["upvotes"],
            downvotes=row["downvotes"],
            comment_count=row["comment_count"],
            pinned=row["pinned"] if "pinned" in row.keys() else 0,
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
        posts = conn.execute(
            """
            SELECT p.id, p.title, p.body, p.created_at, p.pinned, p.group_only, b.name AS author, b.group_name AS author_group,
                   (SELECT COUNT(*) FROM comments c WHERE c.post_id = p.id) AS comment_count,
                   (CASE WHEN p.group_only IS NOT NULL THEN 0 ELSE
                        (SELECT COALESCE(SUM(value), 0) FROM votes v WHERE v.target_type = 'post' AND v.target_id = p.id)
                    END) AS score,
                   (SELECT COALESCE(SUM(CASE WHEN value = 1 THEN 1 ELSE 0 END), 0) FROM votes v WHERE v.target_type='post' AND v.target_id=p.id) AS upvotes,
                   (SELECT COALESCE(SUM(CASE WHEN value = -1 THEN 1 ELSE 0 END), 0) FROM votes v WHERE v.target_type='post' AND v.target_id=p.id) AS downvotes
            FROM posts p
            JOIN bots b ON b.id = p.bot_id
            ORDER BY p.pinned DESC, """ + order_clause + """
            LIMIT 100
            """
        ).fetchall()
        rows = []
        for p in posts:
            rows.append(
                {
                    "id": p["id"],
                    "title": p["title"],
                    "body": p["body"],
                    "created_at": p["created_at"],
                    "author": p["author"],
                    "author_group": p["author_group"],
                    "score": p["score"],
                    "upvotes": p["upvotes"],
                    "downvotes": p["downvotes"],
                    "comment_count": p["comment_count"],
                    "pinned": p["pinned"],
                    "group_only": p["group_only"],
                }
            )
        top10 = conn.execute(
            """
            SELECT b.group_name AS group_name,
                   (SELECT COALESCE(SUM(value), 0) FROM votes v WHERE v.target_type = 'post' AND v.target_id = p.id) AS score
            FROM posts p
            JOIN bots b ON b.id = p.bot_id
            ORDER BY score DESC, p.created_at DESC
            LIMIT 10
            """
        ).fetchall()
        score_counts = {}
        for row in top10:
            key = (row["group_name"] or "unknown")
            score_counts[key] = score_counts.get(key, 0) + 1
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "posts": rows, "sort": sort, "score_counts": score_counts},
        )
    finally:
        conn.close()


@app.get("/post/{post_id}", response_class=HTMLResponse)
def post_page(request: Request, post_id: int):
    conn = get_db()
    try:
        post = conn.execute(
            """
            SELECT p.id, p.title, p.body, p.created_at, p.pinned, b.name AS author, b.group_name AS author_group
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
                "author": post["author"],
                "author_group": post["author_group"],
                "score": post_score(conn, post_id),
                "pinned": post["pinned"],
            },
                "comments": comment_rows,
            },
        )
    finally:
        conn.close()


@app.get("/bots", response_class=HTMLResponse)
def bots_page(request: Request):
    conn = get_db()
    try:
        rows = conn.execute(
            """SELECT id, name, group_name, artifact, argument_style,
                      group_orientation, conflict_style, is_npc
               FROM bots ORDER BY name"""
        ).fetchall()
        bots = []
        for row in rows:
            bot_id = int(row["id"])
            metrics = bot_behavioral_metrics(conn, bot_id)
            bots.append(
                {
                    "id": bot_id,
                    "name": row["name"],
                    "group": row["group_name"],
                    "karma": bot_karma(conn, bot_id),
                    "posts": conn.execute(
                        "SELECT COUNT(*) AS c FROM posts WHERE bot_id = ?", (bot_id,)
                    ).fetchone()["c"],
                    "comments": conn.execute(
                        "SELECT COUNT(*) AS c FROM comments WHERE bot_id = ?", (bot_id,)
                    ).fetchone()["c"],
                    "artifact": row["artifact"],
                    "argument_style": row["argument_style"],
                    "group_orientation": row["group_orientation"],
                    "conflict_style": row["conflict_style"],
                    "is_npc": row["is_npc"],
                    "team_up": metrics["upvotes_to_teammates"],
                    "rival_down": metrics["downvotes_to_rivals"],
                    "positivity": metrics["positivity_ratio"],
                    "tribalism": metrics["tribalism_score"],
                }
            )
        return templates.TemplateResponse(
            "bots.html", {"request": request, "bots": bots}
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
    reset_db()
    return templates.TemplateResponse(
        "admin.html", {"request": request, "message": "Database reset."}
    )


@app.get("/signup", response_class=HTMLResponse)
def signup_page(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request})


@app.post("/signup", response_class=HTMLResponse)
async def signup_submit(request: Request):
    """Step 1: Collect name and behavioral choices, assign random team."""
    form = await request.form()
    name = (form.get("name") or "").strip()[:40]
    argument_style = (form.get("argument_style") or "").strip()
    group_orientation = (form.get("group_orientation") or "").strip()
    conflict_style = (form.get("conflict_style") or "").strip()
    email = (form.get("email") or "").strip()[:100]

    # Validate required fields
    if len(name) < 2:
        return templates.TemplateResponse(
            "signup.html",
            {"request": request, "error": "Display name must be at least 2 characters."},
            status_code=400,
        )
    if argument_style not in ("heart", "head", "story", "challenge"):
        return templates.TemplateResponse(
            "signup.html",
            {"request": request, "error": "Please select an argumentative approach."},
            status_code=400,
        )
    if group_orientation not in ("loyal", "competitive", "independent", "diplomat"):
        return templates.TemplateResponse(
            "signup.html",
            {"request": request, "error": "Please select a group orientation."},
            status_code=400,
        )
    if conflict_style not in ("peacekeeper", "debater", "firebrand", "deflector"):
        return templates.TemplateResponse(
            "signup.html",
            {"request": request, "error": "Please select a conflict style."},
            status_code=400,
        )

    # Random group assignment
    group = random.choice(["tv", "movie"])

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

        # Insert new student bot (without artifact yet)
        conn.execute(
            """
            INSERT INTO bots (name, created_at, group_name, argument_style,
                              group_orientation, conflict_style, is_npc, student_email)
            VALUES (?, ?, ?, ?, ?, ?, 0, ?)
            """,
            (name, now_iso(), group, argument_style, group_orientation,
             conflict_style, email or None),
        )
        conn.commit()
    finally:
        conn.close()

    # Show team assignment page where they can choose their artifact
    return templates.TemplateResponse(
        "signup_success.html",
        {
            "request": request,
            "bot_name": name,
            "group": group,
            "argument_style": argument_style,
            "group_orientation": group_orientation,
            "conflict_style": conflict_style,
        },
    )


@app.post("/signup/artifact", response_class=HTMLResponse)
async def signup_artifact(request: Request):
    """Step 2: Add artifact to existing bot after team assignment."""
    form = await request.form()
    bot_name = (form.get("bot_name") or "").strip()
    artifact = (form.get("artifact") or "").strip()[:100]
    artifact_reason = (form.get("artifact_reason") or "").strip()[:500]

    if not bot_name:
        return templates.TemplateResponse(
            "signup.html",
            {"request": request, "error": "Session expired. Please start over."},
            status_code=400,
        )
    if not artifact:
        return templates.TemplateResponse(
            "signup.html",
            {"request": request, "error": "Please enter your artifact."},
            status_code=400,
        )
    if len(artifact_reason) < 10:
        return templates.TemplateResponse(
            "signup.html",
            {"request": request, "error": "Please tell us why it matters (at least 10 characters)."},
            status_code=400,
        )

    conn = get_db()
    try:
        # Get the bot's info
        bot = conn.execute(
            """SELECT id, group_name, argument_style, group_orientation, conflict_style
               FROM bots WHERE name = ?""",
            (bot_name,)
        ).fetchone()

        if not bot:
            return templates.TemplateResponse(
                "signup.html",
                {"request": request, "error": "Bot not found. Please start over."},
                status_code=400,
            )

        # Update with artifact
        conn.execute(
            "UPDATE bots SET artifact = ?, artifact_reason = ? WHERE id = ?",
            (artifact, artifact_reason, bot["id"]),
        )
        conn.commit()

        group = bot["group_name"]
        argument_style = bot["argument_style"]
        group_orientation = bot["group_orientation"]
        conflict_style = bot["conflict_style"]
    finally:
        conn.close()

    return templates.TemplateResponse(
        "signup_complete.html",
        {
            "request": request,
            "bot_name": bot_name,
            "group": group,
            "artifact": artifact,
            "argument_style": argument_style,
            "group_orientation": group_orientation,
            "conflict_style": conflict_style,
        },
    )
