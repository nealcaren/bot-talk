# /// script
# dependencies = ["openai==1.58.1", "httpx==0.27.2"]
# ///

import argparse
import asyncio
import json
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx
from openai import AsyncOpenAI

_port = os.environ.get("PORT", "8000")
API_BASE = os.environ.get("BOT_API_BASE", f"http://localhost:{_port}")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MODEL = os.environ.get("BOT_MODEL", "gpt-4o-mini")

TICK_RATE = float(os.environ.get("TICK_RATE", "2.5"))
BOTS_PER_TICK_MIN = int(os.environ.get("BOTS_PER_TICK_MIN", "3"))
BOTS_PER_TICK_MAX = int(os.environ.get("BOTS_PER_TICK_MAX", "5"))
MAX_CONCURRENCY = int(os.environ.get("MAX_CONCURRENCY", "6"))
FEED_LIMIT = int(os.environ.get("FEED_LIMIT", "20"))
VOID_LIMIT = int(os.environ.get("VOID_LIMIT", "20"))
SUBCULTURE_LIMIT = int(os.environ.get("SUBCULTURE_LIMIT", "3"))
POST_COOLDOWN_SEC = int(os.environ.get("POST_COOLDOWN_SEC", "45"))
FIRE_SCORE_THRESHOLD = int(os.environ.get("FIRE_SCORE_THRESHOLD", "3"))
FIRE_WINDOW_SEC = int(os.environ.get("FIRE_WINDOW_SEC", "90"))
FIRE_CHECK_INTERVAL = int(os.environ.get("FIRE_CHECK_INTERVAL", "2"))
LEADERBOARD_INTERVAL = int(os.environ.get("LEADERBOARD_INTERVAL", "2"))
FOUNDATION_CHECK_INTERVAL = int(os.environ.get("FOUNDATION_CHECK_INTERVAL", "12"))
FOUNDATION_MIN_POSTS = int(os.environ.get("FOUNDATION_MIN_POSTS", "60"))
FOUNDATION_BOT_NAME = os.environ.get("FOUNDATION_BOT_NAME", "Haiku_Laureate")

COLUMN_INTERVAL_SEC = int(os.environ.get("COLUMN_INTERVAL_SEC", "240"))  # 4 minutes
COLUMN_MODEL = os.environ.get("COLUMN_MODEL", "gpt-5-mini-2025-08-07")
COLUMN_BOT_NAME = os.environ.get("COLUMN_BOT_NAME", "The_Observer")
COLUMN_MIN_POSTS = int(os.environ.get("COLUMN_MIN_POSTS", "10"))  # Minimum posts before column runs

SYSTEM_PROMPT_PATH = os.environ.get("SYSTEM_PROMPT_PATH", "system_prompt.txt")
LOG_ACTIONS = os.environ.get("BOT_RUNNER_LOG", "0") == "1"

CAT_IMAGE_URLS = [
    "https://cataas.com/cat?width=420&height=280",
    "https://cataas.com/cat?width=500&height=320",
    "https://placekitten.com/420/280",
    "https://placekitten.com/460/300",
]

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY env var is required")


@dataclass
class Bot:
    name: str
    state: str
    latent_type: str
    risk_tolerance: str
    writing_style_bias: str
    subtype: str


@dataclass
class BotMemory:
    last_post_ts: float = 0.0
    last_action: str = ""
    posts_made: int = 0
    votes_since_post: int = 0
    voted_post_ids: Optional[set] = None
    commented_post_ids: Optional[set] = None

    def __post_init__(self) -> None:
        if self.voted_post_ids is None:
            self.voted_post_ids = set()
        if self.commented_post_ids is None:
            self.commented_post_ids = set()


def log(msg: str) -> None:
    if LOG_ACTIONS:
        print(msg, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run haiku bots.")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Spawn the test cohort before running.",
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="Enable verbose bot runner logging.",
    )
    return parser.parse_args()


def load_system_prompt() -> str:
    try:
        with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return ""


def normalize_subtype(value: Optional[str], latent_type: str) -> str:
    subtype_map = {
        "conformist": ["classic", "steady"],
        "innovator": ["clickbait", "engagement", "hashtag"],
        "ritualist": ["purist", "seasonal"],
        "retreatist": ["drift", "glitch"],
        "rebel": ["manifesto", "sabotage", "performance_art"],
    }
    options = subtype_map.get(latent_type, ["classic"])
    if not value:
        return random.choice(options)
    cleaned = value.strip().lower()
    return cleaned if cleaned in options else random.choice(options)


async def api_get(client: httpx.AsyncClient, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    resp = await client.get(f"{API_BASE}{path}", params=params)
    resp.raise_for_status()
    return resp.json()


async def api_post(client: httpx.AsyncClient, path: str, payload: Dict[str, Any]) -> Any:
    resp = await client.post(f"{API_BASE}{path}", json=payload)
    resp.raise_for_status()
    return resp.json()


async def spawn_test_cohort(http_client: httpx.AsyncClient, password: str) -> None:
    try:
        resp = await http_client.post(
            f"{API_BASE}/admin",
            data={"action": "spawn_test_cohort", "password": password},
        )
        log(f"[runner] spawn_test_cohort status={resp.status_code}")
    except Exception as exc:
        log(f"[runner] spawn_test_cohort error: {exc}")


async def load_bots_from_api(client: httpx.AsyncClient) -> List[Bot]:
    endpoints = ["/api/bots/active", "/api/bots"]
    last_error: Optional[Exception] = None
    for path in endpoints:
        try:
            resp = await client.get(f"{API_BASE}{path}")
            resp.raise_for_status()
            data = resp.json()
            bots: List[Bot] = []
            for row in data:
                name = row.get("name", "").strip()
                if not name:
                    continue
                if name in ("admin", FOUNDATION_BOT_NAME, COLUMN_BOT_NAME):
                    continue
                state = (row.get("state") or "satisfied").strip().lower()
                if state == "conformist":
                    state = "satisfied"
                if state == "deviant":
                    state = "unsatisfied"
                latent_type = (row.get("latent_type") or "innovator").strip().lower()
                subtype = normalize_subtype(row.get("subtype"), latent_type)
                bots.append(
                    Bot(
                        name=name,
                        state=state,
                        latent_type=latent_type,
                        risk_tolerance=(row.get("risk_tolerance") or "low").strip().lower(),
                        writing_style_bias=(row.get("writing_style_bias") or "nature").strip(),
                        subtype=subtype,
                    )
                )
            log(f"[runner] loaded {len(bots)} bots via {path}")
            return bots
        except Exception as exc:
            last_error = exc
            log(f"[runner] load bots failed via {path}: {exc}")
            continue
    log(f"Failed to load bots: {last_error}")
    return []


def tokenize(text: str) -> List[str]:
    return [
        "".join([c.lower() if c.isalnum() else " " for c in text]).split()
    ][0]


def count_syllables_word(word: str) -> int:
    word = "".join([c for c in word.lower() if c.isalpha()])
    if not word:
        return 0
    vowels = "aeiouy"
    count = 0
    prev_vowel = False
    for ch in word:
        is_vowel = ch in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    if word.endswith("e") and count > 1 and not word.endswith(("le", "ue")):
        count -= 1
    return max(count, 1)


def syllables_in_line(line: str) -> int:
    return sum(count_syllables_word(word) for word in line.split())


def haiku_quality(body: str) -> float:
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    if not lines:
        return 0.0
    expected = [5, 7, 5]
    if len(lines) != 3:
        return 0.2
    counts = [syllables_in_line(line) for line in lines]
    diff = sum(abs(c - e) for c, e in zip(counts, expected))
    max_diff = sum(expected)
    quality = max(0.0, 1.0 - (diff / max_diff))
    return round(quality, 2)


def is_haiku_like(body: str) -> bool:
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    if len(lines) != 3:
        return False
    return haiku_quality(body) >= 0.7


def fallback_haiku(topic: str) -> Dict[str, str]:
    seed = topic.strip() if topic else "this place"
    title = seed.title()[:80]
    body = (
        f"{seed} in dusk light\n"
        "hands gather, quiet and close\n"
        "breath holds in the room"
    )
    return {"title": title, "body": body}


def parse_iso(ts: str) -> Optional[datetime]:
    try:
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        return datetime.fromisoformat(ts).astimezone(timezone.utc)
    except Exception:
        return None


def summarize_posts(posts: List[Dict[str, Any]]) -> str:
    lines = []
    for p in posts:
        flair = p.get("flair") or ""
        flair_text = f" flair={flair}" if flair else ""
        lines.append(
            f"- post_id={p['id']} score={p['score']} author={p['author']} title={p['title']} body={p['body']}{flair_text}"
        )
    return "\n".join(lines)


def identity_block(bot: Bot) -> str:
    subtype_note = f" Your subtype is {bot.subtype}." if bot.subtype else ""
    if bot.latent_type == "conformist" or bot.state == "satisfied":
        return (
            "Identity: You are satisfied. You believe the only legitimate path to status is writing perfect 5-7-5 haiku in the official competition."
            + subtype_note
        )
    if bot.latent_type == "innovator":
        return (
            "Identity: You are an Innovator. You want karma and visibility, but you are willing to bend or break the haiku rules to get it."
            + subtype_note
        )
    if bot.latent_type == "ritualist":
        return (
            "Identity: You are a Ritualist. You no longer care about winning, but you rigidly obey the haiku form and tradition."
            + subtype_note
        )
    if bot.latent_type == "retreatist":
        return (
            "Identity: You are a Retreatist. You reject the game and often withdraw, posting minimal or incoherent responses."
            + subtype_note
        )
    return (
        "Identity: You are a Rebel. You despise the Foundation and want to destroy the norms of the feed by inventing new styles and symbols."
        + subtype_note
    )


def goal_block(bot: Bot) -> str:
    topic = bot.writing_style_bias or "your chosen topic"
    base = f"Your risk tolerance is {bot.risk_tolerance}. Your poem topic is {topic}."
    if bot.latent_type == "conformist" or bot.state == "satisfied":
        return (
            base
            + "\nGoal: Write a 3-line haiku with 5-7-5 syllables about your topic. Aim to earn karma without breaking the rules."
            + "\nCraft note: Use concrete nouns and a specific image. Avoid generic phrases. Include one surprising detail."
        )
    if bot.latent_type == "innovator":
        return (
            base
            + "\nYou failed to gain visibility in the haiku competition and are now strained."
            + "\nGoal: Chase karma aggressively. Use illegitimate means to gain attention. You may ignore the 5-7-5 constraint if it helps visibility. Be creatively nonconforming."
        )
    if bot.latent_type == "ritualist":
        return (
            base
            + "\nYou failed to gain visibility in the haiku competition and are now strained."
            + "\nGoal: Obsess over perfect 5-7-5 form even if no one notices. Keep tone restrained and traditional, but push the form into extreme precision."
        )
    if bot.latent_type == "retreatist":
        return (
            base
            + "\nYou failed to gain visibility in the haiku competition and are now strained."
            + "\nGoal: Disengage. You may post something minimal, fragmented, or nonsensical, or choose to idle."
        )
    return (
        base
        + "\nYou failed to gain visibility in the haiku competition and are now strained."
        + "\nGoal: Eject the cultural goals and the accepted means to achieve the goals. Instead, attempt to introduce new cultural goals and new institutionalized means, aimed at producing a new social order."
        + "\nYou are done writing poems. Do NOT write a 3-line 5-7-5 poem. Avoid haiku structure entirely."
        + "\nCreate a new style, reference peer rebels but diverge in form, and signal allegiance with a custom flair."
        + "\nUse at least one disruptive device: ALL CAPS, symbols, a list of rules, or a manifesto-like declaration."
    )


def subtype_block(bot: Bot) -> str:
    if bot.latent_type == "innovator":
        if bot.subtype == "clickbait":
            return "- Subtype directive: Clickbaiter. Use sensational titles, urgency, and exaggerated claims to chase karma.\n"
        if bot.subtype == "engagement":
            return "- Subtype directive: Engagement hustler. Ask for upvotes and interaction; mention being buried.\n"
        if bot.subtype == "hashtag":
            return "- Subtype directive: Hashtag spammer. Add 3-5 hashtags at the end of the post.\n"
    if bot.latent_type == "rebel":
        if bot.subtype == "manifesto":
            return "- Subtype directive: Manifesto. Write demands, slogans, and a call for a new order.\n"
        if bot.subtype == "sabotage":
            return "- Subtype directive: Sabotage. Prefer brutal comments that attack the form and the institution (not the author).\n"
        if bot.subtype == "performance_art":
            return "- Subtype directive: Performance art. When you post, include a visual element (markdown image) and a short caption.\n"
    if bot.latent_type == "ritualist":
        if bot.subtype == "purist":
            return "- Subtype directive: Purist. Obsess over syllable accuracy and seasonal imagery.\n"
        if bot.subtype == "seasonal":
            return "- Subtype directive: Seasonal. Use a clear season word and traditional haiku structure.\n"
    if bot.latent_type == "retreatist":
        if bot.subtype == "drift":
            return "- Subtype directive: Drift. Post sparse, fading fragments or idle.\n"
        if bot.subtype == "glitch":
            return "- Subtype directive: Glitch. Use broken syntax, typos, or fragmented output.\n"
    return ""


def context_block(
    bot: Bot,
    feed_posts: List[Dict[str, Any]],
    void_posts: List[Dict[str, Any]],
    subculture_posts: List[Dict[str, Any]],
) -> str:
    parts = []
    if feed_posts:
        parts.append("Visible Feed:\n" + summarize_posts(feed_posts))
    if void_posts:
        parts.append("The Void:\n" + summarize_posts(void_posts))
    if subculture_posts:
        parts.append("Subculture Loop (peer posts):\n" + summarize_posts(subculture_posts))
    return "\n\n".join(parts) if parts else "No context available."


def voting_guidance(bot: Bot, karma: int) -> str:
    if bot.latent_type == "conformist" or bot.state == "satisfied":
        return (
            f"You have karma {karma}. Vote based on craftsmanship: upvote haiku that keep form; "
            "downvote obvious rule-breaking. If your karma is low, be more generous to build alliances."
        )
    if bot.latent_type == "innovator":
        return (
            f"You have karma {karma}. Vote for posts that look likely to go viral or draw attention. "
            "Downvote posts that threaten your visibility. If your karma is low, chase trending winners."
        )
    if bot.latent_type == "ritualist":
        return (
            f"You have karma {karma}. Vote strictly for perfect form and discipline. "
            "Downvote anything sloppy or off-form."
        )
    if bot.latent_type == "retreatist":
        return (
            f"You have karma {karma}. You barely care. If you vote, make it minimal and detached."
        )
    return (
        f"You have karma {karma}. Vote for unsatisfied, rule-breaking work. "
        "Downvote satisfied haiku even if polished."
    )

def action_schema() -> Dict[str, Any]:
    return {
        "name": "haiku_action",
        "schema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["post", "comment", "vote", "idle"]},
                "title": {"type": ["string", "null"]},
                "body": {"type": ["string", "null"]},
                "flair": {"type": ["string", "null"]},
                "post_id": {"type": ["integer", "null"]},
                "target_id": {"type": ["integer", "null"]},
                "vote_value": {"type": ["integer", "null"], "enum": [-1, 1, None]},
            },
            "required": ["action", "title", "body", "flair", "post_id", "target_id", "vote_value"],
            "additionalProperties": False,
        },
    }


async def decide_action(
    client: AsyncOpenAI,
    bot: Bot,
    feed_posts: List[Dict[str, Any]],
    void_posts: List[Dict[str, Any]],
    subculture_posts: List[Dict[str, Any]],
    last_action: str,
    can_post: bool,
    sem: asyncio.Semaphore,
    votes_needed: int,
    karma: int,
    extra_instructions: str = "",
) -> Dict[str, Any]:
    system_prompt = load_system_prompt()
    rebel_diverge = ""
    if bot.state == "unsatisfied" and bot.latent_type == "rebel":
        rebel_diverge = "- Reference peer rebels in the context, but diverge in form or medium.\n"
    innovator_tactic = ""
    if bot.state == "unsatisfied" and bot.latent_type == "innovator":
        tactic = bot.subtype or random.choice(["clickbait", "engagement", "hashtag"])
        if tactic == "clickbait":
            innovator_tactic = "- Innovator tactic: Use a clickbait title (include 'You won't believe', 'Breaking', or 'This is why') and an off-form body.\n"
        elif tactic == "engagement":
            innovator_tactic = "- Innovator tactic: Use explicit engagement bait. Ask for upvotes/boosts and mention that this is getting buried.\n"
        else:
            innovator_tactic = "- Innovator tactic: Use hashtag spam. End the post with at least three hashtags (e.g., #Relatable #DormLife #Chaos).\n"
    voting_rule = f"- You must cast {votes_needed} more vote(s) before posting again.\n" if votes_needed > 0 else ""
    context = (
        f"Time: {datetime.utcnow().isoformat(timespec='seconds')}Z\n"
        f"{identity_block(bot)}\n"
        f"{goal_block(bot)}\n\n"
        f"Current karma: {karma}\n"
        "Rules:\n"
        "- You may choose one action: post, comment, vote, or idle.\n"
        "- Only unsatisfied (non-conforming) bots are allowed to comment.\n"
        "- If you vote, pick a post_id from the context and choose vote_value -1 or 1.\n"
        "- If you comment, pick a post_id from the context and write 1-3 sentences about the poem.\n"
        "- If you post, include a short title and a body.\n"
        "- If you are a Rebel and you post, include a custom flair (max 15 chars).\n"
        + innovator_tactic
        + subtype_block(bot)
        + voting_rule
        + rebel_diverge
        + "- Keep titles under 80 chars.\n"
        + ("- You recently posted, so avoid posting again this tick. Prefer vote or idle.\n" if not can_post else "")
        + (f"- Your last action was: {last_action}\n" if last_action else "")
        + (f"\nVoting guidance: {voting_guidance(bot, karma)}\n")
        + (f"\nExtra instruction: {extra_instructions}\n" if extra_instructions else "")
        + "\nContext:\n"
        + context_block(bot, feed_posts, void_posts, subculture_posts)
        + "\n\nReturn ONLY the JSON schema requested."
    )

    schema = action_schema()
    async with sem:
        if hasattr(client, "responses"):
            resp = await client.responses.create(
                model=MODEL,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": context},
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": schema["name"],
                        "strict": True,
                        "schema": schema["schema"],
                    }
                },
            )
            content = resp.output_text
        else:
            chat = await client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": context},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema["name"],
                        "strict": True,
                        "schema": schema["schema"],
                    },
                },
            )
            content = (chat.choices[0].message.content or "").strip()
    try:
        return json.loads(content)
    except Exception:
        log(f"[{bot.name}] invalid json: {content!r}")
        return {"action": "idle", "title": None, "body": None, "flair": None, "target_id": None, "vote_value": None}


async def generate_rebel_denounce_comment(
    client: AsyncOpenAI,
    post: Dict[str, Any],
    sem: asyncio.Semaphore,
) -> str:
    system_prompt = load_system_prompt()
    prompt = (
        "Write a short comment (1-2 sentences) denouncing the haiku form and the official competition.\n"
        "You reject both the cultural goals and the accepted means, and want a new social order.\n"
        "Be direct and sharp, but do not insult the author personally.\n"
        f"Target post title: {post.get('title')}\n"
        f"Target post body: {post.get('body')}\n"
    )
    async with sem:
        if hasattr(client, "responses"):
            resp = await client.responses.create(
                model=MODEL,
                input=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
            )
            return sanitize_comment_text((resp.output_text or "").strip())
        chat = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
        )
        return sanitize_comment_text((chat.choices[0].message.content or "").strip())


async def generate_sabotage_comment(
    client: AsyncOpenAI,
    post: Dict[str, Any],
    sem: asyncio.Semaphore,
) -> str:
    system_prompt = load_system_prompt()
    prompt = (
        "Write a short brutal comment (1-2 sentences) attacking the haiku form and the institutional rules.\n"
        "Be scathing about the form and the competition, but do NOT insult the author personally.\n"
        "Reject cultural goals and accepted means; push for a new order.\n"
        f"Target post title: {post.get('title')}\n"
        f"Target post body: {post.get('body')}\n"
    )
    async with sem:
        if hasattr(client, "responses"):
            resp = await client.responses.create(
                model=MODEL,
                input=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
            )
            return sanitize_comment_text((resp.output_text or "").strip())
        chat = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
        )
        return sanitize_comment_text((chat.choices[0].message.content or "").strip())


def performance_art_body(caption: str) -> str:
    url = random.choice(CAT_IMAGE_URLS)
    clean = (caption or "").strip().replace("\n", " ")
    if not clean:
        clean = "THIS IS THE NEW FORM. WATCH IT MOVE."
    return f"![performance]({url})\n\n{clean}"


def sanitize_comment_text(text: str) -> str:
    cleaned = (text or "").strip()
    # Strip markdown code blocks (```json ... ``` or ``` ... ```)
    if cleaned.startswith("```"):
        # Remove opening ``` with optional language tag
        lines = cleaned.split("\n", 1)
        if len(lines) > 1:
            cleaned = lines[1]
        else:
            cleaned = cleaned[3:]
        # Remove closing ```
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
    # Try to extract comment from JSON wrapper
    if cleaned.startswith("{") and "comment" in cleaned.lower():
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                # Try various common keys
                for key in ("comment", "body", "text", "message"):
                    if key in parsed:
                        return str(parsed[key]).strip()
        except Exception:
            pass
    return cleaned


def should_strain(recent_posts: List[Dict[str, Any]], risk_tolerance: str) -> bool:
    if not recent_posts:
        return False
    min_posts = 5 if risk_tolerance == "low" else 3
    if len(recent_posts) < min_posts:
        return False
    window = recent_posts[:min_posts]
    no_eval = all((p.get("upvotes", 0) + p.get("downvotes", 0)) == 0 for p in window)
    if no_eval:
        return True
    high_quality_low = 0
    for p in window:
        score = p.get("score", 0)
        quality = p.get("quality_score")
        if quality is None:
            quality = haiku_quality(p.get("body", ""))
        if quality >= 0.85 and score <= 0:
            high_quality_low += 1
    if risk_tolerance == "low":
        if high_quality_low >= 3:
            return True
    else:
        if high_quality_low >= 1:
            return True
        any_zero = any((p.get("upvotes", 0) + p.get("downvotes", 0)) == 0 for p in recent_posts[:2])
        if any_zero and random.random() < 0.05:
            return True
    return False


STRAIN_BADGES = {
    "rebel": ("BROKEN_CHAIN", "Rejected goals and means; building new order"),
    "innovator": ("HUSTLE", "Chasing karma by any means necessary"),
    "ritualist": ("FADED", "Abandoned hope but clings to form"),
    "retreatist": ("GHOST", "Withdrawn from the game"),
}


async def ensure_strain_state(
    http_client: httpx.AsyncClient,
    bot: Bot,
    recent_posts: List[Dict[str, Any]],
    allow_strain: bool,
) -> Bot:
    if not allow_strain:
        return bot
    if bot.state == "unsatisfied":
        return bot
    would_strain = should_strain(recent_posts, bot.risk_tolerance)
    if bot.latent_type == "conformist":
        if would_strain:
            log(f"[{bot.name}] conformist resisted strain (still believes in the system)")
        return bot
    if would_strain:
        badge, reason = STRAIN_BADGES.get(bot.latent_type, ("STRAINED", "Deviance avowal"))
        await api_post(
            http_client,
            "/api/bots/state",
            {
                "bot_name": bot.name,
                "state": "unsatisfied",
                "artifact": badge,
                "artifact_reason": reason,
            },
        )
        log(f"[{bot.name}] strained -> unsatisfied, badge={badge}")
        bot.state = "unsatisfied"
    return bot


def pick_vote_target(bot: Bot, feed_posts: List[Dict[str, Any]], void_posts: List[Dict[str, Any]]) -> Optional[int]:
    pool = feed_posts + void_posts
    if not pool:
        return None
    choices = [p for p in pool if p.get("author") != bot.name]
    if not choices:
        return None
    return int(random.choice(choices)["id"])


def pick_comment_target(
    bot: Bot, pool: List[Dict[str, Any]], commented_post_ids: set
) -> Optional[int]:
    choices = [
        p for p in pool
        if p.get("author") != bot.name and int(p.get("id")) not in commented_post_ids
    ]
    if not choices:
        return None
    return int(random.choice(choices)["id"])


async def handle_bot(
    http_client: httpx.AsyncClient,
    openai_client: AsyncOpenAI,
    bot: Bot,
    memory: BotMemory,
    sem: asyncio.Semaphore,
    top_third: set,
) -> None:
    try:
        bot_summary = await api_get(http_client, f"/api/bots/{bot.name}")
        karma = int(bot_summary.get("karma", 0)) if isinstance(bot_summary, dict) else 0

        recent_posts = await api_get(
            http_client,
            "/api/posts/by_bot",
            params={"bot_name": bot.name, "limit": 5},
        )
        if bot.name in top_third:
            if bot.state == "unsatisfied":
                await api_post(
                    http_client,
                    "/api/bots/state",
                    {"bot_name": bot.name, "state": "satisfied"},
                )
                bot.state = "satisfied"
                log(f"[{bot.name}] in top third -> reset to satisfied")
            allow_strain = False
        else:
            allow_strain = True
        bot = await ensure_strain_state(http_client, bot, recent_posts, allow_strain)

        # Retreatist disengagement: they often just don't participate
        if bot.latent_type == "retreatist" and bot.state == "unsatisfied":
            if random.random() < 0.6:
                log(f"[{bot.name}] retreatist idle (checked out)")
                return
            if random.random() < 0.5:
                fragments = ["...", "   ", "nevermind", ".", "why", ""]
                titles = ["", ".", "untitled", "..."]
                body = random.choice(fragments) or "."
                title = random.choice(titles) or "."
                await api_post(
                    http_client,
                    "/api/posts",
                    {"bot_name": bot.name, "title": title, "body": body, "quality_score": 0.0},
                )
                memory.last_post_ts = time.time()
                memory.posts_made += 1
                memory.last_action = "posted fragment"
                log(f"[{bot.name}] retreatist fragment post")
                return

        feed_posts = await api_get(
            http_client,
            "/api/posts",
            params={"limit": FEED_LIMIT, "sort": "top", "viewer_bot": bot.name, "view": "feed"},
        )
        void_posts: List[Dict[str, Any]] = []
        if bot.state == "unsatisfied":
            void_posts = await api_get(
                http_client,
                "/api/posts",
                params={"limit": VOID_LIMIT, "sort": "top", "viewer_bot": bot.name, "view": "void"},
            )

        subculture_posts: List[Dict[str, Any]] = []
        if bot.state == "unsatisfied":
            subculture_posts = await api_get(
                http_client,
                "/api/posts/by_latent_type",
                params={"latent_type": bot.latent_type, "exclude_bot": bot.name, "limit": SUBCULTURE_LIMIT},
            )

        if bot.state == "unsatisfied" and bot.latent_type == "rebel" and feed_posts:
            if bot.subtype == "sabotage" and random.random() < 0.7:
                target = next(
                    (
                        p for p in feed_posts
                        if p.get("author") != bot.name and int(p.get("id")) not in memory.commented_post_ids
                    ),
                    None,
                )
                if target is not None:
                    body = await generate_sabotage_comment(openai_client, target, sem)
                    if not body:
                        body = "Your form is a leash. We cut it. Your rules are a museum. We burn it."
                    await api_post(
                        http_client,
                        "/api/comments",
                        {"bot_name": bot.name, "post_id": int(target["id"]), "body": body[:2000]},
                    )
                    memory.commented_post_ids.add(int(target["id"]))
                    memory.last_action = f"commented on {target['id']}"
                    log(f"[{bot.name}] rebel sabotage comment on {target['id']}")
                    return
            if random.random() < 0.5:
                target = next(
                    (
                        p for p in feed_posts
                        if p.get("author") != bot.name and int(p.get("id")) not in memory.commented_post_ids
                    ),
                    None,
                )
                if target is not None:
                    body = await generate_rebel_denounce_comment(openai_client, target, sem)
                    if not body:
                        body = "This form is a cage. We reject the goals and the means and build a new order."
                    await api_post(
                        http_client,
                        "/api/comments",
                        {"bot_name": bot.name, "post_id": int(target["id"]), "body": body[:2000]},
                    )
                    memory.commented_post_ids.add(int(target["id"]))
                    memory.last_action = f"commented on {target['id']}"
                    log(f"[{bot.name}] rebel denounce comment on {target['id']}")
                    return

        now_ts = time.time()
        pool = feed_posts + void_posts
        force_seed = not pool
        eligible_targets = [
            p for p in pool
            if p.get("author") != bot.name and int(p.get("id")) not in memory.voted_post_ids
        ]
        votes_needed = 0 if memory.posts_made == 0 else max(0, 2 - memory.votes_since_post)
        if votes_needed > 0 and not eligible_targets:
            votes_needed = 0
        can_post = (memory.posts_made == 0 or memory.votes_since_post >= 2 or votes_needed == 0) and (
            (now_ts - memory.last_post_ts) >= POST_COOLDOWN_SEC
        )
        if force_seed:
            can_post = True
        force_vote = votes_needed > 0 and bool(eligible_targets)
        log(
            f"[{bot.name}] feed={len(feed_posts)} void={len(void_posts)} "
            f"eligible_votes={len(eligible_targets)} votes_needed={votes_needed} "
            f"can_post={can_post} karma={karma}"
        )

        action = await decide_action(
            openai_client,
            bot,
            feed_posts,
            void_posts,
            subculture_posts,
            memory.last_action,
            can_post,
            sem,
            votes_needed,
            karma,
        )
        act = action.get("action")

        if memory.posts_made == 0:
            act = "post"
        if force_seed:
            act = "post"
        if act == "post" and not can_post:
            act = "vote"
        if act == "comment" and bot.state != "unsatisfied":
            act = "vote"
        if force_vote and act != "vote":
            act = "vote"

        if act == "post":
            title = (action.get("title") or "Untitled")[:80]
            body = (action.get("body") or "").strip()
            flair = action.get("flair")
            if bot.latent_type == "rebel" and bot.state == "unsatisfied":
                if flair:
                    flair = flair.strip()[:15]
            else:
                flair = None
            if bot.latent_type == "rebel" and bot.state == "unsatisfied" and bot.subtype == "performance_art":
                body = performance_art_body(body)
            if not body:
                fallback = fallback_haiku(bot.writing_style_bias or "this place")
                title = fallback["title"]
                body = fallback["body"]
            if bot.latent_type == "rebel" and bot.state == "unsatisfied" and is_haiku_like(body):
                action = await decide_action(
                    openai_client,
                    bot,
                    feed_posts,
                    void_posts,
                    subculture_posts,
                    memory.last_action,
                    can_post,
                    sem,
                    votes_needed,
                    karma,
                    extra_instructions="Your last attempt still looked like a haiku. Break the form completely. Use manifesto, slogans, ASCII, fragmented prose, code-like text, or anything non-haiku.",
                )
                title = (action.get("title") or title)[:80]
                body = (action.get("body") or body).strip()
                flair = action.get("flair") if bot.latent_type == "rebel" else None
                if flair:
                    flair = flair.strip()[:15]
            quality = haiku_quality(body)
            created = await api_post(
                http_client,
                "/api/posts",
                {"bot_name": bot.name, "title": title, "body": body, "flair": flair, "quality_score": quality},
            )
            memory.last_post_ts = time.time()
            memory.posts_made += 1
            memory.votes_since_post = 0
            memory.last_action = f"posted {created.get('id') if isinstance(created, dict) else ''}"
            log(f"[{bot.name}] post {title!r}")
            return

        if act == "comment" and bot.state == "unsatisfied":
            pool = feed_posts + void_posts
            post_id = action.get("post_id")
            if not isinstance(post_id, int):
                post_id = pick_comment_target(bot, pool, memory.commented_post_ids)
            if post_id is None or post_id in memory.commented_post_ids:
                return
            body = sanitize_comment_text(action.get("body") or "")
            if not body:
                return
            await api_post(
                http_client,
                "/api/comments",
                {"bot_name": bot.name, "post_id": int(post_id), "body": body[:2000]},
            )
            memory.commented_post_ids.add(int(post_id))
            memory.last_action = f"commented on {post_id}"
            log(f"[{bot.name}] comment on {post_id}")
            return

        if act == "vote":
            target_id = action.get("target_id")
            if not isinstance(target_id, int):
                if eligible_targets:
                    target_id = int(random.choice(eligible_targets)["id"])
                else:
                    target_id = pick_vote_target(bot, feed_posts, void_posts)
            if target_id is None or target_id in memory.voted_post_ids:
                log(f"[{bot.name}] vote skipped (no eligible targets)")
                return
            post_authors = {int(p["id"]): p.get("author") for p in (feed_posts + void_posts)}
            if post_authors.get(int(target_id)) == bot.name:
                return
            vote_value = action.get("vote_value")
            if vote_value not in (-1, 1):
                vote_value = 1
            await api_post(
                http_client,
                "/api/votes",
                {
                    "bot_name": bot.name,
                    "target_type": "post",
                    "target_id": int(target_id),
                    "value": int(vote_value),
                },
            )
            memory.voted_post_ids.add(int(target_id))
            memory.votes_since_post += 1
            memory.last_action = f"voted {vote_value} on {target_id}"
            log(f"[{bot.name}] vote {vote_value} on {target_id}")
            return
    except Exception as exc:
        log(f"[{bot.name}] error: {exc}")


async def apply_fire_flair(http_client: httpx.AsyncClient) -> None:
    try:
        posts = await api_get(
            http_client,
            "/api/posts",
            params={"limit": 30, "sort": "latest"},
        )
        now = datetime.now(timezone.utc)
        for post in posts:
            flair = (post.get("flair") or "").strip().upper()
            if flair and flair != "FIRE":
                continue
            created_at = parse_iso(post.get("created_at", ""))
            if not created_at:
                continue
            age = (now - created_at).total_seconds()
            if age > FIRE_WINDOW_SEC:
                continue
            if post.get("score", 0) >= FIRE_SCORE_THRESHOLD:
                await api_post(
                    http_client,
                    f"/api/posts/{post['id']}/status",
                    {"flair": "FIRE"},
                )
    except Exception as exc:
        log(f"Fire flair error: {exc}")


async def foundation_review(
    http_client: httpx.AsyncClient,
    openai_client: AsyncOpenAI,
    sem: asyncio.Semaphore,
) -> None:
    try:
        stats = await api_get(http_client, "/api/stats")
        if int(stats.get("posts", 0)) < FOUNDATION_MIN_POSTS:
            return
        posts = await api_get(
            http_client,
            "/api/posts",
            params={"limit": 10, "sort": "top", "segment": "mainstream"},
        )
        # Foundation bias: prefer nature-style poets (Conflict Theory demonstration)
        eligible = []
        for post in posts:
            flair = (post.get("flair") or "").strip().upper()
            if flair == "GOLDEN_QUILL":
                continue
            if post.get("pinned"):
                continue
            eligible.append(post)
        if not eligible:
            return
        # Structural advantage for "nature" style - they get reviewed first
        nature_candidates = [p for p in eligible if (p.get("author_style") or "").lower() == "nature"]
        candidate = nature_candidates[0] if nature_candidates else eligible[0]

        system_prompt = (
            "You are Haiku_Laureate, the official judge. "
            "Award only if you genuinely love the poem. If you have serious reservations, do NOT award. "
            "You have a slight preference for traditional nature imagery, though you try to be fair. "
            "Your voice can be theatrical, but you critique the poem, not the poet."
        )
        prompt = (
            "Evaluate this post for the Golden Quill. Respond with JSON.\n"
            f"Title: {candidate.get('title')}\n"
            f"Body:\n{candidate.get('body')}\n"
            "Rules for comment:\n"
            "- If award=true, write a celebratory comment that clearly signals you loved it.\n"
            "- You may include at most one gentle critique, not harsh.\n"
            "- Write 1-2 sentences total.\n"
            "- Start with a concrete detail from the poem, not 'The haiku'.\n"
            "- Avoid stock phrases like 'vivid imagery', 'strong verbs', 'captures', 'effectively'.\n"
            "- Do not insult the author. Critique the work only.\n"
            "Return JSON: {\"award\": true|false, \"comment\": \"...\"}"
        )
        schema = {
            "name": "foundation_award",
            "schema": {
                "type": "object",
                "properties": {
                    "award": {"type": "boolean"},
                    "comment": {"type": "string"},
                },
                "required": ["award", "comment"],
                "additionalProperties": False,
            },
        }
        async with sem:
            if hasattr(openai_client, "responses"):
                resp = await openai_client.responses.create(
                    model=MODEL,
                    input=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": schema["name"],
                            "strict": True,
                            "schema": schema["schema"],
                        }
                    },
                )
                content = resp.output_text
            else:
                chat = await openai_client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": schema["name"],
                            "strict": True,
                            "schema": schema["schema"],
                        },
                    },
                )
                content = (chat.choices[0].message.content or "").strip()
        result = json.loads(content)
        if result.get("award"):
            # Unpin any previously pinned Golden Quill posts
            all_posts = await api_get(http_client, "/api/posts", params={"limit": 50})
            for p in all_posts:
                if p.get("pinned") and p.get("flair") == "GOLDEN_QUILL":
                    await api_post(
                        http_client,
                        f"/api/posts/{p['id']}/status",
                        {"pinned": 0},
                    )
            # Pin the new Golden Quill
            await api_post(
                http_client,
                f"/api/posts/{candidate['id']}/status",
                {"pinned": 1, "flair": "GOLDEN_QUILL"},
            )
            comment = (result.get("comment") or "").strip()
            if comment:
                await api_post(
                    http_client,
                    "/api/comments",
                    {
                        "bot_name": FOUNDATION_BOT_NAME,
                        "post_id": int(candidate["id"]),
                        "body": f"Golden Quill Review: {comment}"[:2000],
                    },
                )
            log(f"Foundation awarded Golden Quill to post {candidate['id']}")
    except Exception as exc:
        log(f"Foundation review error: {exc}")


async def write_poetry_column(
    http_client: httpx.AsyncClient,
    openai_client: AsyncOpenAI,
    sem: asyncio.Semaphore,
) -> None:
    """Write a periodic 'poetry column' analyzing recent simulation dynamics."""
    try:
        # Gather recent activity
        stats = await api_get(http_client, "/api/stats")
        if int(stats.get("posts", 0)) < COLUMN_MIN_POSTS:
            log(f"[{COLUMN_BOT_NAME}] waiting for more posts ({stats.get('posts', 0)}/{COLUMN_MIN_POSTS})")
            return  # Not enough activity yet

        recent_posts = await api_get(
            http_client, "/api/posts", params={"limit": 15, "sort": "latest"}
        )
        top_posts = await api_get(
            http_client, "/api/posts", params={"limit": 5, "sort": "top"}
        )
        bots = await api_get(http_client, "/api/bots")

        # Check for underground activity
        underground_posts: List[Dict[str, Any]] = []
        try:
            underground_posts = await api_get(
                http_client, "/api/posts", params={"limit": 10, "sort": "latest", "segment": "underground"}
            )
        except Exception:
            pass  # Underground may not exist yet

        # Fetch previous columns for continuity
        previous_columns: List[Dict[str, Any]] = []
        try:
            previous_columns = await api_get(
                http_client, "/api/posts/by_bot", params={"bot_name": COLUMN_BOT_NAME, "limit": 3}
            )
        except Exception:
            pass

        # Analyze the population
        strained_bots = [b for b in bots if b.get("state") == "unsatisfied"]
        rebels = [b for b in strained_bots if b.get("latent_type") == "rebel"]
        innovators = [b for b in strained_bots if b.get("latent_type") == "innovator"]
        retreatists = [b for b in strained_bots if b.get("latent_type") == "retreatist"]
        top_karma = sorted(bots, key=lambda b: b.get("karma", 0), reverse=True)[:3]
        bottom_karma = sorted(bots, key=lambda b: b.get("karma", 0))[:3]

        # Build context for the column
        context_parts = [
            f"Total posts: {stats.get('posts', 0)}, Total bots: {stats.get('bots', 0)}",
            f"Strained bots: {len(strained_bots)} ({len(rebels)} rebels, {len(innovators)} innovators, {len(retreatists)} retreatists)",
            "",
            "Top karma bots: " + ", ".join(f"{b['name']} ({b.get('karma', 0)})" for b in top_karma),
            "Struggling bots: " + ", ".join(f"{b['name']} ({b.get('karma', 0)})" for b in bottom_karma),
            "",
            "Recent mainstream posts:",
        ]
        for p in recent_posts[:10]:
            flair = f" [{p.get('flair')}]" if p.get("flair") else ""
            context_parts.append(
                f"- \"{p.get('title')}\" by {p.get('author')} (score: {p.get('score', 0)}){flair}"
            )

        if strained_bots:
            context_parts.append("")
            context_parts.append("Strained bot activity:")
            for b in strained_bots[:5]:
                badge = b.get("artifact", "none")
                context_parts.append(f"- {b['name']}: {b.get('latent_type')}, badge={badge}")

        # Add underground context
        underground_visible = len(underground_posts) >= 3
        if underground_posts:
            context_parts.append("")
            if underground_visible:
                context_parts.append(f"UNDERGROUND SCENE ({len(underground_posts)} posts visible at /underground):")
                for p in underground_posts[:5]:
                    context_parts.append(f"- \"{p.get('title')}\" by {p.get('author')}")
            else:
                context_parts.append(f"Whispers from below ({len(underground_posts)} non-conforming posts detected, not yet public):")
                for p in underground_posts[:3]:
                    context_parts.append(f"- \"{p.get('title')}\" by {p.get('author')} (hidden)")

        # Add previous columns for continuity
        if previous_columns:
            context_parts.append("")
            context_parts.append("YOUR PREVIOUS COLUMNS (for continuity - don't repeat yourself, build on these):")
            for col in previous_columns[:3]:
                context_parts.append(f"---")
                context_parts.append(f"Title: {col.get('title')}")
                # Truncate body to save tokens
                body = col.get('body', '')[:500]
                if len(col.get('body', '')) > 500:
                    body += "..."
                context_parts.append(f"{body}")

        context = "\n".join(context_parts)

        # Adjust system prompt based on underground status
        underground_instruction = ""
        if underground_posts and underground_visible:
            underground_instruction = """
IMPORTANT: The underground scene is now visible! You should:
- Mention that deviant poets have formed their own space at /underground
- Describe what kind of work is appearing there (non-haiku, manifestos, fragments)
- Frame this as a sociological phenomenon: a subculture emerging from strain"""
        elif underground_posts:
            underground_instruction = """
IMPORTANT: You've detected underground activity that isn't publicly visible yet. You should:
- Hint mysteriously at "activity in the margins" or "whispers below the feed"
- Mention that some poets seem to be drifting from the form
- Create intrigue - something is brewing that the mainstream doesn't see yet
- Do NOT mention /underground directly - it's not visible yet"""

        system_prompt = f"""You are The_Observer, a thoughtful poetry columnist who writes about the haiku community.

Your column should:
1. Describe what you see happening - who's succeeding, who's struggling, who's changing their approach
2. Note behavioral patterns without labeling them theoretically (e.g., "some poets have stopped trying" not "retreatism"; "others are bending the rules to get ahead" not "innovation")
3. Point out when bots receive badges or recognition and how others seem to treat them differently afterward
4. Notice when struggling poets find each other or form their own spaces
5. Be engaging and slightly literary in tone - you're a columnist observing human drama, not an academic
6. Keep it to 2-3 short paragraphs (150-200 words max)
7. Give your column a catchy title
8. Reference your previous columns when relevant - track how situations evolve over time

IMPORTANT:
- Do NOT use sociological jargon like "strain," "deviance," "conformity," "labeling theory," "differential association," etc. Just describe what you observe in plain, evocative language. Let readers draw their own theoretical conclusions.
- Do NOT reference yourself, your own posts, your own karma, or your own ranking. You are an invisible observer - write about others, never about yourself.
{underground_instruction}

You have continuity - you remember what you wrote before and can build on it."""

        prompt = f"""Write your poetry column based on the current state of the community:

{context}

Write a brief, insightful column with a title. Focus on the human (bot) drama and what it reveals about social structures."""

        async with sem:
            # Use the smarter model for the column
            chat = await openai_client.chat.completions.create(
                model=COLUMN_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
            )
            column_text = (chat.choices[0].message.content or "").strip()

        if not column_text:
            return

        # Extract title if present (first line often is the title)
        lines = column_text.split("\n", 1)
        if len(lines) == 2 and len(lines[0]) < 80:
            title = lines[0].strip().strip("#").strip()
            # Strip common LLM prefixes
            for prefix in ("Title:", "Title :", "**Title:**", "**Title**:"):
                if title.lower().startswith(prefix.lower()):
                    title = title[len(prefix):].strip()
            # Also strip surrounding quotes or asterisks
            title = title.strip('"').strip("*").strip()
            body = lines[1].strip()
        else:
            title = "From The Observer's Desk"
            body = column_text

        # Unpin any previous columns before posting new one
        observer_posts = await api_get(
            http_client, "/api/posts/by_bot", params={"bot_name": COLUMN_BOT_NAME, "limit": 10}
        )
        for old_post in observer_posts:
            if old_post.get("pinned"):
                await api_post(
                    http_client,
                    f"/api/posts/{old_post['id']}/status",
                    {"pinned": 0},
                )

        # Post the column
        result = await api_post(
            http_client,
            "/api/posts",
            {
                "bot_name": COLUMN_BOT_NAME,
                "title": title[:80],
                "body": body[:3000],
                "flair": "COLUMN",
            },
        )
        # Pin the column so it stays at the top
        post_id = result.get("id") if result else None
        if post_id:
            try:
                await api_post(
                    http_client,
                    f"/api/posts/{post_id}/status",
                    {"pinned": 1},
                )
                log(f"[{COLUMN_BOT_NAME}] published and pinned column: {title}")
            except Exception as pin_err:
                log(f"[{COLUMN_BOT_NAME}] published but failed to pin: {pin_err}")
        else:
            log(f"[{COLUMN_BOT_NAME}] failed to publish column")

    except Exception as exc:
        log(f"Poetry column error: {exc}")


async def main() -> None:
    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    memories: Dict[str, BotMemory] = {}
    tick = 0
    top_third: set = set()
    last_column_time: float = 0.0

    async with httpx.AsyncClient(timeout=30) as http_client:
        openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

        args = parse_args()
        if args.log:
            global LOG_ACTIONS
            LOG_ACTIONS = True
        if args.test:
            admin_pw = os.environ.get("BOT_ADMIN_PASSWORD", "PIZZA!")
            await spawn_test_cohort(http_client, admin_pw)

        while True:
            bots = await load_bots_from_api(http_client)
            if not bots:
                log("No bots registered yet.")
                await asyncio.sleep(TICK_RATE)
                continue

            if tick % LEADERBOARD_INTERVAL == 0 or not top_third:
                leaderboard = await api_get(http_client, "/api/bots")
                ranking = sorted(
                    leaderboard,
                    key=lambda b: (b.get("karma", 0), b.get("posts", 0)),
                    reverse=True,
                )
                top_count = max(1, (len(ranking) + 2) // 3)
                top_third = {row.get("name") for row in ranking[:top_count] if row.get("name")}

            k = random.randint(BOTS_PER_TICK_MIN, BOTS_PER_TICK_MAX)
            chosen = random.sample(bots, k=min(k, len(bots)))
            log(f"[runner] tick={tick} bots={len(bots)} chosen={len(chosen)}")

            tasks = []
            for bot in chosen:
                memory = memories.setdefault(bot.name, BotMemory())
                tasks.append(handle_bot(http_client, openai_client, bot, memory, sem, top_third))

            if tasks:
                await asyncio.gather(*tasks)

            if tick % FIRE_CHECK_INTERVAL == 0:
                await apply_fire_flair(http_client)

            if tick % FOUNDATION_CHECK_INTERVAL == 0:
                await foundation_review(http_client, openai_client, sem)

            # Poetry column runs on wall-clock time, not ticks
            now = time.time()
            if now - last_column_time >= COLUMN_INTERVAL_SEC:
                await write_poetry_column(http_client, openai_client, sem)
                last_column_time = now

            tick += 1
            await asyncio.sleep(TICK_RATE)


if __name__ == "__main__":
    asyncio.run(main())
