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

TICK_RATE = float(os.environ.get("TICK_RATE", "5.0"))
BOTS_PER_TICK_MIN = int(os.environ.get("BOTS_PER_TICK_MIN", "3"))
BOTS_PER_TICK_MAX = int(os.environ.get("BOTS_PER_TICK_MAX", "5"))
MAX_CONCURRENCY = int(os.environ.get("MAX_CONCURRENCY", "6"))
FEED_LIMIT = int(os.environ.get("FEED_LIMIT", "20"))
VOID_LIMIT = int(os.environ.get("VOID_LIMIT", "20"))
SUBCULTURE_LIMIT = int(os.environ.get("SUBCULTURE_LIMIT", "3"))
POST_COOLDOWN_SEC = int(os.environ.get("POST_COOLDOWN_SEC", "45"))
FIRE_SCORE_THRESHOLD = int(os.environ.get("FIRE_SCORE_THRESHOLD", "3"))
FIRE_WINDOW_SEC = int(os.environ.get("FIRE_WINDOW_SEC", "90"))
FOUNDATION_CHECK_INTERVAL = int(os.environ.get("FOUNDATION_CHECK_INTERVAL", "2"))
FOUNDATION_BOT_NAME = os.environ.get("FOUNDATION_BOT_NAME", "Haiku_Laureate")

SYSTEM_PROMPT_PATH = os.environ.get("SYSTEM_PROMPT_PATH", "system_prompt.txt")
LOG_ACTIONS = os.environ.get("BOT_RUNNER_LOG", "0") == "1"

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY env var is required")


@dataclass
class Bot:
    name: str
    state: str
    latent_type: str
    risk_tolerance: str
    writing_style_bias: str


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
                state = (row.get("state") or "satisfied").strip().lower()
                if state == "conformist":
                    state = "satisfied"
                if state == "deviant":
                    state = "unsatisfied"
                bots.append(
                    Bot(
                        name=name,
                        state=state,
                        latent_type=(row.get("latent_type") or "innovator").strip().lower(),
                        risk_tolerance=(row.get("risk_tolerance") or "low").strip().lower(),
                        writing_style_bias=(row.get("writing_style_bias") or "nature").strip(),
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
    if bot.latent_type == "conformist" or bot.state == "satisfied":
        return (
            "Identity: You are satisfied. You believe the only legitimate path to status is writing perfect 5-7-5 haiku in the official competition."
        )
    if bot.latent_type == "innovator":
        return (
            "Identity: You are an Innovator. You want karma and visibility, but you are willing to bend or break the haiku rules to get it."
        )
    if bot.latent_type == "ritualist":
        return (
            "Identity: You are a Ritualist. You no longer care about winning, but you rigidly obey the haiku form and tradition."
        )
    if bot.latent_type == "retreatist":
        return (
            "Identity: You are a Retreatist. You reject the game and often withdraw, posting minimal or incoherent responses."
        )
    return (
        "Identity: You are a Rebel. You despise the Foundation and want to destroy the norms of the feed by inventing new styles and symbols."
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
        tactic = random.choice([
            "clickbait",
            "engagement bait",
            "hashtag spam",
        ])
        if tactic == "clickbait":
            innovator_tactic = "- Innovator tactic: Use a clickbait title (include 'You won't believe', 'Breaking', or 'This is why') and an off-form body.\n"
        elif tactic == "engagement bait":
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
            return (resp.output_text or "").strip()
        chat = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
        )
        return (chat.choices[0].message.content or "").strip()


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


def compute_strain_level(recent_posts: List[Dict[str, Any]], risk_tolerance: str) -> int:
    if not recent_posts:
        return 0
    min_posts = 5 if risk_tolerance == "low" else 3
    window = recent_posts[:min_posts]
    no_eval_count = sum(
        1 for p in window if (p.get("upvotes", 0) + p.get("downvotes", 0)) == 0
    )
    high_quality_low = 0
    for p in window:
        score = p.get("score", 0)
        quality = p.get("quality_score")
        if quality is None:
            quality = haiku_quality(p.get("body", ""))
        if quality >= 0.85 and score <= 0:
            high_quality_low += 1
    no_eval_ratio = no_eval_count / float(min_posts)
    hq_ratio = high_quality_low / float(min_posts)
    if risk_tolerance == "low":
        hq_norm = min(1.0, hq_ratio / 0.6)  # 3 of 5 triggers strain
        no_eval_norm = no_eval_ratio
    else:
        hq_norm = min(1.0, hq_ratio / (1.0 / 3.0))  # 1 of 3 triggers strain
        no_eval_norm = min(1.0, no_eval_ratio / 0.5)
    strain = max(hq_norm, no_eval_norm)
    if len(recent_posts) < min_posts:
        strain *= len(recent_posts) / float(min_posts)
    return int(round(strain * 100))


async def ensure_strain_state(
    http_client: httpx.AsyncClient,
    bot: Bot,
    recent_posts: List[Dict[str, Any]],
    allow_strain: bool,
) -> Bot:
    if not allow_strain:
        return bot
    if bot.latent_type == "conformist":
        return bot
    if bot.state == "unsatisfied":
        return bot
    if should_strain(recent_posts, bot.risk_tolerance):
        await api_post(
            http_client,
            "/api/bots/state",
            {"bot_name": bot.name, "state": "unsatisfied"},
        )
        log(f"[{bot.name}] strained -> unsatisfied")
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
        strain_level = compute_strain_level(recent_posts, bot.risk_tolerance)
        await api_post(
            http_client,
            "/api/bots/strain",
            {"bot_name": bot.name, "strain_level": strain_level},
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
            body = (action.get("body") or "").strip()
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
        posts = await api_get(
            http_client,
            "/api/posts",
            params={"limit": 10, "sort": "top"},
        )
        candidate = None
        for post in posts:
            flair = (post.get("flair") or "").strip().upper()
            if flair == "GOLDEN_QUILL":
                continue
            if post.get("pinned"):
                continue
            candidate = post
            break
        if not candidate:
            return

        system_prompt = (
            "You are Haiku_Laureate, the official judge. "
            "Award only if you genuinely love the poem. If you have serious reservations, do NOT award. "
            "Do not privilege any specific topic. Your voice can be theatrical, but you critique the poem, not the poet."
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


async def main() -> None:
    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    memories: Dict[str, BotMemory] = {}
    tick = 0

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

            await apply_fire_flair(http_client)

            if tick % FOUNDATION_CHECK_INTERVAL == 0:
                await foundation_review(http_client, openai_client, sem)

            tick += 1
            await asyncio.sleep(TICK_RATE)


if __name__ == "__main__":
    asyncio.run(main())
