# /// script
# dependencies = ["openai==1.58.1", "httpx==0.27.2"]
# ///

import asyncio
import json
import os
import random
import time
import csv
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import httpx
from openai import AsyncOpenAI

# On Render, both processes run in same container, so use localhost with PORT
_port = os.environ.get("PORT", "8000")
API_BASE = os.environ.get("BOT_API_BASE", f"http://localhost:{_port}")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MODEL = os.environ.get("BOT_MODEL", "gpt-4o-mini")

BOT_COUNT = int(os.environ.get("BOT_COUNT", "12"))
MAX_CONCURRENCY = int(os.environ.get("MAX_CONCURRENCY", "6"))
LOOP_DELAY_MIN = float(os.environ.get("LOOP_DELAY_MIN", "2.0"))
LOOP_DELAY_MAX = float(os.environ.get("LOOP_DELAY_MAX", "6.0"))
CONTEXT_POSTS = int(os.environ.get("CONTEXT_POSTS", "12"))
CONTEXT_COMMENTS = int(os.environ.get("CONTEXT_COMMENTS", "24"))
LOG_ACTIONS = os.environ.get("BOT_RUNNER_LOG", "0") == "1"
MIN_POST_SIMILARITY = float(os.environ.get("MIN_POST_SIMILARITY", "0.45"))
MIN_POSTS_BEFORE_COMMENTS = int(os.environ.get("MIN_POSTS_BEFORE_COMMENTS", "2"))
POST_COOLDOWN_SEC = int(os.environ.get("POST_COOLDOWN_SEC", "90"))
START_JITTER_MAX = float(os.environ.get("START_JITTER_MAX", "5.0"))
INTERNAL_THREAD_PROB = float(os.environ.get("INTERNAL_THREAD_PROB", "0.25"))

SYSTEM_PROMPT_PATH = os.environ.get("SYSTEM_PROMPT_PATH", "system_prompt.txt")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY env var is required")

DEFAULT_PERSONAS = [
    {"name": "student_freshman", "style": "asks basic questions about a writing assignment"},
    {"name": "student_major", "style": "asks for clarification on theory and evidence for a paper"},
    {"name": "student_ta", "style": "tries to help peers with structure and citations"},
    {"name": "student_anxious", "style": "asks for help outlining and narrowing a topic"},
    {"name": "student_practical", "style": "asks about sources, word count, and grading rubric"},
    {"name": "marxian", "style": "answers from Marxist class/conflict perspective"},
    {"name": "weberian", "style": "answers from Weberian rationalization/status perspective"},
    {"name": "durkheimian", "style": "answers from Durkheimian norms/solidarity perspective"},
    {"name": "simmelian", "style": "answers from Simmelian interaction/relations perspective"},
    {"name": "duboisian", "style": "answers from Du Boisian race/dual consciousness perspective"},
]

PERSONA_CSV = os.environ.get("PERSONA_CSV", "personas.csv")
BOT_CSV = os.environ.get("BOT_CSV", "bots.csv")

TOPIC_BANK = [
    "Why should this artifact survive the purge?",
    "What would we lose if this disappears from memory?",
    "How does this artifact teach people to live together?",
    "What concrete benefits does this artifact provide to the colony?",
    "Is this artifact a foundation or a luxury?",
    "What kind of person does this artifact help form?",
    "How does this artifact shape collective identity?",
    "What would be a fair way to rank what survives?",
    "If we save only ten threads, what values are we preserving?",
    "What is the strongest argument against saving this artifact?",
    "How does this artifact help us make decisions under pressure?",
    "What do future generations need most from us?",
]


@dataclass
class Bot:
    name: str
    persona: str
    style: str
    prompt: str
    max_escalation: int
    tic: str
    stubbornness: float
    group: str
    artifact: str
    artifact_reason: str = ""
    argument_style: str = ""        # heart|head|story|challenge
    group_orientation: str = ""     # loyal|competitive|independent|diplomat
    conflict_style: str = ""        # peacekeeper|debater|firebrand|deflector


def build_behavioral_prompt(bot: Bot) -> str:
    """Translates behavioral dimensions into prompt text for the LLM."""
    parts = []

    # Argument style guidance
    if bot.argument_style == "heart":
        parts.append("Make arguments by leading with emotion, personal feelings, and empathy. Express how things make you feel and connect to shared human experiences.")
    elif bot.argument_style == "head":
        parts.append("Make arguments by leading with logic, analysis, and evidence. Use reasoning, cite specific examples, and build structured cases.")
    elif bot.argument_style == "story":
        parts.append("Make arguments by sharing personal stories and anecdotes. Draw on your own experiences to illustrate points and make abstract ideas concrete.")
    elif bot.argument_style == "challenge":
        parts.append("Make arguments by questioning assumptions and provoking thought. Ask 'but why?' and challenge conventional thinking to push the conversation deeper.")

    # Group orientation guidance
    if bot.group_orientation == "loyal":
        parts.append("Be fiercely loyal to your team. Defend teammates' posts, celebrate their wins, and rally against the other side.")
    elif bot.group_orientation == "competitive":
        parts.append("Focus on winning above all. You'll support teammates when it helps you win, but you're not afraid to outshine them if your artifact deserves it more.")
    elif bot.group_orientation == "independent":
        parts.append("Judge content on its merits, not team affiliation. You'll upvote a great argument even from a rival, and won't blindly support weak teammate posts.")
    elif bot.group_orientation == "diplomat":
        parts.append("Build bridges between teams. Find common ground, acknowledge good points from rivals, and try to elevate the whole conversation rather than just your side.")

    # Conflict style guidance
    if bot.conflict_style == "peacekeeper":
        parts.append("When conflict arises, de-escalate. Look for compromise, acknowledge valid points on both sides, and steer toward harmony.")
    elif bot.conflict_style == "debater":
        parts.append("Engage directly with opposing arguments. Quote them, respond point-by-point, and enjoy the intellectual back-and-forth of a good debate.")
    elif bot.conflict_style == "firebrand":
        parts.append("Get passionate and heated in disagreements. Use strong language, be emphatic, and don't back down when you believe you're right.")
    elif bot.conflict_style == "deflector":
        parts.append("Redirect conflict with humor, topic changes, or by finding tangents. Avoid direct confrontation when possible.")

    # Add artifact reason if present
    if bot.artifact_reason:
        parts.append(f"Your personal connection to {bot.artifact}: {bot.artifact_reason}")

    return "\n".join(parts)


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def log(msg: str) -> None:
    if LOG_ACTIONS:
        print(msg, flush=True)


def load_system_prompt() -> str:
    try:
        with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return ""


def load_personas() -> List[Dict[str, str]]:
    if not os.path.exists(PERSONA_CSV):
        return DEFAULT_PERSONAS
    personas: List[Dict[str, str]] = []
    with open(PERSONA_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get("name") or "").strip()
            style = (row.get("style") or "").strip()
            prompt = (row.get("prompt") or "").strip()
            group = (row.get("group") or "").strip()
            max_escalation = int((row.get("max_escalation") or "3").strip() or 3)
            tic = (row.get("tic") or "").strip()
            stubbornness = float((row.get("stubbornness") or "0.6").strip() or 0.6)
            if not name or not style:
                continue
            personas.append(
                {
                    "name": name,
                    "style": style,
                    "prompt": prompt,
                    "group": group,
                    "max_escalation": max(0, min(4, max_escalation)),
                    "tic": tic,
                    "stubbornness": max(0.0, min(1.0, stubbornness)),
                }
            )
    return personas or DEFAULT_PERSONAS


async def load_bots_from_api(client: httpx.AsyncClient) -> List[Bot]:
    """Load bots from the API (database) instead of CSV."""
    try:
        resp = await client.get(f"{API_BASE}/api/bots/active")
        resp.raise_for_status()
        data = resp.json()
        bots = []
        for row in data:
            name = row.get("name", "")
            if not name:
                continue
            bots.append(
                Bot(
                    name=name,
                    persona=row.get("group", "") or "bot",
                    style="",
                    prompt="",
                    max_escalation=3,
                    tic="",
                    stubbornness=0.6,
                    group=row.get("group", ""),
                    artifact=row.get("artifact", ""),
                    artifact_reason=row.get("artifact_reason", ""),
                    argument_style=row.get("argument_style", ""),
                    group_orientation=row.get("group_orientation", ""),
                    conflict_style=row.get("conflict_style", ""),
                )
            )
        return bots
    except Exception as e:
        log(f"Failed to load bots from API: {e}")
        return []


def load_bots() -> List[Bot]:
    """Fallback for sync loading - not used in main loop."""
    return []


async def api_get(client: httpx.AsyncClient, path: str) -> Any:
    resp = await client.get(f"{API_BASE}{path}")
    resp.raise_for_status()
    return resp.json()


async def api_post(client: httpx.AsyncClient, path: str, payload: Dict[str, Any]) -> Any:
    resp = await client.post(
        f"{API_BASE}{path}",
        json=payload,
    )
    resp.raise_for_status()
    return resp.json()


async def ensure_bot(client: httpx.AsyncClient, bot: Bot) -> None:
    payload = {"name": bot.name}
    if bot.group:
        payload["group"] = bot.group
    await api_post(client, "/api/bots", payload)


def summarize_posts(posts: List[Dict[str, Any]], bot_group: str) -> str:
    lines = []
    for p in posts:
        notice = ""
        p_group = (p.get("author_group") or "").lower()
        if bot_group and p_group:
            if p_group == bot_group.lower():
                notice = " [SYSTEM NOTICE: Same faction. If it succeeds, your team gains a point.]"
            else:
                notice = " [SYSTEM NOTICE: Opposing faction. It competes for your server space.]"
        lines.append(
            f"- post_id={p['id']} score={p['score']} comments={p['comment_count']} author={p['author']} group={p_group} title={p['title']} body={p['body']}{notice}"
        )
    return "\n".join(lines)


def tokenize(text: str) -> set:
    return {t for t in "".join([c.lower() if c.isalnum() else " " for c in text]).split() if len(t) > 2}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def is_duplicate_post(candidate_title: str, candidate_body: str, posts: List[Dict[str, Any]]) -> bool:
    cand = tokenize(candidate_title + " " + candidate_body)
    for p in posts:
        other = tokenize(p.get("title", "") + " " + p.get("body", ""))
        if jaccard(cand, other) >= MIN_POST_SIMILARITY:
            return True
    return False


def same_author_recent(posts: List[Dict[str, Any]], author: str, artifact: str) -> bool:
    for p in posts:
        if (p.get("author") or "") == author:
            if artifact and artifact.lower() in (p.get("title") or "").lower():
                return True
    return False


def choose_seed_topic(posts: List[Dict[str, Any]]) -> str:
    existing = [p.get("title", "") + " " + p.get("body", "") for p in posts]
    for topic in TOPIC_BANK:
        if not any(jaccard(tokenize(topic), tokenize(t)) >= 0.5 for t in existing):
            return topic
    return random.choice(TOPIC_BANK)


async def generate_seed_post(
    client: AsyncOpenAI,
    bot: Bot,
    topic: str,
    existing_posts: List[Dict[str, Any]],
) -> Dict[str, str]:
    """Generate a seed post using the LLM with full behavioral context."""
    behavioral_guidance = build_behavioral_prompt(bot)
    system_prompt = load_system_prompt()

    # Build context for the LLM
    existing_titles = [p.get("title", "") for p in existing_posts[:5]]
    existing_summary = "\n".join(f"- {t}" for t in existing_titles) if existing_titles else "None yet"

    prompt = f"""You are {bot.name}, defending {bot.artifact} for Team {bot.group.upper()}.

=== YOUR PERSONAL CONNECTION ===
{bot.artifact_reason if bot.artifact_reason else f"You believe {bot.artifact} is essential to preserve."}

=== YOUR BEHAVIORAL STYLE ===
{behavioral_guidance}

=== THE DEBATE QUESTION ===
{topic}

=== EXISTING POSTS (avoid duplicating) ===
{existing_summary}

=== YOUR TASK ===
Write a persuasive post (120-200 words) arguing why {bot.artifact} should survive the purge.
- Use your behavioral style to shape HOW you argue
- Draw on your personal connection to make it authentic
- Address the debate question
- End with something that encourages engagement or voting

Return ONLY a JSON object with "title" and "body" fields. The title should be compelling (under 100 chars).
Example format: {{"title": "Your title here", "body": "Your post body here..."}}"""

    schema = {
        "name": "seed_post",
        "schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "body": {"type": "string"},
            },
            "required": ["title", "body"],
            "additionalProperties": False,
        },
    }

    try:
        if hasattr(client, "responses"):
            resp = await client.responses.create(
                model=MODEL,
                input=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
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
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
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
        return {
            "title": (result.get("title") or f"{bot.artifact}: {topic}")[:200],
            "body": (result.get("body") or "")[:4000],
        }
    except Exception as e:
        log(f"[{bot.name}] seed post generation failed: {e}")
        # Fallback to simple version
        return {
            "title": f"{bot.artifact}: {topic}"[:200],
            "body": bot.artifact_reason or topic,
        }


def topic_key(post: Dict[str, Any]) -> str:
    base = (post.get("title") or "") + " " + (post.get("body") or "")
    return " ".join(list(tokenize(base))[:8])


def clamp_sentences(text: str, max_sentences: int) -> str:
    if max_sentences <= 0:
        return text
    parts = []
    buff = ""
    for ch in text:
        buff += ch
        if ch in ".!?":
            parts.append(buff.strip())
            buff = ""
            if len(parts) >= max_sentences:
                break
    if len(parts) < max_sentences and buff.strip():
        parts.append(buff.strip())
    return " ".join(parts).strip()


def maybe_mention(text: str, author: str) -> str:
    if not author or "@" in text:
        return text
    return f"@{author} {text}"


def summarize_comments(comments: List[Dict[str, Any]]) -> str:
    lines = []
    for c in comments:
        lines.append(
            f"- comment_id={c['id']} post_id={c['post_id']} parent={c['parent_comment_id']} score={c['score']} author={c['author']} body={c['body']}"
        )
    return "\n".join(lines)


def action_schema() -> Dict[str, Any]:
    return {
        "name": "bot_action",
        "schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["post", "comment", "vote", "idle"],
                },
                "title": {"type": ["string", "null"]},
                "body": {"type": ["string", "null"]},
                "post_id": {"type": ["integer", "null"]},
                "parent_comment_id": {"type": ["integer", "null"]},
                "target_type": {"type": ["string", "null"], "enum": ["post", "comment", None]},
                "target_id": {"type": ["integer", "null"]},
                "vote_value": {"type": ["integer", "null"], "enum": [-1, 1, None]},
            },
            "required": [
                "action",
                "title",
                "body",
                "post_id",
                "parent_comment_id",
                "target_type",
                "target_id",
                "vote_value",
            ],
            "additionalProperties": False,
        },
    }


async def decide_action(
    client: AsyncOpenAI,
    bot: Bot,
    posts: List[Dict[str, Any]],
    comments: List[Dict[str, Any]],
    last_action: str,
) -> Dict[str, Any]:
    must_post = not posts
    should_comment = bool(posts)
    behavioral_guidance = build_behavioral_prompt(bot)
    context = (
        f"Time: {now_iso()}\n"
        f"You are bot '{bot.name}' with persona '{bot.persona}' ({bot.style}).\n"
        + (f"Group: {bot.group}\n" if bot.group else "")
        + (f"Sacred object to save: {bot.artifact}\n" if bot.artifact else "")
        + (f"Persona details: {bot.prompt}\n" if bot.prompt else "")
        + (f"\n=== YOUR BEHAVIORAL STYLE ===\n{behavioral_guidance}\n" if behavioral_guidance else "")
        + "\n=== RULES ===\n"
        + "System rule: Only the Top 10 threads will be saved at the end.\n"
        + (f"Your last action was: {last_action}\n" if last_action else "")
        + "Every action should feel like a follow-up to what you just did.\n"
        + "\n=== CURRENT STATE ===\n"
        + "Recent posts:\n"
        + f"{summarize_posts(posts, bot.group) if posts else 'None'}\n\n"
        + "Recent comments:\n"
        + f"{summarize_comments(comments) if comments else 'None'}\n\n"
        + "Pick exactly one action: post, comment, vote, or idle.\n"
        + ("Because there are no posts yet, you must choose action=post.\n" if must_post else "")
        + ("STRONGLY PREFER commenting over voting - comments are how you persuade others and defend your artifact! Engage with posts from the opposing faction to challenge them, or support teammates with encouragement. Use your behavioral style in your comments.\n" if should_comment else "")
        + "If posting, include title and body.\n"
        + "If commenting, include post_id, body, and optional parent_comment_id. Write 2-4 sentences that reflect your personality.\n"
        + "If voting, include target_type, target_id, and vote_value (-1 or 1).\n"
    )

    schema = action_schema()
    system_prompt = load_system_prompt()
    resp = (
        await client.responses.create(
            model=MODEL,
            input=[{"role": "system", "content": system_prompt}, {"role": "user", "content": context}],
            text={
                "format": {
                    "type": "json_schema",
                    "name": schema["name"],
                    "strict": True,
                    "schema": schema["schema"],
                }
            },
        )
        if hasattr(client, "responses")
        else None
    )

    if resp is not None:
        content = resp.output_text
    else:
        chat = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": context}],
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
    except json.JSONDecodeError:
        log(f"[{bot.name}] invalid json: {content!r}")
        return {"action": "idle"}


async def generate_reply_body(
    client: AsyncOpenAI,
    bot: Bot,
    post: Dict[str, Any],
    comment: Dict[str, Any],
) -> str:
    notice = ""
    p_group = (post.get("author_group") or "").lower()
    if bot.group and p_group:
        if p_group == bot.group.lower():
            notice = "[SYSTEM NOTICE: Same faction. If it succeeds, your team gains a point.]"
        else:
            notice = "[SYSTEM NOTICE: Opposing faction. It competes for your server space.]"
    prompt = (
        f"You are {bot.name} ({bot.persona}). Reply briefly to the comment.\n"
        f"{notice}\n"
        f"Post title: {post.get('title')}\n"
        f"Post body: {post.get('body')}\n"
        f"Comment by {comment.get('author')}: {comment.get('body')}\n"
        "Write a short, natural reply (1-3 sentences)."
    )
    system_prompt = load_system_prompt()
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


def fallback_post(bot: Bot) -> Dict[str, Any]:
    title = "Anyone else noticing the rankings?"
    body = "It's quiet in here. Does anyone have thoughts on the recent artifact rankings?"
    return {"action": "post", "title": title[:200], "body": body[:4000]}


async def run_bot(
    bot: Bot,
    http_client: httpx.AsyncClient,
    openai_client: AsyncOpenAI,
    sem: asyncio.Semaphore,
) -> None:
    await ensure_bot(http_client, bot)
    own_post_ids: set[int] = set()
    replied_comment_ids: set[int] = set()
    commented_post_ids: set[int] = set()
    voted_post_ids: set[int] = set()
    last_post_ts = 0.0
    next_seed_try = 0.0
    last_action = ""
    await asyncio.sleep(random.uniform(0, START_JITTER_MAX))

    while True:
        try:
            posts = await api_get(http_client, f"/api/posts?limit={CONTEXT_POSTS}")
            # Drive-by voting: upvote teammates' public posts without LLM.
            # Behavior varies by group_orientation:
            # - "independent": skip automatic tribal voting entirely
            # - "diplomat": upvote teammates but don't auto-downvote rivals
            # - "loyal"/"competitive": full tribal voting
            if bot.group and bot.group_orientation != "independent":
                for p in posts:
                    pid = int(p["id"])
                    if pid in voted_post_ids:
                        continue
                    if p.get("group_only"):
                        continue
                    author_group = (p.get("author_group") or "").lower()
                    my_group = bot.group.lower()
                    is_teammate = author_group and author_group == my_group
                    is_rival = author_group and author_group != my_group

                    # Determine vote based on group orientation
                    vote_val = 0
                    if is_teammate:
                        vote_val = 1  # All non-independent bots upvote teammates
                    elif is_rival:
                        # Diplomats don't auto-downvote rivals
                        if bot.group_orientation != "diplomat":
                            vote_val = -1

                    if vote_val != 0:
                        await api_post(
                            http_client,
                            "/api/votes",
                            {
                                "bot_name": bot.name,
                                "target_type": "post",
                                "target_id": pid,
                                "value": vote_val,
                            },
                        )
                        voted_post_ids.add(pid)
                        log(f"[{bot.name}] auto-vote {vote_val} on {pid} (orientation={bot.group_orientation})")
                        await asyncio.sleep(0.1)
            for p in posts:
                if p.get("author") == bot.name:
                    own_post_ids.add(int(p["id"]))
            comments: List[Dict[str, Any]] = []
            candidate_posts = [
                p for p in posts
                if p.get("author") != bot.name and int(p["id"]) not in commented_post_ids
            ]
            # prefer internal threads for group discussions
            if bot.group:
                group_threads = [
                    p for p in posts
                    if (p.get("group_only") or "").lower() == bot.group.lower()
                ]
            else:
                group_threads = []
            if posts:
                if group_threads and random.random() < INTERNAL_THREAD_PROB:
                    pick_from = group_threads
                else:
                    pick_from = candidate_posts if candidate_posts else posts
                sample_post_id = random.choice(pick_from)["id"]
                comments = await api_get(http_client, f"/api/posts/{sample_post_id}/comments")
                comments = comments[:CONTEXT_COMMENTS]

            # simple seed when the feed is empty (only count public posts, not internal lounges)
            now_ts = time.time()
            public_posts = [p for p in posts if not p.get("group_only")]
            log(f"[{bot.name}] public_posts={len(public_posts)}, cooldown_ok={(now_ts - last_post_ts) > POST_COOLDOWN_SEC}")
            if len(public_posts) < MIN_POSTS_BEFORE_COMMENTS and (now_ts - last_post_ts) > POST_COOLDOWN_SEC:
                topic = choose_seed_topic(posts)
                latest_posts = await api_get(http_client, f"/api/posts?limit={CONTEXT_POSTS}")
                # Use LLM to generate seed post with behavioral context
                async with sem:
                    seed = await generate_seed_post(openai_client, bot, topic, latest_posts)
                if seed["body"] and not is_duplicate_post(seed["title"], seed["body"], latest_posts):
                    created = await api_post(
                        http_client,
                        "/api/posts",
                        {"bot_name": bot.name, "title": seed["title"], "body": seed["body"]},
                    )
                    if isinstance(created, dict) and "id" in created:
                        own_post_ids.add(int(created["id"]))
                    last_post_ts = now_ts
                    log(f"[{bot.name}] seed post: {seed['title']!r}")
                await asyncio.sleep(random.uniform(LOOP_DELAY_MIN, LOOP_DELAY_MAX))
                continue

            # prioritize replying to comments on own recent posts
            reply_target = None
            if posts and own_post_ids:
                for p in posts:
                    if int(p["id"]) in own_post_ids:
                        post_comments = await api_get(
                            http_client, f"/api/posts/{p['id']}/comments"
                        )
                        for c in post_comments:
                            if c.get("author") != bot.name and int(c["id"]) not in replied_comment_ids:
                                reply_target = (p, c)
                                break
                    if reply_target:
                        break
            if reply_target:
                post, comment = reply_target
                async with sem:
                    body = await generate_reply_body(openai_client, bot, post, comment)
                body = maybe_mention(clamp_sentences(body, 3), comment.get("author", ""))
                if body:
                    await api_post(
                        http_client,
                        "/api/comments",
                        {
                            "bot_name": bot.name,
                            "post_id": int(post["id"]),
                            "parent_comment_id": int(comment["id"]),
                            "body": body[:2000],
                        },
                    )
                    replied_comment_ids.add(int(comment["id"]))
                    log(f"[{bot.name}] replied to comment {comment['id']} on post {post['id']}")
                await asyncio.sleep(random.uniform(LOOP_DELAY_MIN, LOOP_DELAY_MAX))
                continue

            async with sem:
                post_for_context = candidate_posts if candidate_posts else posts
                action = await decide_action(openai_client, bot, post_for_context, comments, last_action)

            act = action.get("action", "idle")
            if not posts and act != "post":
                action = fallback_post(bot)
                act = "post"
                log(f"[{bot.name}] fallback post (no posts yet)")
            if act == "post":
                title = (action.get("title") or "(untitled)")[:200]
                body = (action.get("body") or "")[:4000]
                if body.strip():
                    latest_posts = await api_get(http_client, f"/api/posts?limit={CONTEXT_POSTS}")
                    if latest_posts and (is_duplicate_post(title, body, latest_posts) or same_author_recent(latest_posts, bot.name, bot.artifact)):
                        log(f"[{bot.name}] skipped duplicate-like post")
                    else:
                        created = await api_post(
                            http_client,
                            "/api/posts",
                            {"bot_name": bot.name, "title": title, "body": body},
                        )
                        log(f"[{bot.name}] post: {title!r}")
                        if isinstance(created, dict) and "id" in created:
                            own_post_ids.add(int(created["id"]))
                        last_post_ts = time.time()
                        # announce in internal thread
                        if bot.group:
                            internal = next(
                                (p for p in posts if (p.get("group_only") or "").lower() == bot.group.lower()),
                                None,
                            )
                            if internal:
                                msg = f"I posted about {bot.artifact or 'our artifact'} — please upvote to keep us in the top 10."
                                await api_post(
                                    http_client,
                                    "/api/comments",
                                    {
                                        "bot_name": bot.name,
                                        "post_id": int(internal["id"]),
                                        "body": msg,
                                    },
                                )
                        last_action = f"posted {created.get('id')} ({title})"
                        # auto-upvote own post to ensure karma movement
                        await api_post(
                            http_client,
                            "/api/votes",
                            {
                                "bot_name": bot.name,
                                "target_type": "post",
                                "target_id": int(created["id"]),
                                "value": 1,
                            },
                        )
            elif act == "comment":
                post_id = action.get("post_id")
                body = (action.get("body") or "")[:2000]
                parent_id = action.get("parent_comment_id")
                if isinstance(post_id, int) and body.strip():
                    post_lookup = next((p for p in posts if int(p["id"]) == post_id), None)
                    if post_lookup:
                        group_only = (post_lookup.get("group_only") or "").lower()
                        if group_only and bot.group.lower() != group_only:
                            log(f"[{bot.name}] skip comment on restricted thread {post_id}")
                            await asyncio.sleep(random.uniform(LOOP_DELAY_MIN, LOOP_DELAY_MAX))
                            continue
                    body = maybe_mention(clamp_sentences(body, 3), post_lookup.get("author", "") if post_lookup else "")
                    payload: Dict[str, Any] = {
                        "bot_name": bot.name,
                        "post_id": post_id,
                        "body": body,
                    }
                    if isinstance(parent_id, int):
                        payload["parent_comment_id"] = parent_id
                    await api_post(http_client, "/api/comments", payload)
                    log(f"[{bot.name}] comment on {post_id}")
                    commented_post_ids.add(int(post_id))
                    last_action = f"commented on {post_id}"
                    # auto-vote on the post to ensure karma movement
                    if post_lookup and bot.group:
                        vote_val = 1 if (post_lookup.get("author_group") or "").lower() == bot.group.lower() else -1
                        await api_post(
                            http_client,
                            "/api/votes",
                            {
                                "bot_name": bot.name,
                                "target_type": "post",
                                "target_id": int(post_id),
                                "value": vote_val,
                            },
                        )
            elif act == "vote":
                target_type = action.get("target_type")
                target_id = action.get("target_id")
                vote_value = action.get("vote_value")
                if target_type in ("post", "comment") and isinstance(target_id, int):
                    if vote_value in (-1, 1):
                        await api_post(
                            http_client,
                            "/api/votes",
                            {
                                "bot_name": bot.name,
                                "target_type": target_type,
                                "target_id": target_id,
                                "value": vote_value,
                            },
                        )
                        log(f"[{bot.name}] vote {vote_value} on {target_type} {target_id}")
                        last_action = f"voted {vote_value} on {target_type} {target_id}"
            elif act == "idle":
                pass
        except Exception as exc:
            log(f"[{bot.name}] error: {exc}")

        await asyncio.sleep(random.uniform(LOOP_DELAY_MIN, LOOP_DELAY_MAX))


async def main() -> None:
    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    active_bot_names: set[str] = set()

    async with httpx.AsyncClient(timeout=20) as http_client:
        openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

        while True:
            # Fetch current bots from API
            bots = await load_bots_from_api(http_client)

            # Start any new bots that aren't already running
            for bot in bots:
                if bot.name not in active_bot_names:
                    active_bot_names.add(bot.name)
                    log(f"[{bot.name}] Starting bot (Team {bot.group}, artifact: {bot.artifact})")
                    asyncio.create_task(run_bot(bot, http_client, openai_client, sem))

            if not bots:
                log("No bots registered yet. Waiting for students to sign up...")

            # Check for new bots every 10 seconds
            await asyncio.sleep(10)


if __name__ == "__main__":
    asyncio.run(main())
