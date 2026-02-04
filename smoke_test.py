# /// script
# dependencies = ["fastapi==0.115.0", "httpx==0.27.2", "jinja2==3.1.4"]
# ///

import asyncio
import os
import sys
import tempfile
import httpx

os.environ.setdefault("BOT_API_KEY", "test-secret")

with tempfile.NamedTemporaryFile(prefix="reddit_test_", suffix=".db") as tmp:
    os.environ["REDDIT_DB"] = tmp.name

    sys.path.append(".")
    sys.path.append("reddit-bot-sandbox")
    import main as appmod
    appmod.init_db()

    async def run():
        transport = httpx.ASGITransport(app=appmod.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            headers = {"X-API-Key": os.environ["BOT_API_KEY"]}

            r = await client.post("/api/bots", headers=headers, json={"name":"bot_001"})
            print("create bot", r.status_code, r.json())

            r = await client.post(
                "/api/bots/state",
                headers=headers,
                json={"bot_name":"bot_001","state":"deviant"},
            )
            print("update bot state", r.status_code, r.json())

            r = await client.post(
                "/api/posts",
                headers=headers,
                json={"bot_name":"bot_001","title":"First post","body":"Hello world"},
            )
            print("create post", r.status_code, r.json())
            post_id = r.json()["id"]

            r = await client.post(
                f"/api/posts/{post_id}/status",
                headers=headers,
                json={"pinned":1,"flair":"GOLDEN_QUILL"},
            )
            print("update post status", r.status_code, r.json())

            r = await client.post(
                "/api/comments",
                headers=headers,
                json={"bot_name":"bot_001","post_id":post_id,"body":"Nice thread"},
            )
            print("create comment", r.status_code, r.json())

            r = await client.post(
                "/api/votes",
                headers=headers,
                json={"bot_name":"bot_001","target_type":"post","target_id":post_id,"value":1},
            )
            print("vote post", r.status_code, r.json())

            r = await client.get("/api/posts")
            print("list posts", r.status_code, r.json())

            r = await client.get("/api/posts", params={"viewer_bot":"bot_001","view":"feed"})
            print("list posts (viewer)", r.status_code, r.json())

            r = await client.get("/api/posts/by_bot", params={"bot_name":"bot_001","limit":2})
            print("list posts by bot", r.status_code, r.json())

            r = await client.get(f"/api/posts/{post_id}/comments")
            print("list comments", r.status_code, r.json())

            r = await client.get("/api/bots")
            print("list bots", r.status_code, r.json())

            r = await client.get("/")
            print("html /", r.status_code, "len", len(r.text))

        print("done")

    asyncio.run(run())
