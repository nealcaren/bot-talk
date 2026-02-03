# Source this file: . ./env.sh

export BOT_API_BASE="http://localhost:8000"
export BOT_API_KEY="your-secret-key"
export OPENAI_API_KEY="your-openai-key"

export BOT_CSV="bots.csv"

export BOT_COUNT=250
export MAX_CONCURRENCY=25
export LOOP_DELAY_MIN=2.5
export LOOP_DELAY_MAX=7.0

export MIN_POSTS_BEFORE_COMMENTS=6
export POST_COOLDOWN_SEC=180
export POST_PROB=0.25
export MIN_POST_CHARS=240
export MAX_COMMENT_SENTENCES=3
export MAX_COMMENT_CHARS=350
export ASK_FOLLOWUP_PROB=0.7
export INTERNAL_THREAD_PROB=0.25
export MENTION_PROB=0.5

export START_JITTER_MAX=5
export SEED_RETRY_COOLDOWN=300
