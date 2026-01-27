from post_on_x import XPoster

poster = XPoster()

result = poster.post(
    content="",
    hashtags="ai, agents, reliability",
    dry_run=True,
)

if result.ok:
    print("posted", result.tweet_ids)
else:
    print("failed", result.error_code, result.error_message)
