"""Three growing requests against a harness-owned server; emit TSV measurements."""
import json
import sys
import time
import urllib.request

url = sys.argv[1]
history = [{"role": "system", "content": "Read the supplied notes. Reply with only OK."}]
for turn in range(1, 4):
    notes = (f"Observation {turn}: the workshop has blue doors, wooden tables, and labeled storage shelves. " * 16)
    history.append({"role": "user", "content": notes + "\nReply only OK."})
    payload = {"messages": history, "temperature": 0, "reasoning": False,
               "max_tokens": 8, "stream": True}
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    first = None
    content = ""
    with urllib.request.urlopen(req, timeout=600) as response:
        for line in response:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {}).get("content", "")
                if delta:
                    if first is None:
                        first = time.perf_counter()
                    content += delta
    total_ms = (time.perf_counter() - started) * 1000
    assert first is not None and content.strip() == "OK", (turn, content)
    # The end chunk may precede the owner's idle transition; counters already
    # describe the completed request and are published under the progress lock.
    with urllib.request.urlopen(url + "/health", timeout=10) as response:
        health = json.load(response)
    ttft_ms = (first - started) * 1000
    print(f"{turn}\t{ttft_ms:.1f}\t{total_ms:.1f}\t{health['cached_tokens']}\t{health['prompt_tokens']}", flush=True)
    history.append({"role": "assistant", "content": content})
