"""Check responsive status and exclusive admission on an idle, updated server."""
import argparse
import concurrent.futures
import json
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:9999")
    parser.add_argument("--prompt-repetitions", type=int, default=1,
                        help="Repeat the prompt to exercise longer batched prefill")
    args = parser.parse_args()

    def request(path, body=None, timeout=2):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(args.url + path, data=data,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return response.status, json.load(response)
        except urllib.error.HTTPError as exc:
            return exc.code, json.load(exc)

    _, initial = request("/health")
    if initial.get("phase") != "idle":
        raise SystemExit("Requires an idle server built with responsive HTTP status; no generation submitted.")
    body = {"model": initial["model"], "messages": [{"role": "user", "content":
            "Explain how a bicycle works in three short paragraphs. " * max(1, args.prompt_repetitions)}],
            "max_tokens": 32, "temperature": 0, "stream": False}
    phases = set()
    checked_busy = False
    max_latency = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        generation = pool.submit(request, "/v1/chat/completions", body, 1200)
        while not generation.done():
            started = time.monotonic()
            code, status = request("/health")
            max_latency = max(max_latency, time.monotonic() - started)
            assert code == 200 and status["max_context"] > 0, status
            assert 0 <= status["context_used"] <= status["max_context"], status
            phase = status["phase"]
            if phase not in phases:
                print(json.dumps(status), flush=True)
            phases.add(phase)
            if phase in ("preparing", "prefill", "generating") and not checked_busy:
                # Empty body cannot trigger useful model work even if generation
                # happens to finish between this snapshot and admission.
                code, error = request("/v1/responses", {})
                assert code == 503 and error["error"]["type"] == "server_busy", (code, error)
                checked_busy = True
            time.sleep(0.1)
        code, completion = generation.result()
        assert code == 200 and completion.get("choices"), completion
    assert checked_busy and "generating" in phases, phases
    _, final = request("/health")
    assert final["phase"] == "idle" and final["context_used"] > 0, final
    assert final["generated_tokens"] > 0, final
    print(f"PASS: exclusive inference; responsive health (worst {max_latency * 1000:.1f} ms); retained context after completion")


if __name__ == "__main__":
    main()
