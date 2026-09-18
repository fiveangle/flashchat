"""Compare multi-turn API output with conversation caching off/on; no speed claims."""
import argparse
import copy
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def request(url, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as response:
        if not body or not body.get("stream"):
            return json.load(response)
        text = ""
        for line in response:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            for choice in chunk.get("choices", []):
                text += choice.get("delta", {}).get("content", "")
        return {"choices": [{"message": {"role": "assistant", "content": text}}]}


def wait_idle(url, process):
    for _ in range(600):
        if process.poll() is not None:
            raise AssertionError(f"test server exited: {process.returncode}")
        try:
            health = request(url, "/health")
            if health.get("phase") == "idle":
                return health
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(0.2)
    raise AssertionError("test server did not become idle")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--edge-only", action="store_true", help="Check cancellation and input rejection with system caching disabled")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cases = []
    expected = []
    for enabled in ((1,) if args.edge_only else (0, 1)):
        run_dir = output / f"cache-{enabled}"
        run_dir.mkdir(exist_ok=True)
        home_dir = run_dir / "home"
        home_dir.mkdir(exist_ok=True)
        config = run_dir / "config"
        config.write_text('MODEL="Qwen-Qwen36-35B-A3B"\nMTP="0"\nSYSTEM_PROMPT_CACHE="1"\n'
                          f'SYSTEM_PROMPT_CACHE_DIR="{run_dir / "system-cache"}"\n'
                          f'CONVERSATION_CACHE="{enabled}"\nREASONING="0"\n')
        if args.edge_only:
            config.write_text(config.read_text().replace('SYSTEM_PROMPT_CACHE="1"', 'SYSTEM_PROMPT_CACHE="0"'))
        env = dict(os.environ, HOME=str(home_dir), FLASHCHAT_CONTEXT_WINDOW="4096",
                   FLASHCHAT_KV_QUANT="8", FLASHCHAT_EXPERT_PIN_MAX_GB="0",
                   FLASHCHAT_CONVERSATION_CACHE=str(enabled), FLASHCHAT_MTP="0",
                   FLASHCHAT_SERVER_LOG=str(run_dir / "server.log"), FLASHCHAT_SERVER_DEBUG="0",
                   FLASHCHAT_SERVER_HTTP_LOG="0", FLASHCHAT_PREAD_PROFILE="",
                   FLASHCHAT_SYSTEM_PROMPT=str(home_dir / "absent-system.md"))
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        url = f"http://127.0.0.1:{port}"
        with (run_dir / "console.log").open("w") as log:
            process = subprocess.Popen([str(ROOT / "metal_infer/infer"), "--serve", str(port),
                                        "--model-id", "Qwen-Qwen36-35B-A3B", "--model", str(args.model_path),
                                        "--config", str(config), "--no-mtp"],
                                       cwd=ROOT / "metal_infer", env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                wait_idle(url, process)
                responses = []
                if args.edge_only:
                    body = {"messages": [{"role": "user", "content": "Reply only OK."}],
                            "session_id": "one", "max_tokens": 8, "temperature": 0,
                            "reasoning": False, "stream": False}
                    first = request(url, "/v1/chat/completions", body)
                    second = request(url, "/v1/chat/completions", body)
                    assert first["choices"][0]["message"] == second["choices"][0]["message"]
                    body["session_id"] = "two"
                    request(url, "/v1/chat/completions", body)
                    oversized = copy.deepcopy(body)
                    oversized["messages"][0]["content"] = "word " * 5000
                    try:
                        request(url, "/v1/chat/completions", oversized)
                        raise AssertionError("oversized prompt was accepted")
                    except urllib.error.HTTPError as exc:
                        assert exc.code == 400, exc.code
                    wait_idle(url, process)
                    cancel = {"messages": [{"role": "user", "content": "Count from 1 to 200, one number per line."}],
                              "max_tokens": 512, "temperature": 0, "reasoning": False, "stream": True}
                    data = json.dumps(cancel).encode()
                    with socket.create_connection(("127.0.0.1", port), timeout=120) as sock:
                        sock.sendall((f"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
                                      f"Content-Type: application/json\r\nContent-Length: {len(data)}\r\n\r\n").encode() + data)
                        received = b""
                        while b'"content":' not in received:
                            chunk = sock.recv(4096)
                            assert chunk, "stream ended before a content token"
                            received += chunk
                        sock.shutdown(socket.SHUT_RDWR)
                    wait_idle(url, process)
                    cancel.update(stream=False, max_tokens=8)
                    request(url, "/v1/chat/completions", cancel)
                    wait_idle(url, process)
                    text = (run_dir / "server.log").read_text()
                    assert "chatcmpl-2 conversation cache: reused" in text
                    assert "chatcmpl-3 conversation cache: no reusable history" in text
                    assert "chatcmpl-5 cancelled during generation" in text
                    assert "chatcmpl-6 conversation cache: no reusable history" in text
                    print("PASS: system-cache-independent reuse, session change, oversized input, disconnect invalidation", flush=True)
                elif enabled == 0:
                    history = [{"role": "system", "content": "Answer briefly and follow the user's instructions."}]
                    for index, content in enumerate(("Remember the code word amber. Reply only OK.",
                                                     "What is the code word? Reply with that word only.",
                                                     "Repeat the code word in uppercase only.")):
                        history.append({"role": "user", "content": content})
                        body = {"messages": copy.deepcopy(history), "max_tokens": 16,
                                "temperature": 0, "reasoning": False, "stream": index == 2,
                                "presence_penalty": 0.4, "repetition_penalty": 1.05}
                        cases.append((f"chat-turn-{index + 1}", "/v1/chat/completions", body))
                        result = request(url, cases[-1][1], body)
                        expected.append(result["choices"][0]["message"])
                        history.append(copy.deepcopy(expected[-1]))
                    edited = copy.deepcopy(cases[1][2])
                    edited["messages"][1]["content"] = "Remember the code word violet. Reply only OK."
                    cases.append(("edited-history", "/v1/chat/completions", edited))
                    cases.append(("shortened-history", "/v1/chat/completions", cases[0][2]))
                    cases.append(("identical-retry", "/v1/chat/completions", cases[0][2]))
                    new_system = copy.deepcopy(cases[0][2])
                    new_system["messages"][0]["content"] = "Answer in French."
                    cases.append(("changed-system", "/v1/chat/completions", new_system))
                    tool = {"type": "function", "function": {"name": "record_number", "description": "Record a number.",
                            "parameters": {"type": "object", "properties": {"value": {"type": "integer"}}, "required": ["value"]}}}
                    tool_body = {"messages": [{"role": "user", "content": "Record the number 7."}],
                                 "tools": [tool], "tool_choice": {"type": "function", "function": {"name": "record_number"}},
                                 "max_tokens": 48, "temperature": 0, "reasoning": False, "stream": False}
                    cases.append(("forced-tool", "/v1/chat/completions", tool_body))
                    for name, path, body in cases[3:]:
                        result = request(url, path, body)
                        expected.append(result["choices"][0]["message"])
                    tool_message = copy.deepcopy(expected[-1])
                    assert tool_message.get("tool_calls"), tool_message
                    tool_message["tool_calls"][0]["id"] = "fixture-call"
                    followup = copy.deepcopy(tool_body)
                    followup["tool_choice"] = "auto"
                    followup["messages"] += [tool_message, {"role": "tool", "tool_call_id": "fixture-call",
                                                          "name": "record_number", "content": "Recorded 7 successfully. Reply only done."}]
                    cases.append(("tool-result", "/v1/chat/completions", followup))
                    result = request(url, cases[-1][1], followup)
                    expected.append(result["choices"][0]["message"])
                    # Both endpoints use the same native renderer and exact-prefix checks.
                    inp = [{"role": "user", "content": "Remember the number 27. Reply only OK."}]
                    for index in range(2):
                        body = {"input": copy.deepcopy(inp), "temperature": 0, "max_output_tokens": 16,
                                "reasoning": False, "stream": False}
                        cases.append((f"responses-turn-{index + 1}", "/v1/responses", body))
                        result = request(url, cases[-1][1], body)
                        message = result["output"][0]
                        text = "".join(part.get("text", "") for part in message.get("content", []))
                        expected.append({"role": "assistant", "content": text})
                        inp += [{"role": "assistant", "content": text},
                                {"role": "user", "content": "Repeat the number only."}]
                    responses = expected
                    (output / "requests.json").write_text(json.dumps(cases, indent=2))
                else:
                    for (name, path, body), reference in zip(cases, expected):
                        result = request(url, path, body)
                        if path.endswith("responses"):
                            actual = {"role": "assistant", "content": "".join(
                                part.get("text", "") for part in result["output"][0].get("content", []))}
                        else:
                            actual = result["choices"][0]["message"]
                        # Call IDs are delivery identifiers, not generated semantic content.
                        def normalized(message):
                            message = copy.deepcopy(message)
                            for call in message.get("tool_calls", []):
                                call.pop("id", None)
                                call["function"]["arguments"] = json.loads(call["function"]["arguments"])
                            return message
                        responses.append(actual)
                        (run_dir / "responses.json").write_text(json.dumps(responses, indent=2))
                        assert normalized(actual) == normalized(reference), (name, reference, actual)
                        print(f"PASS: {name} output matches cache disabled", flush=True)
                (run_dir / "responses.json").write_text(json.dumps(responses, indent=2))
                wait_idle(url, process)
            finally:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
        print(f"Completed cache={enabled}; server stopped", flush=True)
    if args.edge_only:
        return
    (output / "requests.json").write_text(json.dumps(cases, indent=2))
    log = (output / "cache-1/server.log").read_text()
    for request_id in (2, 3, 6, 9, 11):
        prefix = "resp" if request_id == 11 else "chatcmpl"
        assert f"{prefix}-{request_id} conversation cache: reused" in log, request_id
    for request_id in (4, 5, 7, 8, 10):
        prefix = "resp" if request_id == 10 else "chatcmpl"
        assert f"{prefix}-{request_id} conversation cache: no reusable history" in log, request_id
    print("PASS: expected continuation/retry hits and changed-input misses; all test servers exited", flush=True)


if __name__ == "__main__":
    main()
