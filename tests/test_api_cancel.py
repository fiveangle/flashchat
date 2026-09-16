"""Functional cancellation checks against an explicitly started test server."""
import argparse
import http.client
import json
import socket
import time


def request(port, path, body=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    try:
        conn.request("POST" if body is not None else "GET", path,
                     json.dumps(body) if body is not None else None,
                     {"Content-Type": "application/json"})
        response = conn.getresponse()
        data = response.read()
        assert response.status == 200, (response.status, data)
        return json.loads(data)
    finally:
        conn.close()


def wait_phase(port, phase, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = request(port, "/health")
        if state["phase"] == phase:
            return state
        time.sleep(0.02)
    raise AssertionError(f"server did not reach {phase}: {state}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--case", choices=("all", "prefill", "generating", "tool"), default="all")
    args = parser.parse_args()
    port = args.port
    recovery = {"messages": [{"role": "user", "content": "Say hello."}],
                "stream": False, "max_tokens": 16, "temperature": 0}
    baseline = request(port, "/v1/chat/completions", recovery)["choices"][0]["message"]
    cases = [
        ("/v1/chat/completions", "prefill", {
            "messages": [{"role": "user", "content": "Summarize: " + "the blue sky " * 2000}],
            "stream": True, "max_tokens": 4096}),
        ("/v1/responses", "generating", {
            "input": "Count from 1 to 10000, writing every number. Do not stop early.",
            "stream": True, "max_output_tokens": 4096}),
        ("/v1/chat/completions", "tool", {
            "messages": [{"role": "user", "content":
                          "Use write_text to store every integer from 1 through 10000."}],
            "tools": [{"type": "function", "function": {
                "name": "write_text", "description": "Store generated text.",
                "parameters": {"type": "object", "properties": {"text": {"type": "string"}},
                               "required": ["text"]}}}],
            "tool_choice": {"type": "function", "function": {"name": "write_text"}},
            "stream": True, "max_tokens": 4096}),
    ]
    for path, phase, body in cases:
        if args.case not in ("all", phase):
            continue
        payload = json.dumps(body).encode()
        with socket.create_connection(("127.0.0.1", port), timeout=10) as stream:
            stream.sendall(f"POST {path} HTTP/1.1\r\nHost: localhost\r\n"
                           f"Content-Type: application/json\r\nContent-Length: {len(payload)}\r\n\r\n".encode()
                           + payload)
            assert b"200 OK" in stream.recv(4096)
            wait_phase(port, "generating" if phase == "tool" else phase)
        wait_phase(port, "idle")
        actual = request(port, "/v1/chat/completions", recovery)["choices"][0]["message"]
        assert actual == baseline, (phase, baseline, actual)
        print(f"PASS: disconnect during {phase}; next request accepted with unchanged output")


if __name__ == "__main__":
    main()
