"""Exercise the production HTTP transport without loading model weights."""
import concurrent.futures
import http.client
import json
from pathlib import Path
import signal
import socket
import struct
import subprocess
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]


class TransportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="flashchat-http-")
        cls.binary = str(Path(cls.temp.name) / "server")
        subprocess.run(["clang", "-Wall", "-Wextra", "-Werror", "-pthread", "-x", "objective-c",
                        "-fobjc-arc", "-framework", "Foundation",
                        str(ROOT / "tests/server_http_fixture.c"), "-o", cls.binary], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.process = subprocess.Popen([self.binary], stdout=subprocess.PIPE, text=True)
        self.port = int(self.process.stdout.readline())

    def tearDown(self):
        self.process.terminate()
        try:
            self.process.wait(timeout=3)
        finally:
            if self.process.poll() is None:
                self.process.kill()
                self.process.wait()
            self.process.stdout.close()

    def request(self, path="/health", method="GET", body=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=2)
        try:
            conn.request(method, path, body=body)
            response = conn.getresponse()
            return response.status, response.read()
        finally:
            conn.close()

    def generation(self, body=b"{}"):
        sock = socket.create_connection(("127.0.0.1", self.port), timeout=2)
        sock.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nContent-Length: " +
                     str(len(body)).encode() + b"\r\n\r\n" + body)
        self.assertIn(b"200 OK", sock.recv(4096))
        return sock

    def test_health_busy_and_complete_stream(self):
        with self.generation() as stream:
            start = time.monotonic()
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                replies = list(pool.map(lambda _: self.request(), range(8)))
            self.assertLess(time.monotonic() - start, 0.8)
            self.assertTrue(all(code == 200 for code, _ in replies))
            self.assertEqual(json.loads(replies[0][1])["context_used"], 1234)
            code, body = self.request("/v1/responses", "POST", "{}")
            self.assertEqual(code, 503)
            self.assertEqual(json.loads(body)["error"]["type"], "server_busy")
            data = b""
            while part := stream.recv(4096):
                data += part
            self.assertEqual(data.count(b"data: token"), 15)
        self.assertEqual(self.request("/v1/models")[0], 200)
        # A completed generation releases admission for the next one.
        with self.generation():
            pass

    def title_request(self, stream=True, role="system", content=None):
        return json.dumps({"model": 'fixture-"quoted"', "stream": stream, "messages": [
            {"role": role, "content": content if content is not None else
             "You are a title generator. You output ONLY a thread title. Nothing else."},
            {"role": "user", "content": "Generate a title for this conversation:\n"},
            {"role": "user", "content": "hello"},
        ]})

    def assert_title(self, code, body, stream, expected="hello"):
        self.assertEqual(code, 200)
        if stream:
            events = [line[6:] for line in body.decode().splitlines() if line.startswith("data: ")]
            self.assertEqual(events[-1], "[DONE]")
            chunks = [json.loads(event) for event in events[:-1]]
            self.assertEqual(chunks[0]["choices"][0]["delta"]["role"], "assistant")
            self.assertEqual("".join(c["choices"][0]["delta"].get("content", "") for c in chunks),
                             expected)
            self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "stop")
            self.assertEqual(len({c["id"] for c in chunks}), 1)
        else:
            reply = json.loads(body)
            self.assertEqual(reply["choices"][0]["message"]["content"], expected)
            self.assertEqual(reply["choices"][0]["finish_reason"], "stop")
            self.assertEqual(reply["model"], 'fixture-"quoted"')

    def test_title_excerpt_and_character_boundaries(self):
        cases = [
            ("please create an appropriate .gitignore file and init this blank repo",
             "please create an appropriate .gitignore file and…"),
            ('  Fix\n\t"quoted"   paths \\ on macOS  ', 'Fix "quoted" paths \\ on macOS'),
            ([{"type": "text", "text": "Fix parser"}, {"type": "text", "text": "errors"}],
             "Fix parser errors"),
            ([{"type": "image_url", "image_url": {"url": "data:image/png;base64,"}},
              {"type": "text", "text": "Explain this image"}], "Explain this image"),
            ("x" * 50, "x" * 50),
            ("x" * 51, "x" * 49 + "…"),
            ("x" * 49 + " more", "x" * 49 + "…"),
            ("👩🏽‍💻" * 51, "👩🏽‍💻" * 49 + "…"),
            ("e\u0301" * 51, "e\u0301" * 49 + "…"),
            ("\n\t ", "New conversation"),
            (None, "New conversation"),
        ]
        for stream in (True, False):
            for content, expected in cases:
                with self.subTest(stream=stream, content=content):
                    request = json.loads(self.title_request(stream))
                    request["messages"][-1]["content"] = content
                    self.assert_title(*self.request("/v1/chat/completions", "POST", json.dumps(request)),
                                      stream, expected)
            request = json.loads(self.title_request(stream))
            request["messages"].insert(2, {"role": "user", "content": "  "})
            request["messages"].append({"role": "user", "content": "Later follow-up"})
            self.assert_title(*self.request("/v1/chat/completions", "POST", json.dumps(request)), stream)

    def test_title_before_chat_does_not_reserve_inference(self):
        for stream in (True, False):
            with self.subTest(stream=stream):
                title = http.client.HTTPConnection("127.0.0.1", self.port, timeout=2)
                try:
                    title.request("POST", "/v1/chat/completions", self.title_request(stream))
                    response = title.getresponse()
                    # Leave the title body unread while the chat starts.
                    with self.generation() as chat:
                        self.assert_title(response.status, response.read(), stream)
                        while chat.recv(4096):
                            pass
                finally:
                    title.close()

    def test_titles_bypass_busy_without_releasing_real_generation(self):
        with self.generation(b"quiet"):
            for stream in (True, False):
                for role in ("system", "developer"):
                    code, body = self.request("/v1/chat/completions", "POST",
                                              self.title_request(stream, role, [
                                                  {"type": "text", "text": "  You are a title generator.\n"}]))
                    self.assert_title(code, body, stream)
                    self.assertEqual(self.request("/v1/chat/completions", "POST", "{}")[0], 503)
            self.assertEqual(self.request()[0], 200)

    def test_title_matching_uses_parsed_leading_system_instruction(self):
        with self.generation(b"quiet"):
            for body in (self.title_request(role="user"),
                         self.title_request(content="Discuss: You are a title generator"),
                         '{"messages":', '[]',
                         json.dumps({"messages": [{"role": "system", "content": {"text": 3}}]}),
                         json.dumps({"messages": [None]})):
                self.assertEqual(self.request("/v1/chat/completions", "POST", body)[0], 503)
            title = json.loads(self.title_request())
            title["messages"].insert(0, {"role": "system", "content": "You are a coding assistant."})
            self.assertEqual(self.request("/v1/chat/completions", "POST", json.dumps(title))[0], 503)
            self.assertEqual(self.request("/v1/responses", "POST", self.title_request())[0], 503)
            escaped = self.title_request().replace("You are", "\\u0059ou are")
            self.assert_title(*self.request("/v1/chat/completions", "POST", escaped), True)

    def test_explicit_bind_address(self):
        for address in ("127.0.0.1", "0.0.0.0"):
            self.assertEqual(subprocess.check_output([self.binary, address], text=True).strip(), address)
        self.assertNotEqual(subprocess.run([self.binary, "not-an-address"]).returncode, 0)

    def test_slow_upload_does_not_block_status(self):
        with socket.create_connection(("127.0.0.1", self.port)) as slow:
            slow.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nContent-Length: 100\r\n\r\n{")
            self.assertEqual(self.request()[0], 200)
            with self.generation():
                self.assertEqual(self.request()[0], 200)

    def test_request_half_close_preserves_response(self):
        with self.generation() as stream:
            stream.shutdown(socket.SHUT_WR)
            data = b""
            while part := stream.recv(4096):
                data += part
            self.assertEqual(data.count(b"data: token"), 15)

    def test_slow_reader_and_disconnect(self):
        with self.generation(b"bulk") as slow:
            time.sleep(0.3)
            self.assertEqual(self.request()[0], 200)
            self.assertEqual(self.request("/v1/responses", "POST", "{}")[0], 503)
        # Disconnect must unblock a worker stalled behind a full output buffer.
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            code, _ = self.request("/v1/responses", "POST", "{}")
            if code == 200:
                break
            time.sleep(0.05)
        self.assertEqual(code, 200)

    def test_invalid_framing(self):
        for headers in (b"Content-Length: -1", b"Content-Length: 1048576",
                        b"Transfer-Encoding: chunked", b"Content-Length: 0\r\nContent-Length: 0"):
            with socket.create_connection(("127.0.0.1", self.port)) as sock:
                sock.settimeout(2)
                sock.sendall(b"POST /v1/chat/completions HTTP/1.1\r\n" + headers + b"\r\n\r\n")
                self.assertIn(b"400 Bad Request", sock.recv(4096))
        self.assertEqual(self.request()[0], 200)

    def test_cancel_long_compute_and_accept_next_request(self):
        for mode in (b"quiet", b"prefill"):
            with self.subTest(mode=mode):
                stream = self.generation(mode)
                self.assertEqual(self.request()[0], 200)
                if mode == b"quiet":
                    stream.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER,
                                      struct.pack("ii", 1, 0))
                stream.close()
                deadline = time.monotonic() + 2
                while time.monotonic() < deadline:
                    code, _ = self.request("/v1/responses", "POST", "{}")
                    if code == 200:
                        break
                    time.sleep(0.05)
                self.assertEqual(code, 200, "cancelled compute retained inference ownership")

    def test_shutdown_during_blocked_output(self):
        with self.generation(b"bulk"):
            time.sleep(0.2)
            self.process.send_signal(signal.SIGTERM)
            self.assertEqual(self.process.wait(timeout=2), 0)


if __name__ == "__main__":
    unittest.main()
