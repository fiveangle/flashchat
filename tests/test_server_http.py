"""Exercise the production HTTP transport without loading model weights."""
import concurrent.futures
import http.client
import json
from pathlib import Path
import signal
import socket
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
        subprocess.run(["clang", "-Wall", "-Wextra", "-Werror", "-pthread",
                        str(ROOT / "tests/server_http_fixture.c"), "-o", cls.binary], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.process = subprocess.Popen([self.binary], stdout=subprocess.PIPE, text=True)
        self.port = int(self.process.stdout.readline())

    def tearDown(self):
        self.process.terminate()
        self.process.wait(timeout=3)
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

    def test_shutdown_during_blocked_output(self):
        with self.generation(b"bulk"):
            time.sleep(0.2)
            self.process.send_signal(signal.SIGTERM)
            self.assertEqual(self.process.wait(timeout=2), 0)


if __name__ == "__main__":
    unittest.main()
