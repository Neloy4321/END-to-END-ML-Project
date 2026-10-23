"""Bounded-cardinality, in-process request metrics without personal answers.

Counters reset at restart and are per-worker. Never capture URL query strings,
form values, IP addresses, session IDs, medical answers, or request bodies.
"""
from collections import defaultdict
from threading import Lock

SAFE_ROUTES = ("/", "/assess", "/api/compare", "/predict", "/healthz", "/readyz", "/metrics")


def route_group(path):
    return path if path in SAFE_ROUTES else "other"


def status_group(code):
    return str(int(code) // 100) + "xx"


class RequestCounters:
    def __init__(self):
        self._lock = Lock()
        self._counts = defaultdict(int)
        self._duration = defaultdict(float)

    def observe(self, path, status, elapsed_seconds):
        key = (route_group(path), status_group(status))
        with self._lock:
            self._counts[key] += 1
            self._duration[key] += max(0.0, float(elapsed_seconds))

    def exposition(self):
        with self._lock:
            counts, durations = dict(self._counts), dict(self._duration)
        lines = [
            "# HELP sleep_http_requests_total Total HTTP requests by fixed route and status class.",
            "# TYPE sleep_http_requests_total counter",
        ]
        for (route, status), count in sorted(counts.items()):
            lines.append(f'sleep_http_requests_total{{route="{route}",status="{status}"}} {count}')
        lines += [
            "# HELP sleep_http_request_duration_seconds_sum Sum of HTTP request durations.",
            "# TYPE sleep_http_request_duration_seconds_sum counter",
        ]
        for (route, status), seconds in sorted(durations.items()):
            lines.append(f'sleep_http_request_duration_seconds_sum{{route="{route}",status="{status}"}} {seconds:.6f}')
        return "\n".join(lines) + "\n"
