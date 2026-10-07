"""Small, dependency-free DuckDuckGo search and web-page text extractor."""

import base64
import ipaddress
import socket
from html.parser import HTMLParser
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, quote_plus, unquote, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener, urlopen


USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
MAX_PAGE_BYTES = 2_000_000


def validate_public_url(url):
    """Reject non-public destinations before fetching user- or search-provided URLs."""
    parsed = urlparse(str(url))
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Only public HTTP or HTTPS URLs are supported.")
    if parsed.port not in (None, 80, 443):
        raise ValueError("Only standard HTTP and HTTPS ports are supported.")
    host = parsed.hostname.rstrip(".").lower()
    if host in {"localhost", "localhost.localdomain"} or host.endswith(".localhost") or host.endswith(".local"):
        raise ValueError("Local network addresses are not allowed.")
    try:
        addresses = {ipaddress.ip_address(host)}
    except ValueError:
        try:
            addresses = {ipaddress.ip_address(item[4][0]) for item in socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80), type=socket.SOCK_STREAM)}
        except OSError as exc:
            raise ValueError("The public web host could not be resolved.") from exc
    if not addresses or any(not address.is_global for address in addresses):
        raise ValueError("Only publicly routable web hosts are allowed.")
    return parsed


class _PublicRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        validate_public_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class _SearchParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.results = []
        self.current = None
        self.capture_snippet = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = attrs.get("class", "").split()
        if tag == "a" and ({"result__a", "result-link"} & set(classes)):
            self.current = {"url": attrs.get("href", ""), "title": ""}
        elif {"result__snippet", "result-snippet"} & set(classes):
            self.capture_snippet = True

    def handle_data(self, data):
        if self.current is not None:
            self.current["title"] += data
        elif self.capture_snippet and self.results:
            self.results[-1]["snippet"] += data

    def handle_endtag(self, tag):
        if tag == "a" and self.current is not None:
            url = _unwrap_search_url(self.current["url"])
            if url.startswith(("http://", "https://")):
                self.current["url"] = url
                self.current["snippet"] = ""
                self.results.append(self.current)
            self.current = None
        if tag in ("a", "div"):
            self.capture_snippet = False


def _unwrap_search_url(url):
    parsed = urlparse(url)
    if parsed.path == "/l/":
        return parse_qs(parsed.query).get("uddg", [url])[0]
    if parsed.netloc.endswith("bing.com") and parsed.path == "/ck/a":
        wrapped = parse_qs(parsed.query).get("u", [""])[0]
        if wrapped.startswith("a1"):
            try:
                padding = "=" * (-len(wrapped[2:]) % 4)
                return base64.urlsafe_b64decode(wrapped[2:] + padding).decode("utf-8")
            except (ValueError, UnicodeDecodeError):
                pass
    return url


class _BingParser(HTMLParser):
    """Extract ordinary Bing web results without depending on a JS runtime."""

    def __init__(self):
        super().__init__()
        self.results = []
        self.current = None
        self.result_depth = 0
        self.in_heading = False
        self.capture_title = False
        self.capture_snippet = False

    def _finish(self):
        if self.current is None:
            return
        self.current["title"] = " ".join(self.current["title"].split())
        self.current["snippet"] = " ".join(self.current["snippet"].split())
        url = _unwrap_search_url(self.current["url"])
        if url.startswith(("http://", "https://")) and self.current["title"]:
            self.current["url"] = url
            self.results.append(self.current)
        self.current = None
        self.result_depth = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = attrs.get("class", "").split()
        if tag == "li" and "b_algo" in classes:
            self._finish()
            self.current = {"url": "", "title": "", "snippet": ""}
            self.result_depth = 1
            return
        if self.current is None:
            return
        if tag == "li":
            self.result_depth += 1
        elif tag == "h2":
            self.in_heading = True
        elif tag == "a" and self.in_heading:
            self.current["url"] = attrs.get("href", "")
            self.capture_title = True
        elif tag == "p" and "b_lineclamp2" in classes:
            self.capture_snippet = True

    def handle_data(self, data):
        if self.current is None:
            return
        if self.capture_title:
            self.current["title"] += data
        elif self.capture_snippet:
            self.current["snippet"] += data

    def handle_endtag(self, tag):
        if tag == "a" and self.capture_title:
            self.capture_title = False
        elif tag == "h2":
            self.in_heading = False
        elif tag == "p":
            self.capture_snippet = False
        elif tag == "li" and self.current is not None:
            self.result_depth -= 1
            if self.result_depth <= 0:
                self._finish()


def search_web(query, limit=5, offset=0):
    """Search the public web and return title, URL, and snippet records."""
    offset = max(0, int(offset))
    endpoints = (
        ("Bing", "https://www.bing.com/search?q=" + quote_plus(query) + f"&first={offset + 1}", _BingParser),
        ("DuckDuckGo", "https://html.duckduckgo.com/html/?q=" + quote_plus(query) + f"&s={offset}", _SearchParser),
        ("DuckDuckGo Lite", "https://lite.duckduckgo.com/lite/?q=" + quote_plus(query) + f"&s={offset}", _SearchParser),
    )
    failures = []
    for provider, endpoint, parser_type in endpoints:
        request = Request(endpoint, headers={"User-Agent": USER_AGENT})
        try:
            with urlopen(request, timeout=20) as response:
                page = response.read(MAX_PAGE_BYTES).decode("utf-8", errors="replace")
        except (HTTPError, URLError, TimeoutError) as exc:
            failures.append(f"{provider}: {exc}")
            continue

        parser = parser_type()
        parser.feed(page)
        # De-duplicate URLs while keeping the provider's relevance order.
        unique = []
        seen = set()
        for result in parser.results:
            if result["url"] not in seen:
                seen.add(result["url"])
                unique.append(result)
        if unique:
            return unique[:limit]
        lowered = page.lower()
        if "captcha" in lowered or "anomaly-modal" in lowered or "select all squares" in lowered:
            failures.append(f"{provider} returned an anti-bot challenge")
        else:
            failures.append(f"{provider} returned no parseable results")

    if failures and all("returned no parseable results" in failure for failure in failures):
        return []
    detail = "; ".join(failures) or "the search provider returned no parseable results"
    raise RuntimeError(f"Web search unavailable: {detail}. Check the internet connection or try another query later.")


class _TextParser(HTMLParser):
    HIDDEN_TAGS = {"script", "style", "noscript", "svg", "header", "footer", "nav"}

    def __init__(self):
        super().__init__()
        self.hidden_depth = 0
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in self.HIDDEN_TAGS:
            self.hidden_depth += 1
        elif tag in {"p", "br", "div", "li", "h1", "h2", "h3", "tr"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.HIDDEN_TAGS and self.hidden_depth:
            self.hidden_depth -= 1
        elif tag in {"p", "div", "li", "h1", "h2", "h3", "tr"}:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.hidden_depth:
            cleaned = " ".join(data.split())
            if cleaned:
                self.parts.append(cleaned)


def fetch_page_text(url, timeout=20):
    """Fetch a public HTTP(S) page and return readable text, bounded in size."""
    validate_public_url(url)

    request = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with build_opener(_PublicRedirectHandler()).open(request, timeout=timeout) as response:
            validate_public_url(response.geturl())
            content_type = response.headers.get_content_type()
            if content_type not in {"text/html", "application/xhtml+xml"}:
                raise ValueError(f"Unsupported page type: {content_type}")
            html = response.read(MAX_PAGE_BYTES).decode("utf-8", errors="replace")
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"Could not download {url}: {exc}") from exc

    parser = _TextParser()
    parser.feed(html)
    text = unquote(" ".join(parser.parts))
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())
