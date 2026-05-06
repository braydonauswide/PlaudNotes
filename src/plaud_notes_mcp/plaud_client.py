"""Plaud Notes API client.

Reverse-engineered API client for accessing Plaud Notes recordings,
transcripts, and AI summaries. Based on the publicly documented APIs
at api.plaud.ai used by web.plaud.ai.
"""

from __future__ import annotations

import gzip
import json as _json
import logging
import os
import re
import stat
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

# Strict pattern for Plaud file IDs (32-char hex)
_FILE_ID_PATTERN = re.compile(r"^[a-fA-F0-9]{24,64}$")

# Allowed API domains for redirect safety
_ALLOWED_API_DOMAINS = frozenset({
    "api.plaud.ai",
    "api-euc1.plaud.ai",
    "api-use1.plaud.ai",
})

# Regional API base URLs
API_DOMAINS = {
    "us": "https://api.plaud.ai",
    "eu": "https://api-euc1.plaud.ai",
}

DEFAULT_TIMEOUT = 30.0


@dataclass
class Recording:
    """A Plaud recording entry."""

    file_id: str
    filename: str
    duration_ms: int
    start_time: int
    end_time: int
    filesize: int
    created_at: datetime | None = None

    is_transcribed: bool = False
    is_summarized: bool = False
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> Recording:
        created = None
        st = data.get("start_time", 0)
        if st:
            try:
                created = datetime.fromtimestamp(st / 1000, tz=timezone.utc)
            except (ValueError, OSError):
                pass
        return cls(
            file_id=data.get("id", data.get("file_id", "")),
            filename=data.get("filename", "Untitled"),
            duration_ms=data.get("duration", 0),
            start_time=st,
            end_time=data.get("end_time", 0),
            filesize=data.get("filesize", 0),
            created_at=created,
            is_transcribed=bool(data.get("is_trans", False)),
            is_summarized=bool(data.get("is_summary", False)),
            tags=data.get("filetag_id_list", []) or [],
        )

    @property
    def duration_str(self) -> str:
        total_seconds = self.duration_ms // 1000
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours}h {minutes}m {seconds}s"
        if minutes:
            return f"{minutes}m {seconds}s"
        return f"{seconds}s"


@dataclass
class TranscriptSegment:
    """A single segment of a transcript."""

    text: str
    speaker: str = ""
    start_ms: int = 0
    end_ms: int = 0

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> TranscriptSegment:
        return cls(
            text=data.get("text", ""),
            speaker=data.get("speaker", data.get("spk", "")),
            start_ms=data.get("start_time_ms", data.get("start", data.get("bg", 0))),
            end_ms=data.get("end_time_ms", data.get("end", data.get("ed", 0))),
        )


@dataclass
class Transcript:
    """Full transcript for a recording."""

    file_id: str
    segments: list[TranscriptSegment] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        """Get the full transcript as plain text with speaker labels."""
        lines = []
        current_speaker = None
        for seg in self.segments:
            if seg.speaker and seg.speaker != current_speaker:
                current_speaker = seg.speaker
                lines.append(f"\n[{current_speaker}]")
            lines.append(seg.text)
        return "\n".join(lines).strip()

    @property
    def text_only(self) -> str:
        """Get transcript text without speaker labels."""
        return " ".join(seg.text for seg in self.segments if seg.text)


@dataclass
class Tag:
    """A Plaud tag/folder."""

    tag_id: str
    name: str
    count: int = 0


class PlaudAPIError(Exception):
    """Raised when the Plaud API returns an error."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class PlaudAuthError(PlaudAPIError):
    """Raised when authentication fails."""


class PlaudClient:
    """Client for the Plaud Notes API.

    Authentication requires a bearer token obtained from web.plaud.ai.
    The token can be provided directly, via PLAUD_TOKEN env var, or
    from a ~/.config/plaud/token file.
    """

    def __init__(
        self,
        token: str | None = None,
        region: str = "us",
        api_domain: str | None = None,
    ):
        self._token = self._resolve_token(token)
        if api_domain:
            self._base_url = self._validate_api_url(api_domain)
        else:
            self._base_url = API_DOMAINS.get(region, API_DOMAINS["us"])

        self._client = httpx.Client(
            base_url=self._base_url,
            headers=self._build_headers(),
            timeout=DEFAULT_TIMEOUT,
        )

    @staticmethod
    def _validate_file_id(file_id: str) -> str:
        """Validate that a file_id is a safe hex string."""
        if not _FILE_ID_PATTERN.match(file_id):
            raise PlaudAPIError(
                f"Invalid file_id format: expected 24-64 character hex string, "
                f"got {file_id!r:.50}"
            )
        return file_id

    @staticmethod
    def _validate_api_url(url: str) -> str:
        """Validate that an API URL is HTTPS and points to a known Plaud domain."""
        url = url.rstrip("/")
        parsed = urlparse(url)
        if parsed.scheme != "https":
            raise PlaudAPIError(
                f"API URL must use HTTPS, got {parsed.scheme!r}"
            )
        if parsed.hostname not in _ALLOWED_API_DOMAINS:
            raise PlaudAPIError(
                f"API domain {parsed.hostname!r} is not a recognized Plaud domain. "
                f"Allowed: {', '.join(sorted(_ALLOWED_API_DOMAINS))}"
            )
        if parsed.path and parsed.path != "/":
            raise PlaudAPIError("API URL must not contain a path")
        if parsed.query or parsed.fragment:
            raise PlaudAPIError("API URL must not contain query or fragment")
        return url

    @staticmethod
    def _resolve_token(token: str | None) -> str:
        """Resolve token from multiple sources."""
        if token:
            return token.removeprefix("bearer ").removeprefix("Bearer ")

        env_token = os.environ.get("PLAUD_TOKEN", "")
        if env_token:
            return env_token.removeprefix("bearer ").removeprefix("Bearer ")

        # Check config file
        config_path = os.path.expanduser("~/.config/plaud/token")
        if os.path.isfile(config_path):
            # Warn if token file is readable by others
            try:
                file_mode = os.stat(config_path).st_mode
                if file_mode & (stat.S_IRGRP | stat.S_IROTH):
                    logger.warning(
                        "Token file %s is readable by other users (mode %o). "
                        "Run: chmod 600 %s",
                        config_path, file_mode & 0o777, config_path,
                    )
            except OSError:
                pass
            with open(config_path) as f:
                file_token = f.read().strip()
            if file_token:
                return file_token.removeprefix("bearer ").removeprefix("Bearer ")

        raise PlaudAuthError(
            "No Plaud token found. Set PLAUD_TOKEN env var, create "
            "~/.config/plaud/token, or pass token directly. "
            "Get your token from web.plaud.ai -> DevTools -> Network -> "
            "Authorization header."
        )

    def _build_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }

    def _request(
        self,
        method: str,
        path: str,
        *,
        _redirected: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Make an API request with error handling and retries."""
        last_error = None
        for attempt in range(3):
            try:
                response = self._client.request(method, path, **kwargs)
                if response.status_code == 401:
                    raise PlaudAuthError(
                        "Authentication failed. Your token may be expired. "
                        "Get a new one from web.plaud.ai.",
                        status_code=401,
                    )
                response.raise_for_status()
                data = response.json()

                # Handle region mismatch: API returns status -302
                # with the correct domain to use.
                # Only redirect once to prevent infinite loops.
                if isinstance(data, dict) and data.get("status") == -302:
                    if _redirected:
                        raise PlaudAPIError(
                            "Multiple API redirects detected; aborting."
                        )
                    correct_domain = data.get("domain", "")
                    if correct_domain and correct_domain in _ALLOWED_API_DOMAINS:
                        new_base = f"https://{correct_domain}"
                        old_client = self._client
                        self._client = httpx.Client(
                            base_url=new_base,
                            headers=self._build_headers(),
                            timeout=DEFAULT_TIMEOUT,
                        )
                        old_client.close()
                        self._base_url = new_base
                        return self._request(
                            method, path, _redirected=True, **kwargs
                        )
                    elif correct_domain:
                        logger.warning(
                            "Ignoring redirect to untrusted domain: %s",
                            correct_domain,
                        )

                return data
            except PlaudAuthError:
                raise
            except httpx.HTTPStatusError as e:
                if e.response.status_code >= 500 and attempt < 2:
                    last_error = e
                    continue
                # Sanitize: don't include full response body which may
                # contain sensitive data in error messages
                raise PlaudAPIError(
                    f"API error: HTTP {e.response.status_code}",
                    status_code=e.response.status_code,
                ) from None
            except httpx.RequestError as e:
                if attempt < 2:
                    last_error = e
                    continue
                # Sanitize: strip potential token/header info from error
                raise PlaudAPIError(
                    f"Request failed: {type(e).__name__}"
                ) from None
        raise PlaudAPIError(
            f"Request failed after retries: {type(last_error).__name__}"
        )

    def _get(self, path: str, **kwargs: Any) -> dict[str, Any]:
        return self._request("GET", path, **kwargs)

    def _post(self, path: str, **kwargs: Any) -> dict[str, Any]:
        return self._request("POST", path, **kwargs)

    # ── Recordings ──────────────────────────────────────────────

    _ALLOWED_SORT_FIELDS = frozenset({"edit_time", "start_time"})

    def list_recordings(
        self,
        limit: int = 100,
        skip: int = 0,
        sort_by: str = "edit_time",
        descending: bool = True,
    ) -> list[Recording]:
        """List all recordings in the account."""
        if sort_by not in self._ALLOWED_SORT_FIELDS:
            sort_by = "edit_time"
        data = self._get(
            "/file/simple/web",
            params={
                "skip": skip,
                "limit": limit,
                "is_trash": 0,
                "sort_by": sort_by,
                "is_desc": str(descending).lower(),
            },
        )
        # API returns data_file_list at the top level
        files = data.get("data_file_list", [])
        if not files:
            # Fallback for alternative response shapes
            files = data.get("data", [])
            if isinstance(files, dict):
                files = files.get("file_list", files.get("files", []))
        return [Recording.from_api(f) for f in files]

    def get_recording_detail(self, file_id: str) -> dict[str, Any]:
        """Get full detail for a recording including transcript and AI content.

        The Plaud `/file/detail/{id}` response (as of 2026-05) returns:
          - `file_name` (not `filename`)
          - `content_list[]` with `data_link` S3 pre-signed URLs to gzipped
            JSON for transcripts (`transaction`, `transaction_polish`) and
            summaries (`auto_sum_note`, `sum_multi_note`).
          - `pre_download_content_list[]` with inline `data_content` (JSON
            string) for the auto summary, so callers can avoid an extra
            S3 fetch for that one item.
        Older code paths expected `filename`, `trans_result`, `ai_content`
        at the top level — those keys no longer exist. We normalise the
        response so downstream code keeps working: we synthesise a
        `filename` alias and an `ai_content` field by parsing the
        pre_download payload.
        """
        file_id = self._validate_file_id(file_id)
        data = self._get(f"/file/detail/{file_id}")
        detail = data.get("data", data)
        if isinstance(detail, dict):
            # Normalise filename
            if "filename" not in detail and "file_name" in detail:
                detail["filename"] = detail["file_name"]
            # Lift inline summary out of pre_download_content_list
            if "ai_content" not in detail:
                pre_list = detail.get("pre_download_content_list") or []
                for item in pre_list:
                    dc = item.get("data_content") if isinstance(item, dict) else None
                    if not isinstance(dc, str):
                        continue
                    try:
                        parsed = _json.loads(dc)
                    except (ValueError, TypeError):
                        continue
                    if isinstance(parsed, dict) and parsed.get("ai_content"):
                        detail["ai_content"] = parsed["ai_content"]
                        break
        return detail

    def _fetch_s3_json(self, url: str) -> Any | None:
        """Fetch a Plaud S3 pre-signed URL, gunzip if needed, parse JSON.

        Returns the parsed JSON, or None on failure. Does not raise so
        callers can fall back gracefully.
        """
        try:
            # Use a fresh client with NO Authorization header — the
            # pre-signed URL carries its own auth in the query string,
            # and S3 rejects unexpected headers.
            with httpx.Client(timeout=DEFAULT_TIMEOUT) as c:
                resp = c.get(url)
                resp.raise_for_status()
                body = resp.content
            if url.endswith(".gz") or body[:2] == b"\x1f\x8b":
                body = gzip.decompress(body)
            return _json.loads(body)
        except (httpx.HTTPError, OSError, ValueError) as e:
            logger.warning("S3 fetch failed: %s", type(e).__name__)
            return None

    def _find_content_link(
        self, detail: dict[str, Any], data_types: tuple[str, ...]
    ) -> str | None:
        """Find the first ready S3 link in detail.content_list matching one
        of the given data_types, in priority order."""
        content_list = detail.get("content_list") or []
        # Build a lookup by data_type so we can honor priority order.
        by_type: dict[str, str] = {}
        for item in content_list:
            if not isinstance(item, dict):
                continue
            dt = item.get("data_type")
            link = item.get("data_link")
            status = item.get("task_status")
            # status==1 means ready; skip in-progress / failed items.
            if dt and link and status == 1 and dt not in by_type:
                by_type[dt] = link
        for dt in data_types:
            if dt in by_type:
                return by_type[dt]
        return None

    def get_audio_url(self, file_id: str) -> str:
        """Get a temporary download URL for the recording audio."""
        file_id = self._validate_file_id(file_id)
        data = self._get(f"/file/temp-url/{file_id}", params={"is_opus": 0})
        # API returns temp_url at top level
        return data.get("temp_url", data.get("data", {}).get("url", ""))

    # ── Transcripts ─────────────────────────────────────────────

    def get_transcript(self, file_id: str) -> Transcript:
        """Get the transcript for a recording.

        As of 2026-05 the transcript text lives in a gzipped JSON file on
        S3, referenced by `content_list[].data_link` with `data_type` of
        `transaction_polish` (speaker-cleaned, preferred) or
        `transaction` (raw fallback). Each segment is shaped:
            {start_time, end_time, content, speaker, original_speaker}
        Older code paths expected `trans_result.segments` inline in the
        detail response — that no longer exists.
        """
        file_id = self._validate_file_id(file_id)
        detail = self.get_recording_detail(file_id)

        segments: list[TranscriptSegment] = []

        # Legacy path (kept for backwards compat if Plaud reverts shape):
        trans_result = detail.get("trans_result", {})
        if isinstance(trans_result, dict):
            raw_segments = trans_result.get("segments", trans_result.get("result", []))
            segments = [TranscriptSegment.from_api(s) for s in raw_segments]
        elif isinstance(trans_result, list):
            segments = [TranscriptSegment.from_api(s) for s in trans_result]

        # New path: fetch from S3.
        if not segments:
            link = self._find_content_link(
                detail, ("transaction_polish", "transaction")
            )
            if link:
                payload = self._fetch_s3_json(link)
                raw_segments = []
                if isinstance(payload, list):
                    raw_segments = payload
                elif isinstance(payload, dict):
                    raw_segments = (
                        payload.get("segments")
                        or payload.get("result")
                        or []
                    )
                # Each segment uses {start_time, end_time, content, speaker}.
                # TranscriptSegment.from_api looks for "text"/"start_time_ms"
                # — adapt the keys here.
                adapted = []
                for s in raw_segments:
                    if not isinstance(s, dict):
                        continue
                    adapted.append({
                        "text": s.get("content", s.get("text", "")),
                        "speaker": s.get("speaker", s.get("spk", "")),
                        "start_time_ms": s.get(
                            "start_time", s.get("start_time_ms", 0)
                        ),
                        "end_time_ms": s.get(
                            "end_time", s.get("end_time_ms", 0)
                        ),
                    })
                segments = [TranscriptSegment.from_api(s) for s in adapted]

        return Transcript(file_id=file_id, segments=segments)

    # ── AI Summaries ────────────────────────────────────────────

    def get_summary(self, file_id: str) -> str:
        """Get the AI-generated summary for a recording.

        get_recording_detail lifts the inline pre-downloaded summary into
        `ai_content` automatically. If that's empty (e.g. summary too
        large for inline payload), fall back to fetching the
        `auto_sum_note` / `sum_multi_note` S3 link.
        """
        file_id = self._validate_file_id(file_id)
        detail = self.get_recording_detail(file_id)
        ai_content = detail.get("ai_content", "")
        if isinstance(ai_content, dict):
            return ai_content.get(
                "content", ai_content.get("summary", str(ai_content))
            )
        if ai_content:
            return str(ai_content)

        # Fallback: fetch from S3.
        link = self._find_content_link(
            detail, ("auto_sum_note", "sum_multi_note")
        )
        if link:
            payload = self._fetch_s3_json(link)
            if isinstance(payload, dict):
                return str(
                    payload.get("ai_content")
                    or payload.get("content")
                    or payload.get("summary")
                    or ""
                )
        return ""

    def get_notes(self, file_id: str) -> str:
        """Get AI-generated notes for a recording."""
        file_id = self._validate_file_id(file_id)
        try:
            data = self._get("/ai/query_note", params={"file_id": file_id})
            return data.get("data", {}).get("content", "")
        except PlaudAPIError:
            return ""

    # ── Tags ────────────────────────────────────────────────────

    def list_tags(self) -> list[Tag]:
        """List all tags/folders."""
        data = self._get("/filetag/")
        # API returns data_filetag_list at top level
        tags_data = data.get("data_filetag_list", data.get("data", []))
        return [
            Tag(
                tag_id=t.get("id", t.get("tag_id", "")),
                name=t.get("name", ""),
                count=t.get("file_count", t.get("count", 0)),
            )
            for t in tags_data
        ]

    # ── Speakers ────────────────────────────────────────────────

    def list_speakers(self) -> list[dict[str, Any]]:
        """List all known speakers."""
        data = self._get("/speaker/list")
        return data.get("data_speaker_list", data.get("data", []))

    # ── User & Devices ─────────────────────────────────────────

    def get_user_info(self) -> dict[str, Any]:
        """Get authenticated user account info."""
        data = self._get("/user/me")
        return data.get("data", data)

    def list_devices(self) -> list[dict[str, Any]]:
        """List all bound Plaud devices."""
        data = self._get("/device/list")
        return data.get("data", [])

    # ── Batch Operations ───────────────────────────────────────

    def get_batch_details(self, file_ids: list[str]) -> list[dict[str, Any]]:
        """Batch fetch details for multiple recordings."""
        data = self._post("/file/list", json=file_ids, timeout=60)
        return data.get("data", [])

    def get_recent_context(
        self,
        count: int = 5,
    ) -> list[dict[str, Any]]:
        """Get transcripts and summaries for the most recent recordings.

        Returns a list of dicts with recording metadata, transcript text,
        and AI summary for each of the most recent recordings.
        """
        recordings = self.list_recordings(limit=count, sort_by="start_time")
        results = []
        for rec in recordings:
            entry: dict[str, Any] = {
                "file_id": rec.file_id,
                "filename": rec.filename,
                "duration": rec.duration_str,
                "created": rec.created_at.isoformat() if rec.created_at else "unknown",
            }
            try:
                # Use get_transcript / get_summary which know about the
                # 2026-05 API shape (S3 links + pre_download_content_list).
                transcript = self.get_transcript(rec.file_id)
                if transcript.segments:
                    entry["transcript"] = transcript.full_text
                summary = self.get_summary(rec.file_id)
                if summary:
                    entry["summary"] = summary
            except PlaudAPIError:
                entry["error"] = "Could not fetch details"

            results.append(entry)
        return results

    def get_recordings_by_tag(self, tag_id: str) -> list[Recording]:
        """Get all recordings that have a specific tag."""
        all_recordings = self.list_recordings(limit=500)
        return [rec for rec in all_recordings if tag_id in rec.tags]

    # ── Search (client-side) ────────────────────────────────────

    def search_recordings(
        self,
        query: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Search recordings by matching query against filenames,
        transcripts, and summaries. This is a client-side search
        since Plaud doesn't provide a server-side search endpoint.
        """
        recordings = self.list_recordings(limit=limit)
        query_lower = query.lower()
        results = []

        for rec in recordings:
            # Check filename
            if query_lower in rec.filename.lower():
                results.append({
                    "recording": rec,
                    "match_type": "filename",
                    "snippet": rec.filename,
                })
                continue

            # Check transcript and summary using new-shape fetchers.
            try:
                transcript = self.get_transcript(rec.file_id)
                trans_text = transcript.text_only
                if query_lower in trans_text.lower():
                    idx = trans_text.lower().index(query_lower)
                    start = max(0, idx - 100)
                    end = min(len(trans_text), idx + len(query) + 100)
                    snippet = trans_text[start:end]
                    results.append({
                        "recording": rec,
                        "match_type": "transcript",
                        "snippet": f"...{snippet}...",
                    })
                    continue

                summary_text = self.get_summary(rec.file_id)
                if query_lower in summary_text.lower():
                    idx = summary_text.lower().index(query_lower)
                    start = max(0, idx - 100)
                    end = min(len(summary_text), idx + len(query) + 100)
                    snippet = summary_text[start:end]
                    results.append({
                        "recording": rec,
                        "match_type": "summary",
                        "snippet": f"...{snippet}...",
                    })
            except PlaudAPIError:
                continue

        return results

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> PlaudClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
