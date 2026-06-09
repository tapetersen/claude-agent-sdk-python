"""Regression tests for the pipe-I/O deadlock on Query.close().

Bug:
    When the consumer stops reading (e.g. breaks out of the async-for loop),
    _read_messages was exiting immediately on the next iteration because
    self._closed was True and it hit ``break``.  transport.close() would then
    close stdin, the subprocess would try to dump its final output into a full
    pipe with nobody reading, block in write(), and process.wait() would hang
    for the full 5 s + SIGKILL budget.

Fix:
    _read_messages now ``continue``s instead of ``break``ing when self._closed
    is True, so it keeps draining stdout while transport.close() waits for the
    subprocess to exit.  transport.close() also defers _stdout_stream.aclose()
    until after process.wait() so the drain is not interrupted mid-flight.

Refs:
    - GitHub issue: #728

These tests stand up a real fake-CLI as a Python subprocess and drive it
through the Query layer — the same path taken by real callers.
"""

from __future__ import annotations

import stat
import sys
import textwrap
import time
from pathlib import Path

import anyio
import pytest

from claude_agent_sdk._internal.query import Query
from claude_agent_sdk._internal.transport.subprocess_cli import SubprocessCLITransport
from claude_agent_sdk.types import ClaudeAgentOptions


@pytest.fixture(autouse=True)
def _skip_version_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bypass ``--version`` probe so the fake CLI is not asked to advertise."""
    monkeypatch.setenv("CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK", "1")


# Wall-clock budget that Query.close() must hit even under the pathological
# pipe-full scenario.
CLOSE_DEADLINE_SECONDS = 2.0


def _write_fake_cli(tmp_path: Path, body: str) -> Path:
    """Write an executable Python script that masquerades as ``claude``."""
    cli = tmp_path / "fake_claude"
    cli.write_text(
        textwrap.dedent(
            """\
            #!{python}
            import os, signal, sys, time
            """
        ).format(python=sys.executable)
        + body
    )
    cli.chmod(stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
    return cli


FAKE_CLI_PIPE_FULL_IGNORES_SIGTERM = textwrap.dedent(
    """
    # Ignore SIGTERM so only the drain (not a signal) can unblock us.
    signal.signal(signal.SIGTERM, signal.SIG_IGN)

    # Emit a tiny "ready" line so _read_messages has something to consume.
    sys.stdout.write('{"type":"system","subtype":"init"}\\n')
    sys.stdout.flush()

    # Block until stdin EOF — the SDK signals shutdown by closing stdin.
    try:
        sys.stdin.read()
    except Exception:
        pass

    # After stdin EOF, dump enough data to overfill the 64 KB kernel pipe
    # buffer many times over.  With the bug, nobody reads this and write()
    # blocks in the kernel; with the fix, _read_messages drains it.
    payload = ('{"type":"text","text":"' + ('x' * 1023) + '"}\\n') * 256  # ~256 KB
    try:
        sys.stdout.write(payload)
        sys.stdout.flush()
    except BrokenPipeError:
        pass
    sys.exit(0)
    """
)


FAKE_CLI_QUICK_EXIT = textwrap.dedent(
    """
    sys.stdout.write('{"type":"text","text":"hi"}\\n')
    sys.stdout.flush()
    try:
        sys.stdin.read()
    except Exception:
        pass
    sys.exit(0)
    """
)


def _make_query(cli: Path) -> tuple[SubprocessCLITransport, Query]:
    transport = SubprocessCLITransport(
        prompt="hello",
        options=ClaudeAgentOptions(cli_path=str(cli)),
    )
    query = Query(transport=transport, is_streaming_mode=False)
    return transport, query


class TestSubprocessCLICloseDoesNotDeadlock:
    """Regression suite for the pipe-I/O deadlock on Query.close()."""

    def test_close_returns_promptly_when_subprocess_ignores_sigterm_and_pipe_is_full(
        self, tmp_path: Path
    ) -> None:
        """Query.close() must not hang when the subprocess floods stdout.

        Scenario: consumer never reads, subprocess dumps >64 KB on stdin EOF,
        ignores SIGTERM.  Without the fix: ~10 s (5 s graceful + 5 s SIGTERM).
        With the fix: well under a second (_read_messages drains the pipe).
        """
        cli = _write_fake_cli(tmp_path, FAKE_CLI_PIPE_FULL_IGNORES_SIGTERM)

        async def _run() -> float:
            transport, query = _make_query(cli)
            await transport.connect()
            await query.start()
            # Do NOT consume any messages — simulates a caller that breaks out
            # of the async-for before the subprocess has exited.
            t0 = time.monotonic()
            await query.close()
            return time.monotonic() - t0

        elapsed = anyio.run(_run)

        assert elapsed < CLOSE_DEADLINE_SECONDS, (
            f"Query.close() took {elapsed:.2f}s — pipe-I/O deadlock fix "
            f"is missing or regressed. Expected < {CLOSE_DEADLINE_SECONDS}s."
        )

    def test_close_happy_path_still_fast(self, tmp_path: Path) -> None:
        """Verify the fix does not regress the trivial-exit happy path."""
        cli = _write_fake_cli(tmp_path, FAKE_CLI_QUICK_EXIT)

        async def _run() -> float:
            transport, query = _make_query(cli)
            await transport.connect()
            await query.start()
            await anyio.sleep(0.1)
            t0 = time.monotonic()
            await query.close()
            return time.monotonic() - t0

        elapsed = anyio.run(_run)

        assert elapsed < CLOSE_DEADLINE_SECONDS, (
            f"Query.close() regressed on the happy path: {elapsed:.2f}s"
        )

    def test_close_is_idempotent(self, tmp_path: Path) -> None:
        """Calling Query.close() twice must not raise."""
        cli = _write_fake_cli(tmp_path, FAKE_CLI_QUICK_EXIT)

        async def _run() -> None:
            transport, query = _make_query(cli)
            await transport.connect()
            await query.start()
            await anyio.sleep(0.1)
            await query.close()
            await query.close()

        anyio.run(_run)
