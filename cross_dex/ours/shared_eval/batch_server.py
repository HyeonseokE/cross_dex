"""Common part of the shared-model evaluation servers (DP: dp/shared_server.py, pi0.5: pi05/openpi_server.py).

One process holds the model; any number of evaluation sessions connect to one port (multiprocessing.connection,
localhost). Per connection the protocol is the evaluator's:
  ("reset", None | dict)  -> backend.reset() reply ("ok")   a new episode of this session
  (<request>, message)    -> backend.reply() payload        e.g. "act" (DP) / "infer" (pi0.5)
  ("close", None)         -> (no reply)                     the session ends
A request the backend can answer at once (observe() -> False, e.g. a DP action still in the queue) is answered at once.
Requests that need the model (observe() -> True) are gathered and run as ONE batched backend.forward(sessions):
everything already waiting is taken, then up to --gather_ms more is waited for, or until every connected session waits
(--gather_ms ~1000 = lockstep: every forward batches all active sessions; ~5 = greedy). Batches are cut at --max_batch.

Per-session randomness is the backend's job (each session keeps its own generator / key, so a session's samples do not
depend on who shares its batch); episode_seed() is the seed convention both backends use for per-episode seeds.
Timing log (--log, jsonl, read by dp/summarize.py and pi05/summarize.py):
  {"kind": "call", sid, ep, call, fwd, batch, wait_ms}   one per request (wait_ms = arrival -> reply sent)
  {"kind": "forward", t, batch, ms, sids}                 one per batched model call
  {"kind": "memory", ...backend.memory()}                 at exit

Backend interface (all calls from the serving thread, never concurrently):
  reset(session, message) -> reply        observe(session, message) -> bool (True: needs forward)
  forward(sessions)                       reply(session) -> payload
  synchronize()  (before/after forward, for timing)      memory() -> dict      opened(session) / closed(session) (optional)
Dependencies: stdlib (numpy only inside fresh_seed); importable in every env (Isaac main, lerobot, openpi, dexlatent).
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
from multiprocessing.connection import Listener

EPISODE_STRIDE = 100000


def episode_seed(seed: int, episode: int) -> int:
    """Noise seed of evaluator episode `episode` (pickle index) in a run with base seed `seed`."""
    return int(seed) * EPISODE_STRIDE + int(episode)


def fresh_seed() -> int:
    import numpy as np  # lazy: this module is imported before Isaac Sim starts (dp/run_eval_dp.py)

    return int(np.random.SeedSequence().entropy % (1 << 63))


def add_server_args(parser, gather_ms: float = 5.0, max_batch: int = 16):
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--authkey", default=os.environ.get("SHARED_EVAL_AUTHKEY"),
                        help="Connection key (default $SHARED_EVAL_AUTHKEY).")
    parser.add_argument("--sessions", type=int, default=0, help="Exit after this many sessions closed (0 = run until killed).")
    parser.add_argument("--gather_ms", type=float, default=gather_ms,
                        help="Wait this long for more requests to batch (~5 greedy, ~1000 lockstep).")
    parser.add_argument("--max_batch", type=int, default=max_batch)
    parser.add_argument("--log", default=None, help="jsonl with one line per call and per batched forward.")
    parser.add_argument("--exit_with_pid", type=int, default=0, help="Exit when this process is gone (the manager).")
    return parser


class Session:
    def __init__(self, sid: int, conn):
        self.sid, self.conn = sid, conn
        self.episode, self.call = -1, 0


class BatchServer:
    def __init__(self, backend, args, tag: str = "shared"):
        if not args.authkey:
            raise SystemExit("--authkey or $SHARED_EVAL_AUTHKEY required")
        self.backend, self.args, self.tag = backend, args, tag
        self.requests: queue.Queue = queue.Queue()
        self.active, self.lock = set(), threading.Lock()
        self.next_sid, self.closed = 0, 0
        self.log = open(args.log, "a", buffering=1 << 16) if args.log else None

    def write_log(self, **row):
        if self.log:
            self.log.write(json.dumps(row) + "\n")

    def say(self, text):
        print(f"[{self.tag}] {text}", flush=True)

    # ---------------------------------------------------------------- connections (threads)
    def _reader(self, session):
        try:
            while True:
                kind, message = session.conn.recv()
                self.requests.put((session, kind, message, time.perf_counter()))
                if kind == "close":
                    return
        except (EOFError, OSError):
            self.requests.put((session, "close", None, time.perf_counter()))

    def _acceptor(self, listener):
        while True:
            try:
                conn = listener.accept()
            except Exception as error:  # wrong authkey etc.: keep serving the others
                self.say(f"accept failed: {error!r}")
                continue
            with self.lock:
                session = Session(self.next_sid, conn)
                self.next_sid += 1
            getattr(self.backend, "opened", lambda s: None)(session)
            with self.lock:
                self.active.add(session)
                count = len(self.active)
            self.say(f"session {session.sid} connected ({count} active)")
            threading.Thread(target=self._reader, args=(session,), daemon=True).start()

    def _guard(self, pid):
        while True:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                self.say(f"process {pid} gone, exiting")
                os._exit(0)
            except PermissionError:
                pass
            time.sleep(5)

    # ---------------------------------------------------------------- serving thread
    def _send(self, session, payload):
        try:
            session.conn.send(payload)
        except (OSError, EOFError, BrokenPipeError):  # the session died; its "close" follows from the reader
            pass

    def _reply(self, session, arrived, fwd, batch):
        self._send(session, self.backend.reply(session))
        session.call += 1
        self.write_log(kind="call", sid=session.sid, ep=session.episode, call=session.call - 1, fwd=fwd, batch=batch,
                       wait_ms=(time.perf_counter() - arrived) * 1e3)

    def handle(self, item, pending):
        session, kind, message, arrived = item
        if kind == "close":
            with self.lock:
                if session in self.active:
                    self.active.discard(session)
                    self.closed += 1
                count = len(self.active)
            getattr(self.backend, "closed", lambda s: None)(session)
            self.say(f"session {session.sid} closed ({count} active)")
            return
        if kind == "reset":
            session.episode, session.call = session.episode + 1, 0
            self._send(session, self.backend.reset(session, message))
            return
        if self.backend.observe(session, message):
            pending.append(item)
        else:
            self._reply(session, arrived, False, 0)

    def run_batches(self, pending):
        size = self.args.max_batch
        for i in range(0, len(pending), size):
            chunk = pending[i:i + size]
            sessions = [s for s, *_ in chunk]
            self.backend.synchronize()
            t0 = time.perf_counter()
            self.backend.forward(sessions)
            self.backend.synchronize()
            ms = (time.perf_counter() - t0) * 1e3
            self.write_log(kind="forward", t=time.time(), batch=len(chunk), ms=ms, sids=[s.sid for s in sessions])
            for session, _, _, arrived in chunk:
                self._reply(session, arrived, True, len(chunk))

    def step(self, timeout=1.0) -> bool:
        """Serve one round (one request, whatever else is waiting, the gather window, the forwards). False if idle."""
        try:
            item = self.requests.get(timeout=timeout)
        except queue.Empty:
            return False
        pending = []
        self.handle(item, pending)
        while True:  # whatever is already waiting
            try:
                self.handle(self.requests.get_nowait(), pending)
            except queue.Empty:
                break
        if not pending:
            return True
        deadline = time.perf_counter() + self.args.gather_ms / 1e3
        while True:
            with self.lock:
                target = min(len(self.active), self.args.max_batch)
            if len(pending) >= target:
                break
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            try:
                self.handle(self.requests.get(timeout=remaining), pending)
            except queue.Empty:
                break
        self.run_batches(pending)
        return True

    def serve(self):
        if self.args.exit_with_pid:
            threading.Thread(target=self._guard, args=(self.args.exit_with_pid,), daemon=True).start()
        with Listener((self.args.host, self.args.port), authkey=self.args.authkey.encode()) as listener:
            threading.Thread(target=self._acceptor, args=(listener,), daemon=True).start()
            self.say(f"listening on {self.args.host}:{self.args.port}")
            while not (self.args.sessions and self.closed >= self.args.sessions):
                self.step()
        self.write_log(kind="memory", **self.backend.memory())
        if self.log:
            self.log.close()
        self.say("all sessions closed")
