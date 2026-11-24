---
description: "Full walkthrough of Qwen Live realtime voice sessions over WebSocket with the dashscope SDK."
---
# Qwen Live Python SDK

Qwen Live is a WebSocket model for realtime voice interaction: the client
streams audio upstream continuously, and the server returns audio, transcripts
and usage events. The interface lives at `client.audio.live` and provides
connection management, session configuration, audio upstream, delegation
write-back and event iteration.

This guide shows how to use the Python SDK with Qwen Live to complete the
minimal loop of "send audio → receive and save the reply", plus text input and
delegation modes. Qwen Live uses the `session.*` event family, which is a
different protocol from Omni / Qwen-Audio's `input_audio_buffer.*` and
`response.*`.

---

## Prerequisites

- **Install the SDK**: Python 3.9 or newer is required, with a DashScope 2.1
  SDK that includes `client.audio.live`. This guide uses the `dashscope` and
  `dashscope.resources.audio` import paths, not `dashscope.v2`.

  ```bash
  python -m pip install -U "dashscope>=2.1.0"
  python -c "from dashscope import Client; from dashscope.resources.audio import LiveSession"
  ```

  If `LiveSession` still cannot be imported after upgrading, use the
  Live-capable SDK version specified by your service provider.

- **Configure the API key**: the `DASHSCOPE_API_KEY` environment variable is
  recommended; never hard-code keys. The Client resolves configuration in the
  order `explicit arguments → process-level settings → environment variables →
  config.toml → key file`; the key must have access to the target environment.

  ```bash
  export DASHSCOPE_API_KEY="<your-api-key>"
  export DASHSCOPE_WORKSPACE="<your-workspace-id>"
  ```

- **Confirm the endpoint and models**: Qwen Live uses a workspace-dedicated
  gateway. The default URL template is

  ```
  wss://{workspace}.{region}.maas.aliyuncs.com/api-ws/v1/realtime
  ```

  `region` defaults to `cn-beijing`; `ap-southeast-1`, `cn-hongkong`,
  `us-east-1`, `eu-central-1` and `ap-northeast-1` are also supported. Without
  a workspace the SDK falls back to the region's shared gateway
  (`wss://dashscope.aliyuncs.com` for Beijing) and prints a `UserWarning`.

  > **Important**: the SDK automatically appends `?model=<route model>` to the
  > URL; if `ws_url` already carries a `model` query parameter, the existing
  > value wins. For private or pre-release deployments, pass the full address
  > explicitly via `ws_url`. The route model name (`connect`) and the session
  > model name (`update_session`) depend on what your environment actually
  > provides.

---

## Request parameters

Parameters come in three layers: **connection parameters** decide where to
connect and how to authenticate; **session configuration** is sent as the
`session.update` payload and decides session behavior; **delegation
configuration** decides who produces the reply. The route model, frontend
model and backend model belong to these three layers respectively — do not
reuse one name across them.

### 1. Connection parameters `client.audio.live.connect(...)`

Returns a `LiveSession` that is not yet connected; the handshake happens only
when you call `connect()` or enter the `with` context.

| Parameter | Type | Required | Description |
| --- | --- | --- | --- |
| `model` | `str` | Yes | Gateway route model name, written into the URL's `?model=`. |
| `ws_url` | `str` | No | Explicit full address; takes precedence over the template. |
| `timeout` | `float` | No | Handshake and per-socket send/receive timeout (seconds); must be positive. Defaults to the Client config when omitted; it is not a cap on the whole session duration. |
| `headers` | `Mapping[str, str]` | No | Custom handshake headers; passed values win on name conflicts. |
| `api_key` | `str` | No | Key for this connection, sent as `Authorization: Bearer`; when omitted or empty, the Client's resolved key is used. |
| `workspace` | `str` | No | Workspace ID; defaults to the Client config. Affects the default URL and the `X-DashScope-WorkSpace` header; an explicit `ws_url` is never rewritten. |
| `connector` | `Callable` | No | Custom connection factory, for tests or special transports only. |

```python
import os
from dashscope import Client

with Client(region="cn-beijing") as client:
    session = client.audio.live.connect(
        model=os.getenv("LIVE_ROUTE_MODEL", "qwen-live"),
        ws_url=os.getenv("LIVE_WS_URL"),
        timeout=30,
    )  # Not yet handshaken; see the full example below
```

### 2. Session configuration `update_session(...)`

Serialized into the `session` node of `session.update`; only explicitly passed
fields are sent. After the server validates, prepares and commits the config
it returns `session.updated` (first success has `revision=1`); on failure no
new config is committed and only an `error` is returned.

| Parameter | Type | Required | Description |
| --- | --- | --- | --- |
| `model` | `str` | Yes | Session (frontend) model name, e.g. `audio-live`, `qwen-audio-realtime`; available names depend on the deployment — the SDK has no model whitelist. |
| `delegation` | `Dict` | Yes | Delegation configuration, see below. |
| `instructions` | `str` | No | System prompt for the frontend model. |
| `voice` | `str` | No | Voice; enum `台湾女声`, `星悦`, `古风女声`, `古风男声`; the deployment default is used when omitted. |
| `tts` | `Dict` | No | Voice configuration for Audio Live; only `system_prompt` is supported. |
| `input_audio_format` | `str` | No | Upstream audio format; only `pcm` is accepted. |
| `output_audio_format` | `str` | No | Downstream audio format; only `pcm` is accepted. |
| `modalities` | `List[str]` | No | Output modalities, default `["text", "audio"]`. |
| `tools` | `List[Dict]` | No | Frontend function tools, default empty array. |
| `context` | `Dict` | No | Context node; actually required by the server, see the note below. |
| `turn_detection` | `Dict` or `None` | No | Turn-detection config; omitting differs from explicit `None` (the latter sends JSON `null`); available options depend on the frontend model. |
| `input_audio_transcription` | `Dict` or `None` | No | Input transcription config; explicit `None` is sent as JSON `null` verbatim. |
| `transport` | `Dict` | No | Omitted or empty object means the default WebSocket; the only explicit transport implemented is `aoq`. |
| `parameters` | `Dict` | No | Pass-through container for new server-side fields, merged key-by-key into the `session` node; cannot override the named fields above. |
| `event_id` | `str` | No | Client event ID for this request, 1–128 characters; generated by the SDK when omitted. Other send methods support it too. |

> **Note**: the server requires `context.compaction` to be configured,
> otherwise it returns `invalid_session_configuration` with the message
> `session.update: context_compaction_not_configured`. Even when compaction is
> disabled it must be sent explicitly:
> `{"compaction": {"enabled": False, "mode": "local"}}`. When setting
> `tts.system_prompt` for `audio-live` explicitly, it must start with the exact
> sentence `You are a helpful assistant and asked to generate both text and
> speech tokens at the same time.`; if you do not customize it, omit `tts` —
> the SDK does not add the prefix for you.

Invalid delegation types, an empty backend model, non-`pcm` audio formats and
similar cases raise `ValueError` locally in the SDK without any network
request.

```python
session.update_session(
    model="audio-live",
    instructions="Answer the user's voice questions concisely",
    input_audio_format="pcm",
    output_audio_format="pcm",
    context={"compaction": {"enabled": False, "mode": "local"}},
    delegation={"type": "none"},
)
```

### 3. Delegation configuration `delegation`

Decides "who produces the reply".

| Parameter | Type | Required | Description |
| --- | --- | --- | --- |
| `type` | `str` | Yes | `none` (no delegation), `client` (external agent), `responses` (backend model). |
| `responses` | `Dict` | Required in `responses` mode | Backend model configuration; must not be present in other modes. |
| `responses.model` | `str` | Yes | Backend model name; must match the current deployment — use what your environment provides. |
| `responses.instructions` | `str` | No | Backend model prompt. |
| `responses.tools` | `List[Dict]` | No | Tools available to the backend model. |

- **`none`**: the frontend model produces replies itself; the client only
  streams audio upstream.
- **`client`**: the server produces no reply; instead it emits
  `session.delegation.created`, and your external agent writes content back
  via `append_instruction` / `append_thinking` / `append_commentary`, which the
  frontend model then renders as speech.
- **`responses`**: the backend model produces replies directly; the client
  only streams audio upstream (optionally use `create_response_item` /
  `create_response` to add input or request a reply proactively).

See example 2 for configuration; the SDK only sends and receives events — it
never runs external agents or tools automatically.

---

## Key interface `LiveSession`

A single instance can hold only one open connection at a time; after close
completes you may explicitly `connect()` again — the SDK clears local session
state but does not restore the old session or replay audio. All send methods
return **the `event_id` of this request**, not a server acknowledgment.

| Method / property | Server event triggered | Description |
| --- | --- | --- |
| `connect()` | `session.created` | Opens the WebSocket; `with` opens and releases it automatically. Only means the connection is open — **not that the session is configured**. |
| `update_session(...)` | `session.updated` | Sends the session configuration. |
| `append_audio(audio, *, event_id=None)` | — | Appends an audio frame; accepts `bytes` or Base64 `str`; empty data or an odd byte count raises `ValueError`. |
| `mute_input_audio()` / `unmute_input_audio()` | `session.input_audio.muted` / `.unmuted` | Mutes / unmutes upstream; audio from the muted period is not replayed after unmuting. |
| `append_commentary(content, *, delegation_id=None, ...)` | `session.commentary.appended` | Without `delegation_id`, sends user text; with `delegation_id`, writes back the task result of a `client` delegation. |
| `append_instruction(content, *, delegation_id, ...)` | `session.instructions.appended` | Writes a prompt back to the given delegation (`client` mode). |
| `append_thinking(content, *, delegation_id, ...)` | `session.thinking.appended` | Writes back silent reasoning context; speech output is not guaranteed. |
| `create_session_item(item=omitted)` | — | Sends `session.item.create`; omitting `item` omits the field, which differs from passing `None` explicitly. |
| `submit_function_output(call_id, output)` | — | Answers a frontend function call inside `session.event`; not for backend tools. |
| `create_response_item(item)` | — | Appends an input item to the Responses backend; does not trigger a reply. |
| `create_response(response=None)` | Depends on mode | Starts a backend task in `responses` mode; requests an answer from frontends that support proactive replies. See example 2 for plain-text usage. |
| `end_session()` | `session.closed` | Sends the business event `session.close`; no further input is allowed afterwards — keep receiving until `session.closed`. Repeated calls while the close request is pending return the same request ID. |
| `close()` | — | Waits up to 1 second for in-flight sends, then sends a standard WebSocket Close (1000) and independently waits up to 1 second for the handshake; either stage timing out forces release, with no guarantee the peer received the Close frame. Does not wait for the business event `session.closed`; repeated calls are safe. |
| `cancel()` | — | Immediately aborts and releases the socket without a close handshake. Corresponds to the Java SDK's `cancel()`. |
| `send_event(event)` | Depends on payload | Low-level entry point for sending any protocol event; `type` must be a non-blank string, `event_id` is auto-generated when omitted. |
| `model` / `url` | — | Route model name / actual handshake URL (including `?model=`). |
| `session_id` / `close_reason` | — | `session.id` from `session.created` / `reason` from `session.closed`; `None` until received. |
| `last_message` | — | The most recent server event. |

> **Audio requirements**: upstream is 16 kHz, mono, 16-bit signed
> little-endian raw PCM (PCM16LE); downstream is 24 kHz, mono, PCM16LE, Base64
> encoded, with no WAV / container header. The SDK does not resample, does not
> recognize WAV containers and does not change byte order — strip WAV headers
> first. Wait for `session.updated` before streaming: 3200 bytes per frame
> (100 ms), wait 100 ms after each `append_audio()` returns; after the delay
> **do not catch up by bursting and do not drop input frames** — the actual
> send rate is slightly below 1x. Pacing is the application's responsibility;
> the SDK does not rate-limit. After the file ends, keep sending silence
> frames to maintain `audio-live`'s duplex audio clock until the application
> decides to end the session.

---

## Event iteration & error contract

Consume events on one receiving thread while audio goes upstream on another;
the main thread can also receive, as in the full example. The SDK locks the
send and receive directions separately and never creates background threads
automatically. Do not consume the same connection with both `events()` and
`recv_event()`.

| Method | Description |
| --- | --- |
| `events()` | Iterator that keeps yielding events until a normal close. Recoverable `error` events are yielded without raising; fatal errors raise. |
| `recv_event()` | Reads one event; server errors always raise, only recoverable errors keep the connection. Returns `None` after a normal close. |

> **Error contract**: `error.type == "invalid_request_error"` is a recoverable
> business error; other business errors close the connection. Unexpected
> disconnects, timeouts and invalid events raise `APIConnectionError`,
> `APITimeoutError` and `WsEventError` respectively; handshake HTTP errors map
> to the corresponding exceptions (401 → `AuthenticationError`, 429 →
> `RateLimitError`). All of these inherit `DashScopeError`; WebSocket business
> errors have `status_code == None`. Fatal failures are remembered — later
> receives keep raising the same exception, and `close()` does not clear the
> failure; after a local close / cancel with no recorded failure, receives
> return `None`. `KeyboardInterrupt` and `SystemExit` propagate unchanged.

```python
from dashscope import DashScopeError

try:
    for event in session.events():
        if event["type"] == "error":
            print(event["error"])  # Recoverable; decide whether to continue
except DashScopeError as exc:
    print(f"[error] {exc}")
```

### Server event types

Distinguish them by the `type` field in the event loop. Event-name constants
are importable from `dashscope.resources.audio` or
`dashscope.resources.audio.live`, e.g. `EVENT_SESSION_UPDATED`,
`EVENT_SESSION_OUTPUT_AUDIO_DELTA`, matching the literals in the table below:

| Event `type` | Meaning |
| --- | --- |
| `session.created` | Session created, with `session.id` and server defaults; one connection may receive several (one from the routing layer, one from the frontend). |
| `session.updated` | Session configuration took effect (**wait for this event before streaming audio**), with the full config and `revision`. |
| `session.input_transcript.delta` | Transcript chunk of the user's upstream speech; timestamps follow the server's actual output. |
| `session.output_transcript.delta` | Assistant reply text chunk. |
| `session.output_audio.delta` | Downstream audio chunk; `delta` is Base64 raw PCM16LE. |
| `session.output_audio.interrupted` | The current downstream audio was interrupted — **not** an end-of-reply signal. |
| `session.commentary.appended` / `session.instructions.appended` / `session.thinking.appended` | Acknowledgments of delegation write-back content. |
| `session.input_audio.muted` / `session.input_audio.unmuted` | Mute-state acknowledgments. |
| `session.delegation.created` | The frontend dispatches a new task or supplement, with `delegation_id`, `offset_ms`, `summary`, `context`; sent in both `client` and `responses` modes. |
| `session.event` | Nested events from the frontend model; function calls inside can be answered with `submit_function_output`. |
| `response.event` | Native events of the backend Responses model; observe backend token usage here. |
| `session.usage.updated` | Usage updates, e.g. cumulative audio duration `usage.seconds`; reporting cadence and fields follow the server. |
| `session.closed` | Session closed, with `reason` (`close_requested` / `expired` / `content` / `remote_hangup` / `connection_lost`) and final `usage`. |
| `error` | Server error, with `error.type` (`invalid_request_error` / `server_error`), `error.code`, `error.message`. |

The protocol may add new event types; ignore unknown `type` values instead of
raising.

---

## Full examples

### Example 1: minimal loop (voice file input)

Save as `qwen_live_quickstart.py`. Streams one PCM / WAV file upstream, prints
transcripts in both directions and saves the reply. After the file ends, the
sender thread keeps streaming silence and independently checks whether the
reply has gone quiet; the main thread keeps receiving until `session.closed`.
This example does not capture a microphone or play audio automatically.

```python
import base64
import os
import sys
import time
import wave
from array import array
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, Lock

from dashscope import Client

FRONTEND_MODEL = os.getenv("LIVE_FRONTEND_MODEL", "audio-live")
FRAME_BYTES = 3200
IDLE_SECONDS = 5.0
REPLY_TIMEOUT = 90.0
CLOSE_TIMEOUT = 5.0


def read_pcm(path):
    path = Path(path)
    if path.suffix.lower() == ".pcm":
        pcm = path.read_bytes()
    elif path.suffix.lower() == ".wav":
        with wave.open(str(path), "rb") as audio:
            if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate(),
                    audio.getcomptype()) != (1, 2, 16000, "NONE"):
                raise ValueError("WAV must be 16 kHz, mono, 16-bit PCM")
            pcm = audio.readframes(audio.getnframes())
            if len(pcm) != audio.getnframes() * 2:
                raise ValueError("WAV data is truncated")
    else:
        raise ValueError("Only .pcm or .wav files are supported")
    if not pcm or len(pcm) % 2:
        raise ValueError("Audio must be non-empty, even-byte PCM16LE")
    return pcm


def send_audio(session, pcm, stopped, progress, lock):
    try:
        for offset in range(0, len(pcm), FRAME_BYTES):
            if stopped.is_set():
                return
            session.append_audio(pcm[offset:offset + FRAME_BYTES].ljust(FRAME_BYTES, b"\0"))
            if stopped.wait(0.1):
                return
        input_finished = time.monotonic()
        while not stopped.is_set():
            now = time.monotonic()
            with lock:
                text_at, audio_at = progress["text"], progress["audio"]
            if text_at is not None and audio_at is not None:
                if now - max(input_finished, text_at, audio_at) >= IDLE_SECONDS:
                    with lock:
                        progress["closing"] = True
                    session.end_session()  # Stop sending audio; main thread waits for the close ack
                    if not stopped.wait(CLOSE_TIMEOUT):
                        raise TimeoutError("session.closed not received")
                    return
            if now - input_finished >= REPLY_TIMEOUT:
                raise TimeoutError("Reply did not finish within the wait window")
            session.append_audio(bytes(FRAME_BYTES))
            stopped.wait(0.1)
    except Exception:
        session.cancel()  # Wake the receiver on send failure; the exception surfaces via sender.result()
        raise


def main(input_path, output_path):
    pcm = read_pcm(input_path)
    stopped, lock = Event(), Lock()
    progress = {"text": None, "audio": None, "closing": False}
    with Client() as client:
        with client.audio.live.connect(
            model=os.getenv("LIVE_ROUTE_MODEL", "qwen-live"),
            ws_url=os.getenv("LIVE_WS_URL"), timeout=30,
        ) as session:
            with ThreadPoolExecutor(max_workers=1) as pool, open(output_path, "xb") as out:
                sender = None
                try:
                    session.update_session(
                        model=FRONTEND_MODEL, delegation={"type": "none"},
                        instructions="Answer the user's voice questions concisely",
                        input_audio_format="pcm", output_audio_format="pcm",
                        context={"compaction": {"enabled": False, "mode": "local"}},
                    )
                    for event in session.events():
                        kind = event["type"]
                        if kind == "session.updated" and sender is None:
                            sender = pool.submit(send_audio, session, pcm, stopped, progress, lock)
                        elif kind == "session.input_transcript.delta":
                            print(f"\n[user] {event['delta']}", flush=True)
                        elif kind == "session.output_transcript.delta":
                            if event["delta"]:
                                with lock:
                                    progress["text"] = time.monotonic()
                            print(event["delta"], end="", flush=True)
                        elif kind == "session.output_audio.delta":
                            data = base64.b64decode(event["delta"], validate=True)
                            if not data or len(data) % 2:
                                raise ValueError("Downstream audio is not valid PCM16LE")
                            out.write(data)
                            samples = array("h", data)
                            if sys.byteorder != "little":
                                samples.byteswap()
                            if sum(x * x for x in samples) >= len(samples) * 150 ** 2:
                                with lock:
                                    progress["audio"] = time.monotonic()
                        elif kind == "error":
                            raise RuntimeError(event["error"])  # This example exits on any error
                        elif kind == "session.closed":
                            with lock:
                                requested = progress["closing"]
                            if not requested or event.get("reason") != "close_requested":
                                raise RuntimeError(f"Session ended early: {event.get('reason')}")
                            print(f"\nClose reason: {event.get('reason')}")
                            break
                finally:
                    stopped.set()
                    session.close()  # Bounded WebSocket close first, then wait for the sender thread
                if sender is not None:
                    sender.result()
    print()


if __name__ == "__main__":
    try:
        main(sys.argv[1], sys.argv[2])
    except KeyboardInterrupt:
        print()
```

Configure a workspace, or pass the endpoint explicitly via `LIVE_WS_URL`;
`LIVE_ROUTE_MODEL` sets the route model and `LIVE_FRONTEND_MODEL` the frontend
model. These three `LIVE_*` variables are read by the example — they are not
SDK global settings. `output.pcm` must not exist when running:

```bash
python qwen_live_quickstart.py input.pcm output.pcm
```

`output.pcm` is 24 kHz, mono, 16-bit raw PCM, including received silence
frames. The example treats RMS ≥ 150 as audible output; after receiving text
and audible audio, it waits until the input has ended and both have been quiet
for 5 consecutive seconds before requesting close. This is not a server-side
completion confirmation — long pauses or background noise can affect the
judgment. After the input ends it waits up to 90 seconds for the reply, and up
to 5 more seconds after sending the close request; timeouts cancel the
connection and raise. Receiving a close event before the completion conditions
are met also raises.

### Example 2: text input & delegation modes

**Plain text input**: on frontends that support proactive text replies,
`append_commentary()` must be followed by `create_response()`; the former's
acknowledgment does not mean answering has started. In example 1's `main()`,
change `pcm = read_pcm(input_path)` to `pcm = b""`, and replace the
`pool.submit(...)` line in the `session.updated` branch with the code below;
everything else stays the same. Empty input only sends silence to maintain the
audio clock:

```python
session.append_commentary("Hello, please introduce yourself in one sentence.")
session.create_response()
sender = pool.submit(send_audio, session, pcm, stopped, progress, lock)
```

**Backend model delegation**: keep example 1's voice input, define the
following variables before configuration, and replace
`delegation={"type": "none"}` in `update_session()` with
`delegation=delegation`. `create_response_item()` is only for backend input in
this mode — it cannot replace `append_commentary()` in `none` mode:

```python
delegation = {
    "type": "responses",
    "responses": {
        "model": "<backend-model>",
        "instructions": "Answer the user directly without external tools",
    },
}
```

**External agent delegation**: change example 1's configuration to
`delegation={"type": "client"}` and add the branch below to the event loop.
Fixed text is used here to demonstrate write-back; real results come from your
agent:

```python
elif kind == "session.delegation.created":
    session.append_commentary("Cloudy turning sunny in Hangzhou tomorrow, 18 to 26 degrees.",
                              delegation_id=event["delegation_id"])
```

The current service accepts only one write-back per `delegation_id`; repeated
write-backs may return `CONFLICT`. Do not call the three write-back methods
back to back. `append_instruction` yields while waiting for user input,
`append_thinking` carries silent context, and `append_commentary` carries task
results; receiving an acknowledgment alone does not mean speech playback has
finished. Move slow agent processing off the receiving thread.

---

## Important notes & limitations

- **Reuse one session for multi-turn conversation**: after the first turn,
  keep the send/receive loop running and continue sending the next turn's
  audio on the same connection; do not call `end_session()` at the end of
  every turn. The full example closes after one turn — it is only a minimal
  integration.
- **There is no downstream-audio completion event**: neither
  `session.usage.updated` nor `session.output_audio.interrupted` can serve as
  an end-of-reply signal; for business completion confirmation, look at the
  `session.closed` event type, not `session.status`.
- **Config updates do not rebuild the connection**: connection-level options
  such as voice or transport may not support hot updates; when the server
  returns `Changing connection options requires a new session`, reconnect and
  re-initialize.
- **Reconnection requires re-initialization**: the SDK does not auto-reconnect
  or resend; calling `connect()` again while open or closing raises
  `RuntimeError`. After reconnecting you must send the configuration and
  create a new `events()` iterator — iterators bound to the old connection
  cannot be reused.
- **Every session must release its connection**: use `with` or
  `finally: session.close()` as a safety net; with multiple threads, stop
  sending and close the connection first, then wait for worker threads to
  exit. When a business close confirmation is needed, stop the audio upstream
  first, then send `end_session()` and keep receiving.
- **Handshake authentication failures**: 401 means the connection was never
  established — check the API key, target address, region and workspace access
  first; changing audio formats or send intervals will not help. When
  troubleshooting with the service provider, share `exc.request_id`, never the
  API key.
- **Troubleshoot by error stage**: a successful handshake does not mean the
  frontend model is usable; handle `session.update` parameter errors and later
  inference errors separately according to the server's `error.code` /
  `message` — do not treat a single `UNSUPPORTED` as the whole model being
  unavailable.