# Home Alert V1.1

V1.1 turns the V1 computer-vision proof of concept into a local, event-driven
room or home monitor. Development is intentionally split into testable
checkpoints rather than delivered as one large build.

## Product principle

The camera remains ready, but files are created only for meaningful person
events. Known, unknown, and uncertain identities can have different alert and
retention policies. Processing and media remain local by default.

## Checkpoints

1. **Event foundation — complete**
   - Typed application configuration
   - SQLite event database with schema versioning
   - Known, unknown, and uncertain identity states
   - Protected/reviewed/alerted event state
   - Age-based and disk-budget retention
   - Automated tests
2. **Triggered recording — next**
   - Reuse NanoDet, ByteTrack, and SFace from V1
   - In-memory pre-event frame buffer
   - One snapshot and MP4 clip per person event
   - Clean start/stop rules and cooldowns
3. **Local alerts**
   - One Windows notification per unknown/uncertain event
   - Click-through to the event, with suppression and cooldown
4. **Local dashboard**
   - Live view, event timeline, playback, review and protection controls
   - Enrol/remove people through a controlled UI
   - Armed, disarmed, and privacy modes
5. **Hardening**
   - Storage and failure testing
   - Authentication and audit trail
   - Packaging and startup behaviour

Phone access and remote live streaming are intentionally deferred to V1.2 so
that authentication and network exposure are designed rather than improvised.

## Run checkpoint 1 tests

From this directory:

```powershell
..\.cctv-venv\Scripts\python.exe -m unittest discover -s tests -v
```

The tests use temporary directories and do not touch real event data.

## Layout

```text
v1.1/
  home_alert/
    config.py       configuration and retention limits
    events.py       event domain model
    repository.py   SQLite persistence
    retention.py    expiry and storage-budget cleanup
  tests/
  README.md
```

## Deferred V1.2 ideas

- Authenticated phone-friendly interface
- Secure remote access through a VPN rather than an exposed router port
- Multiple cameras
- Mobile push notifications
- Narrow, observable rules such as zone entry or out-of-hours presence

Broad behaviour interpretation, cloud face recognition, audio recording, and
automatic security/police actions are out of scope.
