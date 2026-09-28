# Two-host gufo TP=2 server as systemd user units

Runs the gufo LLM server with tensor parallelism 2 across two Linux hosts,
entirely from systemd user units — no wrapper shell scripts:

- **rank 0 (head)**: serves the OpenAI-compatible HTTP API; unit
  `gufo@.service`, instance name = quantization (`q4`, `q8`, ...)
- **rank 1 (worker)**: GPU peer; unit `gufo-rank1@.service` on the second
  host. The head unit starts and stops it over ssh, so a single
  `systemctl --user start gufo` brings the whole pair up.

Each rank runs `podman run --rm` (no `-d`) in the foreground; systemd owns
the process and the restart policy, and all container output is tagged with
the unit in the user journal.

## Units

| Unit                  | Host | Purpose                                    |
|-----------------------|------|--------------------------------------------|
| `gufo@.service`       | rank 0 | head server; instance = quantization    |
| `gufo.service`        | rank 0 | wrapper: `start gufo` → `gufo@q8`        |
| `gufo-rank1@.service` | rank 1 | worker; started by the head unit via ssh |

None of the units has an `[Install]` section: they are started explicitly,
never at boot, so they do not claim the GPU before other local GPU services.
If another local user unit claims the GPU, add a drop-in
(`gufo@.service.d/local.conf` with `Conflicts=<unit>`).

## Prerequisites (both hosts)

- gufo checkout at `~/git/gufo` with a built
  `build/gpu-tp2/gufo` (the container mounts it read-only)
- ROCm runtime image in podman (see `GUFO_IMAGE` in the env examples)
- models under `/opt/models` (mounted read-only into the container)
- on the head host: passwordless ssh to the worker host
- identical control token at `~/.config/gufo.token` on both hosts

## Install

Copy the unit files to `~/.config/systemd/user/` on **both** hosts and
create the environment files from the `*.example` files:

```sh
# both hosts
cp 'gufo@.service' gufo-rank1@.service gufo.service \
   ~/.config/systemd/user/
mkdir -p ~/.config/gufo
cp head.env.example ~/.config/gufo/head.env      # rank 0 host only
cp worker.env.example ~/.config/gufo/worker.env  # rank 1 host only
cp quant.env.example ~/.config/gufo/q8.env       # one per quantization, both hosts
systemctl --user daemon-reload
```

Environment layout (values are host-specific, units stay generic):

| File                            | Host   | Contents                                    |
|---------------------------------|--------|---------------------------------------------|
| `~/.config/gufo/head.env`       | rank 0 | `GUFO_PEER`, `GUFO_HOST`, `GUFO_PORT`, `GUFO_IMAGE`, `GUFO_MTP_MODEL`, `GUFO_SESSIONS`, `GUFO_MAX_PENDING` |
| `~/.config/gufo/worker.env`     | rank 1 | `GUFO_HEAD_HOST`, `GUFO_IMAGE`, `GUFO_MTP_MODEL`, `GUFO_SESSIONS`, `GUFO_MAX_PENDING` |
| `~/.config/gufo/<quant>.env`    | both   | `GUFO_MODEL` (primary model file for that quantization) |
| `~/.config/gufo.token`      | both   | shared TP control token (identical)         |

The instance name of `gufo@<quant>` / `gufo-rank1@<quant>` selects
`<quant>.env`; a missing file fails the start with a clear error.
`GUFO_SESSIONS` / `GUFO_MAX_PENDING` must match on both hosts — the TP
control handshake rejects differing values.

## Usage (all on the head host)

```sh
systemctl --user start gufo          # start pair (q8); returns in ~1 s
systemctl --user stop gufo           # stop pair
systemctl --user status gufo@q8
journalctl --user -u gufo@q8 -f                      # rank 0 logs
ssh <worker-host> journalctl --user -u gufo-rank1@q8 -f   # rank 1 logs
```

Model load takes ~25 s per rank; the server answers
`http://<head>:8000/v1/models` ~1 minute after `start`.

## Crash behavior

- `Restart=on-failure`, `RestartSec=5`: a crashed or killed rank container
  is restarted by systemd; the other rank's 30 s handshake windows mean an
  independently restarted pair converges within one or two restart cycles.
- `podman run --rm` removes the container with the process, so no stale
  containers accumulate.
- Stopping `gufo` (or `gufo@<quant>`) also stops the worker on the peer
  host via ssh (`ExecStopPost`, best effort).

## Notes

- Environment values referenced in `Exec*` lines use systemd's `$VAR`
  expansion from the `EnvironmentFile=` files; unknown variables expand to
  the empty string, so keep all needed variables in the env files.
- The TP bootstrap (port 18515) and control (port 18516) ports must be
  reachable between the hosts.
