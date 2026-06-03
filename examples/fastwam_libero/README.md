# FastWAM LIBERO Input Example

This example is a minimal observation payload for the native FastWAM LIBERO
product path. The image arrays are intentionally tiny; the FastWAM processor
resizes and concatenates them according to the upstream runtime metadata.

Run one inference inside a prepared FastWAM runtime:

```bash
wam run fastwam-libero \
  --input examples/fastwam_libero/obs.json \
  --output /tmp/fastwam-action.json \
  --upstream-dir /path/to/FastWAM \
  --cache-dir /path/to/wam-cache
```

Smoke-test the local policy server with the same observation:

```bash
wam serve fastwam-libero \
  --smoke \
  --smoke-input examples/fastwam_libero/obs.json \
  --upstream-dir /path/to/FastWAM \
  --cache-dir /path/to/wam-cache
```

Both commands require the released FastWAM checkpoint, dataset stats, runtime
assets, and a FastWAM-compatible Python environment or container.
