# Technique API

Local JSON wrapper around Brady's GT Singer technique model.

```bash
python3 server/technique/api.py --port 8765
```

The browser app sends the recorded `take.wav` to:

```text
POST http://127.0.0.1:8765/analyze
```

The response includes `axis_result`, which matches the Project 2 report shape
used by the browser for pitch, tempo, dynamics, and technique.
