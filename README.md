<img width="898" height="882" alt="image" src="https://github.com/user-attachments/assets/6324e0bf-91b8-4f1f-a1ba-48cfe7bb7e3f" />

## Claude Code exporter to Weave

- Exports Claude Code sessions to Weave with call kind metadata.
- Constructs a chat view of Claude Code sessions.

### To run

To open the notebook:
```bash
uv run marimo run exporter.py
```

To edit the notebook:
```bash
uv run marimo edit exporter.py
```

1. Enter your Weave project name and W&B API key.
2. Run the cell to list your Claude projects.
3. Run the cell to initiate exporting the project's Claude Code sessions to Weave.
4. Check out your Claude Code sessions in Weave.