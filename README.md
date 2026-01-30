<img width="898" height="882" alt="image" src="https://github.com/user-attachments/assets/6324e0bf-91b8-4f1f-a1ba-48cfe7bb7e3f" />

## Claude Code exporter to Weave

- Exports Claude Code sessions to Weave.
- Constructs a chat view of Claude Code sessions.

### To run

To open the notebook:
```bash
uv run marimo run exporter.py
```

1. Enter your Weave project name and W&B API key.
    - Run the cell to connect your Weave project.
2. Run the cell to list your Claude projects.
    - Select the projects you want to export.
3. Run the cell to initiate exporting the project's Claude Code sessions to Weave.
4. Emit results of the export to a dataframe below.

> [!NOTE]
> ToDo: I can probably make this more Marimo native with a nicer UI and such.
