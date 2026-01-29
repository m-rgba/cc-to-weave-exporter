import marimo

__generated_with = "0.19.6"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(r"""
    ## Connect to Weave
    """)
    return


@app.cell
def _():
    import os
    import json
    import marimo as mo
    import re
    import pandas as pd
    import weave

    from pathlib import Path
    from datetime import datetime
    from weave.trace.context.weave_client_context import require_weave_client

    WANDB_PROJECT_NAME = mo.ui.text(
        label="W&B Project Name",
        placeholder="Enter your Weights & Biases project name",
        value=""
    )
    WANDB_API_KEY = mo.ui.text(
        label="W&B API Key",
        placeholder="Enter your Weights & Biases API key",
        value="",
        kind="password"
    )
    mo.vstack([WANDB_PROJECT_NAME, WANDB_API_KEY])
    return (
        Path,
        WANDB_API_KEY,
        WANDB_PROJECT_NAME,
        datetime,
        json,
        mo,
        os,
        pd,
        re,
        require_weave_client,
        weave,
    )


@app.cell
def _(WANDB_API_KEY, WANDB_PROJECT_NAME, os, weave):
    # Initialize Weave with provided credentials
    weave_client = None
    weave_init_error = None

    if WANDB_PROJECT_NAME.value and WANDB_API_KEY.value:
        try:
            os.environ["WANDB_API_KEY"] = WANDB_API_KEY.value
            weave_client = weave.init(WANDB_PROJECT_NAME.value)
            print(f"✅ Initialized Weave project: {WANDB_PROJECT_NAME.value}")
        except Exception as e:
            weave_init_error = str(e)
            print(f"❌ Error initializing Weave: {e}")        
    elif WANDB_PROJECT_NAME.value or WANDB_API_KEY.value:
            print(f"⚠️ Please provide both W&B Project Name and API Key to initialize Weave")        
    return weave_client, weave_init_error


@app.cell
def _(mo):
    mo.md(r"""
    ## Select Claude Code projects to export
    """)
    return


@app.cell
def _(Path, mo, os, pd):
    # Find claude projects
    def find_claude_projects():
        # 1. Check for Environment Variable Override
        env_path = os.getenv("CLAUDE_CONFIG_DIR")
        if env_path:
            potential_path = Path(env_path) / "projects"
            if potential_path.exists():
                return potential_path

        # 2. Define common search locations
        home = Path.home()
        search_locations = [
            home / ".claude" / "projects",                        # Standard Default
            home / ".config" / "claude" / "projects",             # XDG Style (Linux)
            home / "AppData" / "Roaming" / "Claude" / "projects", # Windows Style
            Path("/usr/local/share/claude/projects")              # Global Install
        ]

        # 3. Iterate and verify existence
        for loc in search_locations:
            if loc.exists() and loc.is_dir():
                return loc

        return None


    def list_projects(directory):
        """Takes a Path object and returns a sorted list of subdirectories."""
        if not directory:
            return []

        # Only grab directories, ignoring hidden system files like .DS_Store
        return sorted([p.name for p in directory.iterdir() if p.is_dir()])

    # Execution
    project_dir = find_claude_projects()
    if project_dir:
        projects = list_projects(project_dir)
        projects_df = pd.DataFrame({
            'project_name': projects
        })
    else:
        projects_df = pd.DataFrame()
        print("❌ Could not find the Claude projects folder automatically.")

    projects_table = mo.ui.table(data=projects_df)
    projects_table
    return (projects_table,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Export traces
    """)
    return


@app.cell
def _(
    Path,
    WANDB_PROJECT_NAME,
    datetime,
    json,
    pd,
    projects_table,
    re,
    require_weave_client,
    weave_client,
    weave_init_error,
):
    if not projects_table.value.empty:
        selected_projects = projects_table.value
        print(f"✅ Selected {len(selected_projects)} project(s)")
    else:
        selected_projects = None
        print("❌ No projects selected.")

    # UUID pattern to match session files (not agent-* files)
    UUID_PATTERN = re.compile(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\.jsonl$",
        re.IGNORECASE,
    )

    def is_uuid_filename(filename):
        """Check if filename matches UUID pattern (not agent-xxxx files)."""
        return UUID_PATTERN.match(filename) is not None

    def parse_timestamp(ts):
        """Parse ISO timestamp from Claude session."""
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        return datetime.fromisoformat(ts)

    def find_session_files(project_path):
        """Find all UUID-named session JSONL files in a project directory."""
        files = [f for f in project_path.glob("*.jsonl") if is_uuid_filename(f.name)]
        # Sort by modification time, newest first
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return files

    def parse_session_file(path):
        """Parse a Claude session JSONL file into a simple dict structure.

        Returns:
            Tuple of (session_data, error_reason). If parsing succeeds, error_reason is None.
            If parsing fails, session_data is None and error_reason explains why.
        """
        messages = []
        session_info = {}
        pending_tool_calls = {}  # Track tool calls waiting for results
        line_count = 0
        json_errors = 0
        all_message_types = set()

        try:
            with open(path) as f:
                for line in f:
                    line_count += 1
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        msg_type = obj.get("type")
                        all_message_types.add(msg_type)

                        if msg_type == "queue-operation":
                            continue

                        # Capture session info from ANY message that has it (not just the first)
                        if obj.get("sessionId"):
                            if not session_info.get("sessionId"):
                                session_info["sessionId"] = obj.get("sessionId")
                            # Update fields that may have been None initially
                            if not session_info.get("cwd") and obj.get("cwd"):
                                session_info["cwd"] = obj.get("cwd")
                            if not session_info.get("gitBranch") and obj.get("gitBranch"):
                                session_info["gitBranch"] = obj.get("gitBranch")
                            if not session_info.get("version") and obj.get("version"):
                                session_info["version"] = obj.get("version")

                        # Collect user and assistant messages
                        if msg_type in ("user", "assistant"):
                            messages.append(obj)

                    except json.JSONDecodeError as e:
                        json_errors += 1
                        continue
        except Exception as e:
            return None, f"Failed to read file: {e}"

        # Check if we found a valid session
        if not session_info.get("sessionId"):
            # Try to extract sessionId from the filename (UUID pattern)
            filename = path.stem  # Get filename without extension
            if len(filename) == 36 and filename.count('-') == 4:
                session_info["sessionId"] = filename
            else:
                return None, f"No sessionId found in {line_count} lines (types: {all_message_types}, json_errors: {json_errors})"

        if not messages:
            return None, f"No user/assistant messages found in {line_count} lines (types: {all_message_types})"

        # Organize messages into turns (user → assistant pairs)
        turns = []
        current_turn = None

        for msg in messages:
            msg_type = msg.get("type")
            timestamp_str = msg.get("timestamp", "")
            timestamp = parse_timestamp(timestamp_str) if timestamp_str else datetime.now()

            if msg_type == "user":
                # Start new turn
                if current_turn:
                    turns.append(current_turn)

                # Extract user content - handle both string and list formats
                msg_data = msg.get("message", {})
                content = msg_data.get("content", "")
                user_text = []

                if isinstance(content, str):
                    # Content is a plain string
                    user_text.append(content)
                elif isinstance(content, list):
                    # Content is a list of content blocks
                    for c in content:
                        if isinstance(c, dict):
                            if c.get("type") == "text":
                                user_text.append(c.get("text", ""))
                            elif c.get("type") == "tool_result":
                                # Tool results are part of user messages
                                tool_use_id = c.get("tool_use_id")
                                if tool_use_id and tool_use_id in pending_tool_calls:
                                    tc = pending_tool_calls[tool_use_id]
                                    result_content = c.get("content", "")
                                    # Handle both string and list format for tool results
                                    if isinstance(result_content, str):
                                        tc["result"] = result_content[:10000]
                                    elif isinstance(result_content, list):
                                        text_parts = []
                                        for block in result_content:
                                            if isinstance(block, dict) and block.get("type") == "text":
                                                text_parts.append(block.get("text", ""))
                                        tc["result"] = "\n".join(text_parts)[:10000]
                                    tc["result_timestamp"] = timestamp

                current_turn = {
                    "user_message": " ".join(user_text),
                    "assistant_messages": [],
                    "tool_calls": [],
                    "started_at": timestamp,
                    "ended_at": timestamp,
                }

            elif msg_type == "assistant" and current_turn:
                msg_data = msg.get("message", {})
                content = msg_data.get("content", [])
                usage_data = msg_data.get("usage", {})

                # Extract text and tool_use blocks
                text_content = []
                tool_calls = []

                if isinstance(content, list):
                    for c in content:
                        if isinstance(c, dict):
                            if c.get("type") == "text":
                                text_content.append(c.get("text", ""))
                            elif c.get("type") == "tool_use":
                                tc = {
                                    "id": c.get("id", ""),
                                    "name": c.get("name", "unknown"),
                                    "input": c.get("input", {}),
                                    "timestamp": timestamp,
                                    "result": None,
                                    "result_timestamp": None,
                                }
                                tool_calls.append(tc)
                                # Track for later result matching
                                pending_tool_calls[tc["id"]] = tc

                current_turn["assistant_messages"].append({
                    "text": " ".join(text_content),
                    "model": msg_data.get("model", "unknown"),
                    "usage": usage_data,
                    "timestamp": timestamp,
                })
                current_turn["tool_calls"].extend(tool_calls)
                current_turn["ended_at"] = timestamp

        # Add last turn
        if current_turn:
            turns.append(current_turn)

        # Success - return session data with None error
        return {
            "session_id": session_info["sessionId"],
            "cwd": session_info.get("cwd"),
            "git_branch": session_info.get("gitBranch"),
            "version": session_info.get("version"),
            "turns": turns,
        }, None

    def import_single_session(session_file, project_name):
        """Import a single session file with nested traces and return result dict."""
        result = {
            "project": project_name,
            "session_file": session_file.name,
            "session_id": "",
            "status": "error",
            "error": "",
            "turns": 0,
            "tool_calls": 0,
            "weave_calls": 0,
            "tokens": 0,
            "display_name": "",
        }

        try:
            # Get Weave client for create_call/finish_call API
            client = require_weave_client()
            # Parse session
            session, parse_error = parse_session_file(session_file)
            if not session:
                result["error"] = f"Parse failed: {parse_error}"
                return result

            turns = session.get("turns", [])
            if not turns:
                result["status"] = "skipped"
                result["error"] = "No turns found"
                result["session_id"] = session["session_id"]
                return result

            result["session_id"] = session["session_id"]

            # Calculate totals
            total_tool_calls = sum(len(t.get("tool_calls", [])) for t in turns)
            total_tokens = 0
            models = set()

            for turn in turns:
                for msg in turn.get("assistant_messages", []):
                    usage = msg.get("usage", {})
                    total_tokens += usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
                    model = msg.get("model", "")
                    if model:
                        models.add(model)

            # Get first user prompt for display
            first_prompt = turns[0].get("user_message", "") if turns else ""
            display_name = (first_prompt[:50] + "...") if len(first_prompt) > 50 else first_prompt

            # Get session timestamps
            session_started = turns[0].get("started_at") if turns else None
            session_ended = turns[-1].get("ended_at") if turns else None

            # Build messages array for ChatView (OpenAI format)
            messages = []
            for turn in turns:
                # Add user message
                user_msg = turn.get("user_message", "")
                if user_msg:
                    messages.append({
                        "role": "user",
                        "content": user_msg
                    })

                # Add assistant message(s) with tool_calls
                for asst_msg in turn.get("assistant_messages", []):
                    asst_entry = {
                        "role": "assistant",
                        "content": asst_msg.get("text") or None,
                    }

                    # Include tool_calls in OpenAI format if present in this turn
                    tool_calls_for_msg = []
                    for tc in turn.get("tool_calls", []):
                        tool_calls_for_msg.append({
                            "id": tc.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": tc.get("name", "unknown"),
                                "arguments": json.dumps(tc.get("input", {}))
                            }
                        })

                    if tool_calls_for_msg:
                        asst_entry["tool_calls"] = tool_calls_for_msg

                    messages.append(asst_entry)

                # Add tool results
                for tc in turn.get("tool_calls", []):
                    if tc.get("result"):
                        messages.append({
                            "role": "tool",
                            "content": tc.get("result", "")[:1000],  # Truncate long results
                            "tool_call_id": tc.get("id", "")
                        })

            # 1. CREATE SESSION CALL (root trace)
            session_call = client.create_call(
                op="claude_code.session",
                inputs={
                    # ChatView-compatible messages array
                    "messages": messages,
                    # Keep existing metadata
                    "first_prompt": first_prompt[:500],
                    "cwd": session.get("cwd"),
                    "git_branch": session.get("git_branch"),
                    "project": project_name,
                },
                attributes={
                    "session_id": session["session_id"],
                    "claude_code_version": session.get("version"),
                },
                display_name=display_name or session["session_id"],
                use_stack=False,  # Important for retroactive logging
            )
            # Set timestamp directly on Call object for older Weave versions
            if session_started:
                session_call.started_at = session_started

            # Build choices array for ChatView output
            # Get the final assistant message from the last turn
            final_assistant_text = ""
            if turns:
                last_turn = turns[-1]
                last_asst_messages = last_turn.get("assistant_messages", [])
                if last_asst_messages:
                    final_assistant_text = last_asst_messages[-1].get("text", "")

            # Set end timestamp and finish session call
            if session_ended:
                session_call.ended_at = session_ended
            client.finish_call(
                session_call,
                output={
                    # ChatView-compatible choices array
                    "choices": [{
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": final_assistant_text,
                        }
                    }],
                    # Keep existing rollup stats
                    "turns_count": len(turns),
                    "tool_calls_count": total_tool_calls,
                    "total_tokens": total_tokens,
                    "models": list(models),
                },
            )

            weave_calls = 1

            # 2. CREATE TURN CALLS (children of session)
            for i, turn in enumerate(turns):
                user_msg = turn.get("user_message", "")
                turn_display = f"Turn {i+1}: {user_msg[:30]}..." if user_msg else f"Turn {i+1}"

                # Collect assistant response text
                assistant_text = ""
                for msg in turn.get("assistant_messages", []):
                    assistant_text += msg.get("text", "") + "\n"

                turn_call = client.create_call(
                    op="claude_code.turn",
                    inputs={"user_message": user_msg[:1000]},
                    parent=session_call,  # NESTED UNDER SESSION
                    display_name=turn_display,
                    use_stack=False,
                )
                # Set timestamp directly on Call object for older Weave versions
                if turn.get("started_at"):
                    turn_call.started_at = turn.get("started_at")

                # Set end timestamp and finish turn call
                if turn.get("ended_at"):
                    turn_call.ended_at = turn.get("ended_at")
                client.finish_call(
                    turn_call,
                    output={
                        "assistant_response": assistant_text[:2000],
                        "tool_count": len(turn.get("tool_calls", [])),
                    },
                )
                weave_calls += 1

                # 3. CREATE TOOL CALLS (children of turn)
                for tc in turn.get("tool_calls", []):
                    tool_name = tc.get("name", "unknown")
                    tool_input = tc.get("input", {})
                    tool_result = tc.get("result")

                    # Truncate large inputs
                    truncated_input = {}
                    for key, value in tool_input.items():
                        if isinstance(value, str) and len(value) > 1000:
                            truncated_input[key] = value[:1000] + "... [truncated]"
                        else:
                            truncated_input[key] = value

                    # Calculate end timestamp
                    tool_started = tc.get("timestamp")
                    tool_ended = tc.get("result_timestamp") or tool_started

                    tool_call = client.create_call(
                        op=f"claude_code.tool.{tool_name}",
                        inputs=truncated_input,
                        parent=turn_call,  # NESTED UNDER TURN
                        display_name=tool_name,
                        use_stack=False,
                    )
                    # Set timestamp directly on Call object for older Weave versions
                    if tool_started:
                        tool_call.started_at = tool_started

                    # Set end timestamp and finish tool call
                    if tool_ended:
                        tool_call.ended_at = tool_ended
                    client.finish_call(
                        tool_call,
                        output={"result": tool_result[:1000] if tool_result else None},
                    )
                    weave_calls += 1

            # Update result
            result.update({
                "status": "success",
                "turns": len(turns),
                "tool_calls": total_tool_calls,
                "weave_calls": weave_calls,
                "tokens": total_tokens,
                "display_name": display_name,
            })

        except Exception as e:
            result["error"] = str(e)
            import traceback
            result["error"] = f"{str(e)}\n{traceback.format_exc()}"

        return result

    # Main import logic
    import_results = []

    if selected_projects is not None and not selected_projects.empty:
        # Check if Weave is initialized
        if not weave_client or weave_init_error:
            if weave_init_error:
                print(f"❌ Cannot import: Weave initialization failed - {weave_init_error}")
            else:
                print("❌ Please provide both W&B Project Name and API Key to initialize Weave")
        else:
            # Weave is already initialized
            try:
                print(f"✅ Using Weave project: {WANDB_PROJECT_NAME.value}")

                # Find Claude projects directory
                home = Path.home()
                search_locations = [
                    home / ".claude" / "projects",
                    home / ".config" / "claude" / "projects",
                    home / "AppData" / "Roaming" / "Claude" / "projects",
                    Path("/usr/local/share/claude/projects")
                ]

                projects_base = None
                for loc in search_locations:
                    if loc.exists():
                        projects_base = loc
                        break

                if projects_base:
                    # Process each selected project
                    for _, row in selected_projects.iterrows():
                        project_name = row['project_name']
                        project_path = projects_base / project_name

                        if not project_path.exists():
                            print(f"⚠️ Project directory not found: {project_path}")
                            continue

                        # Find session files
                        session_files = find_session_files(project_path)
                        print(f"📁 Found {len(session_files)} session(s) in {project_name}")

                        # Import each session
                        for session_file in session_files:
                            print(f"  ⏳ Importing {session_file.name}...")
                            result = import_single_session(session_file, project_name)
                            import_results.append(result)

                            if result["status"] == "success":
                                print(f"    ✅ Success: {result['turns']} turns, {result['tool_calls']} tools, {result['tokens']:,} tokens")
                            elif result["status"] == "skipped":
                                print(f"    ⏭️ Skipped: {result['error']}")
                            else:
                                print(f"    ❌ Error: {result['error']}")

                    # Summary
                    total_success = sum(1 for r in import_results if r["status"] == "success")
                    total_sessions = len(import_results)
                    print(f"\n📊 Summary: {total_success}/{total_sessions} sessions imported successfully")
                else:
                    print("❌ Could not find Claude projects directory")

            except Exception as e:
                print(f"❌ Error initializing Weave or importing sessions: {e}")
    else:
        print("❌ No projects selected")

    # Create results DataFrame
    results_df = pd.DataFrame(import_results) if import_results else pd.DataFrame()
    return (results_df,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Emit trace export results
    """)
    return


@app.cell
def _(mo, results_df):
    # Write results to JSON file
    if not results_df.empty:
        results_table = mo.ui.table(data=results_df)

    results_table
    return


if __name__ == "__main__":
    app.run()
