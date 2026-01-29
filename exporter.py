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

        Groups streamed assistant messages by their message.id and tracks tool results.

        Returns:
            Tuple of (session_data, error_reason). If parsing succeeds, error_reason is None.
            If parsing fails, session_data is None and error_reason explains why.
        """
        session_info = {}
        line_count = 0
        json_errors = 0
        all_message_types = set()

        # Track API responses by their message ID (handles streaming)
        api_responses = {}  # message_id -> response data
        # Track tool results from user messages
        tool_results = {}  # tool_use_id -> result content
        # Track user prompts
        user_prompts = []  # List of user message data
        # Track order of messages by uuid
        message_order = []  # List of (uuid, type, message_id_or_none)

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

                        # Capture session info from ANY message that has it
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

                        timestamp_str = obj.get("timestamp", "")
                        timestamp = parse_timestamp(timestamp_str) if timestamp_str else datetime.now()
                        uuid = obj.get("uuid")

                        # Handle assistant messages (group by message.id)
                        if msg_type == "assistant":
                            msg_data = obj.get("message", {})
                            msg_id = msg_data.get("id")

                            if msg_id:
                                # Initialize or update the API response for this message ID
                                if msg_id not in api_responses:
                                    api_responses[msg_id] = {
                                        "message_id": msg_id,
                                        "model": msg_data.get("model", "unknown"),
                                        "content_blocks": [],
                                        "usage": {},
                                        "timestamp": timestamp,
                                        "parent_uuid": obj.get("parentUuid"),
                                        "uuid": uuid,
                                    }
                                    message_order.append((uuid, "assistant", msg_id))

                                # Append content blocks from this streamed chunk
                                content = msg_data.get("content", [])
                                if isinstance(content, list):
                                    api_responses[msg_id]["content_blocks"].extend(content)

                                # Update usage if present (later chunks may have complete usage)
                                usage = msg_data.get("usage", {})
                                if usage:
                                    api_responses[msg_id]["usage"] = usage

                        # Handle user messages
                        elif msg_type == "user":
                            msg_data = obj.get("message", {})
                            content = msg_data.get("content", "")
                            user_text = []

                            # Extract text and tool results
                            if isinstance(content, str):
                                user_text.append(content)
                            elif isinstance(content, list):
                                for c in content:
                                    if isinstance(c, dict):
                                        if c.get("type") == "text":
                                            user_text.append(c.get("text", ""))
                                        elif c.get("type") == "tool_result":
                                            # Store tool result for later matching
                                            tool_use_id = c.get("tool_use_id")
                                            if tool_use_id:
                                                result_content = c.get("content", "")
                                                # Handle both string and list format
                                                if isinstance(result_content, str):
                                                    tool_results[tool_use_id] = result_content
                                                elif isinstance(result_content, list):
                                                    text_parts = []
                                                    for block in result_content:
                                                        if isinstance(block, dict) and block.get("type") == "text":
                                                            text_parts.append(block.get("text", ""))
                                                    tool_results[tool_use_id] = "\n".join(text_parts)

                            # Only record user prompts that have actual text content
                            if user_text:
                                user_prompts.append({
                                    "text": " ".join(user_text),
                                    "timestamp": timestamp,
                                    "uuid": uuid,
                                    "parent_uuid": obj.get("parentUuid"),
                                })
                                message_order.append((uuid, "user", None))

                    except json.JSONDecodeError as e:
                        json_errors += 1
                        continue
        except Exception as e:
            return None, f"Failed to read file: {e}"

        # Check if we found a valid session
        if not session_info.get("sessionId"):
            # Try to extract sessionId from the filename (UUID pattern)
            filename = path.stem
            if len(filename) == 36 and filename.count('-') == 4:
                session_info["sessionId"] = filename
            else:
                return None, f"No sessionId found in {line_count} lines (types: {all_message_types}, json_errors: {json_errors})"

        if not api_responses and not user_prompts:
            return None, f"No messages found in {line_count} lines (types: {all_message_types})"

        # Build list of LLM responses with their tool calls
        llm_responses = []
        for msg_id, response_data in api_responses.items():
            # Extract text and tool_use blocks
            text_parts = []
            tool_calls = []

            for block in response_data["content_blocks"]:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif block.get("type") == "tool_use":
                        tool_id = block.get("id", "")
                        tool_calls.append({
                            "id": tool_id,
                            "name": block.get("name", "unknown"),
                            "input": block.get("input", {}),
                            "result": tool_results.get(tool_id),  # Match with tool result
                        })

            llm_responses.append({
                "message_id": msg_id,
                "model": response_data["model"],
                "text": " ".join(text_parts),
                "tool_calls": tool_calls,
                "usage": response_data["usage"],
                "timestamp": response_data["timestamp"],
                "parent_uuid": response_data["parent_uuid"],
                "uuid": response_data["uuid"],
            })

        # Sort by timestamp to maintain order
        llm_responses.sort(key=lambda r: r["timestamp"])

        # Success - return session data with None error
        return {
            "session_id": session_info["sessionId"],
            "cwd": session_info.get("cwd"),
            "git_branch": session_info.get("gitBranch"),
            "version": session_info.get("version"),
            "user_prompts": user_prompts,
            "llm_responses": llm_responses,
        }, None

    def import_single_session(session_file, project_name):
        """Import a single session file with nested traces and return result dict."""
        result = {
            "project": project_name,
            "session_file": session_file.name,
            "session_id": "",
            "status": "error",
            "error": "",
            "llm_responses": 0,
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

            llm_responses = session.get("llm_responses", [])
            user_prompts = session.get("user_prompts", [])

            if not llm_responses:
                result["status"] = "skipped"
                result["error"] = "No LLM responses found"
                result["session_id"] = session["session_id"]
                return result

            result["session_id"] = session["session_id"]

            # Calculate totals
            total_tool_calls = sum(len(r.get("tool_calls", [])) for r in llm_responses)
            total_tokens = 0
            models = set()

            for response in llm_responses:
                usage = response.get("usage", {})
                total_tokens += usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
                model = response.get("model", "")
                if model:
                    models.add(model)

            # Get first user prompt for display
            first_prompt = user_prompts[0]["text"] if user_prompts else ""
            display_name = (first_prompt[:50] + "...") if len(first_prompt) > 50 else first_prompt

            # Get session timestamps
            session_started = user_prompts[0]["timestamp"] if user_prompts else (llm_responses[0]["timestamp"] if llm_responses else None)
            session_ended = llm_responses[-1]["timestamp"] if llm_responses else None

            # Build messages array for ChatView (OpenAI format)
            # Interleave user prompts and LLM responses by timestamp
            all_entries = []
            for prompt in user_prompts:
                all_entries.append({
                    "type": "user",
                    "timestamp": prompt["timestamp"],
                    "data": prompt,
                })
            for response in llm_responses:
                all_entries.append({
                    "type": "llm_response", 
                    "timestamp": response["timestamp"],
                    "data": response,
                })

            # Sort by timestamp to maintain conversation order
            all_entries.sort(key=lambda e: e["timestamp"])

            # Build messages array - exclude the final assistant response
            # (it will be in the output instead)
            messages = []
            for i, entry in enumerate(all_entries):
                # Skip the last entry if it's an assistant response (goes in output)
                is_last_assistant = (i == len(all_entries) - 1 and entry["type"] == "llm_response")
                if is_last_assistant:
                    continue

                if entry["type"] == "user":
                    messages.append({
                        "role": "user",
                        "content": entry["data"]["text"]
                    })
                else:
                    response = entry["data"]
                    asst_entry = {
                        "role": "assistant",
                        "content": response.get("text") or None,
                    }

                    # Include tool_calls in OpenAI format if present
                    tool_calls_for_msg = []
                    for tc in response.get("tool_calls", []):
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

                    # Add tool results as separate messages
                    for tc in response.get("tool_calls", []):
                        if tc.get("result"):
                            messages.append({
                                "role": "tool",
                                "content": tc.get("result", ""),
                                "tool_call_id": tc.get("id", "")
                            })

            # 1. CREATE SESSION CALL (root trace)
            session_id = session["session_id"]
            session_display = f"[{session_id[:8]}] {display_name}" if display_name else session_id

            session_call = client.create_call(
                op="claude_code.session",
                inputs={
                    # ChatView-compatible messages array
                    "messages": messages,
                    # Keep existing metadata
                    "first_prompt": first_prompt,
                    "cwd": session.get("cwd"),
                    "git_branch": session.get("git_branch"),
                    "project": project_name,
                },
                attributes={
                    "weave": {"kind": "agent"},
                    "session_id": session_id,
                    "session_uuid": session_id,  # Full UUID for debugging
                    "claude_code_version": session.get("version"),
                    "session_started_at": session_started.isoformat() if session_started else None,
                    "session_ended_at": session_ended.isoformat() if session_ended else None,
                },
                display_name=session_display,
                use_stack=False,  # Important for retroactive logging
            )
            # Set timestamp directly on Call object
            if session_started:
                session_call.started_at = session_started

            # Build choices array for ChatView output
            # Get the final assistant text from the last response
            final_assistant_text = llm_responses[-1].get("text", "") if llm_responses else ""

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
                    "user_message_count": len(user_prompts),
                    "llm_responses_count": len(llm_responses),
                    "tool_calls_count": total_tool_calls,
                    "total_tokens": total_tokens,
                    "models": list(models),
                },
            )

            weave_calls = 1

            # 2. CREATE INTERLEAVED USER AND LLM RESPONSE TRACES (children of session)
            # Combine user prompts and LLM responses, sorted by timestamp
            all_messages = []
            for prompt in user_prompts:
                all_messages.append({
                    "type": "user",
                    "data": prompt,
                    "timestamp": prompt["timestamp"],
                })
            for response in llm_responses:
                all_messages.append({
                    "type": "llm_response",
                    "data": response,
                    "timestamp": response["timestamp"],
                })

            # Sort by timestamp to maintain conversation order
            all_messages.sort(key=lambda m: m["timestamp"])

            # Create traces for each message
            # Track current user call to nest LLM responses under it
            current_user_call = None

            for i, msg in enumerate(all_messages):
                if msg["type"] == "user":
                    # Create user message trace
                    prompt = msg["data"]
                    prompt_text = prompt.get("text", "")
                    prompt_display = f"User: {prompt_text[:60]}..." if len(prompt_text) > 60 else f"User: {prompt_text}"

                    user_call = client.create_call(
                        op="claude_code.user_message",
                        inputs={"content": prompt_text},
                        parent=session_call,
                        display_name=prompt_display,
                        use_stack=False,
                    )
                    # Set timestamp
                    if prompt.get("timestamp"):
                        user_call.started_at = prompt.get("timestamp")
                        user_call.ended_at = prompt.get("timestamp")

                    client.finish_call(
                        user_call,
                        output={"message_uuid": prompt.get("uuid")},
                    )
                    weave_calls += 1

                    # Track this user call for nesting subsequent LLM responses
                    current_user_call = user_call

                elif msg["type"] == "llm_response":
                    # Create LLM response trace
                    response = msg["data"]
                    response_text = response.get("text", "")
                    message_id = response.get("message_id", f"response_{i}")
                    model = response.get("model", "unknown")

                    # Create display name from text or tool names
                    if response_text:
                        response_display = f"{model}: {response_text[:40]}..."
                    elif response.get("tool_calls"):
                        tool_names = [tc.get("name", "unknown") for tc in response.get("tool_calls", [])]
                        response_display = f"{model}: {', '.join(tool_names[:3])}"
                    else:
                        response_display = f"{model} response {i+1}"

                    # Nest under current user message, or session if no user message yet
                    parent = current_user_call if current_user_call else session_call

                    response_call = client.create_call(
                        op="claude_code.llm_response",
                        inputs={
                            "model": model,
                            "message_id": message_id,
                        },
                        attributes={
                            "weave": {"kind": "llm"},
                        },
                        parent=parent,  # NESTED UNDER USER MESSAGE
                        display_name=response_display,
                        use_stack=False,
                    )
                    # Set timestamp
                    if response.get("timestamp"):
                        response_call.started_at = response.get("timestamp")
                        response_call.ended_at = response.get("timestamp")

                    # Finish LLM response call
                    usage = response.get("usage", {})
                    client.finish_call(
                        response_call,
                        output={
                            "text": response_text,
                            "tool_calls_count": len(response.get("tool_calls", [])),
                            "usage": usage,
                        },
                    )
                    weave_calls += 1

                    # 3. CREATE TOOL CALLS (children of LLM response)
                    for tc in response.get("tool_calls", []):
                        tool_name = tc.get("name", "unknown")
                        tool_input = tc.get("input", {})
                        tool_result = tc.get("result")

                        tool_call = client.create_call(
                            op=f"claude_code.tool.{tool_name}",
                            inputs=tool_input,
                            attributes={
                                "weave": {"kind": "tool"},
                            },
                            parent=response_call,  # NESTED UNDER LLM RESPONSE
                            display_name=tool_name,
                            use_stack=False,
                        )
                        # Use same timestamp as parent response
                        if response.get("timestamp"):
                            tool_call.started_at = response.get("timestamp")
                            tool_call.ended_at = response.get("timestamp")

                        client.finish_call(
                            tool_call,
                            output={"result": tool_result},
                        )
                        weave_calls += 1

            # Update result
            result.update({
                "status": "success",
                "llm_responses": len(llm_responses),
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
                                print(f"    ✅ Success: {result['llm_responses']} responses, {result['tool_calls']} tools, {result['tokens']:,} tokens")
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
    results_table = mo.ui.table(data=results_df) if not results_df.empty else mo.ui.table(data=pd.DataFrame())
    results_table
    return (results_table,)


if __name__ == "__main__":
    app.run()
