import json
import stat

import pytest

from forkeval.policy import ClaudeCodePolicy, render_transcript


def _fake_cli(tmp_path, replies):
    """A stand-in for `claude` that prints one canned reply per call and records its stdin."""
    for i, r in enumerate(replies):
        (tmp_path / f"reply{i}.json").write_text(json.dumps(r))
    script = tmp_path / "claude"
    script.write_text(
        "#!/bin/sh\n"
        f"d={tmp_path}\n"
        'n=$(cat "$d/n" 2>/dev/null || echo 0)\n'
        'cat > "$d/stdin$n"\n'
        'echo $((n+1)) > "$d/n"\n'
        'cat "$d/reply$n.json"\n'
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return str(script)


def _reply(action, **usage):
    return {"is_error": False, "structured_output": action, "result": json.dumps(action),
            "modelUsage": {"m": {"inputTokens": 3, "outputTokens": 5, "cacheReadInputTokens": 7,
                                 "cacheCreationInputTokens": 11, **usage}}}


def test_transcript_shows_every_step_and_result():
    messages = [
        {"role": "user", "content": "fix it"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "a", "name": "shell", "input": {"command": "ls"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "a", "content": "exit code: 0\nf.txt"}]},
    ]
    text = render_transcript(messages)
    assert "fix it" in text and "Step 1: shell" in text and '"command": "ls"' in text and "f.txt" in text
    assert "(none yet)" in render_transcript(messages[:1])


@pytest.mark.asyncio
async def test_actions_become_turns(tmp_path):
    binary = _fake_cli(tmp_path, [
        _reply({"action": "shell", "command": "ls /"}),
        _reply({"action": "write_file", "path": "/a", "content": "x"}),
        _reply({"action": "done", "summary": "all good"}),
    ])
    policy = ClaudeCodePolicy("some-model", binary=binary)
    messages = [{"role": "user", "content": "do the thing"}]

    turn = await policy.act(messages)
    assert turn.stop == "tool_use" and turn.tool_calls[0].name == "shell"
    assert turn.tool_calls[0].input == {"command": "ls /"}
    assert turn.content[0]["id"] == turn.tool_calls[0].id
    assert turn.usage.total == 26
    assert "do the thing" in (tmp_path / "stdin0").read_text()

    turn = await policy.act(messages)
    assert turn.tool_calls[0].name == "write_file" and turn.tool_calls[0].input == {"path": "/a", "content": "x"}

    turn = await policy.act(messages)
    assert turn.stop == "end_turn" and not turn.tool_calls and turn.content[0]["text"] == "all good"


@pytest.mark.asyncio
async def test_a_reply_without_an_action_is_asked_for_once_more(tmp_path):
    prose = {"is_error": False, "structured_output": None, "result": "I think we should look around."}
    binary = _fake_cli(tmp_path, [prose, _reply({"action": "shell", "command": "pwd"}), prose, prose])
    policy = ClaudeCodePolicy("some-model", binary=binary)
    turn = await policy.act([{"role": "user", "content": "t"}])
    assert turn.stop == "tool_use" and turn.tool_calls[0].input == {"command": "pwd"}
    turn = await policy.act([{"role": "user", "content": "t"}])
    assert turn.stop == "no_action" and not turn.tool_calls
