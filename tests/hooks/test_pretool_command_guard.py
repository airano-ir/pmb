"""PreToolUse guard, command-aware: a rule that NAMES a command ('never use git')
must fire when the agent is about to run that command, even though 'git' is
shorter than a distinctive token. Command names are extracted STRUCTURALLY from
the shell line (no hardcoded command list)."""
from __future__ import annotations

from pmb.core.engine import Engine
from pmb.core.text_match import shell_command_names
from pmb.mcp.daemon import pretool_lessons

GIT_RULE = "Never use git in this project - it is git-free by user directive."


def test_shell_command_names_structural():
    assert shell_command_names("git push origin main") == {"git"}
    assert shell_command_names("npm install && git commit -m x") == {"npm", "git"}
    assert shell_command_names("VAR=1 /usr/bin/git status") == {"git"}
    assert shell_command_names("cat f | jq '.x' | rg foo") == {"cat", "jq", "rg"}
    assert shell_command_names("") == set()


def _eng(ws, home):
    return Engine(cwd=ws, pmb_home=home,
                  config_overrides={"recall.cache_size": 0, "dedup.enable": False})


def test_named_command_rule_fires_for_all_git_invocations(tmp_pmb_home, tmp_workspace_dir):
    eng = _eng(tmp_workspace_dir, tmp_pmb_home)
    eng.record_fact(GIT_RULE, metadata={"kind": "lesson", "source": "lesson"})
    # bare invocations share NO >=4-char token with the rule, yet the command
    # name 'git' matches structurally -> the guard fires (advisory).
    assert pretool_lessons(eng, "git push origin main", set()), "git push must fire"
    assert pretool_lessons(eng, "git status", set()), "git status must fire"
    assert pretool_lessons(eng, "npm install && git commit -m x", set()), "chained git must fire"
    assert pretool_lessons(eng, "VAR=1 /usr/bin/git push", set()), "path/env git must fire"


def test_unrelated_command_does_not_fire(tmp_pmb_home, tmp_workspace_dir):
    eng = _eng(tmp_workspace_dir, tmp_pmb_home)
    eng.record_fact(GIT_RULE, metadata={"kind": "lesson", "source": "lesson"})
    assert not pretool_lessons(eng, "ls -la", set()), "ls must not fire a git rule"
    assert not pretool_lessons(eng, "python script.py", set()), "python must not fire a git rule"


def test_guard_fires_once_per_session(tmp_pmb_home, tmp_workspace_dir):
    eng = _eng(tmp_workspace_dir, tmp_pmb_home)
    eng.record_fact(GIT_RULE, metadata={"kind": "lesson", "source": "lesson"})
    seen: set = set()
    assert pretool_lessons(eng, "git push", seen)
    assert not pretool_lessons(eng, "git commit", seen), "same rule fires once per session"
