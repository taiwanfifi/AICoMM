# Claude Code Memory Transfer

## For New Claude Code on a New Computer

When you clone this repo and start Claude Code for the first time, do this:

```
請讀 .claude-memory/MEMORY.md，然後把內容設定到你的 memory 裡。
```

Claude Code will:
1. Read the accumulated knowledge from previous sessions
2. Copy it to its local memory directory (`~/.claude/projects/.../memory/`)
3. Immediately have context about all experiments, technical pitfalls, and project status

## What's in MEMORY.md?

- **Technical pitfalls** we've encountered (FP16 overflow, RoPE bugs, quantization issues)
- **All experiment results** with JSON file references
- **Paper status** and submission targets
- **Key numbers** for quick reference

## Other Important Files to Read First

1. `CLAUDE.md` — Project directives (auto-loaded by Claude Code)
2. `ONBOARDING.md` — Complete research guide for newcomers
3. `docs/PROJECT_STATUS.md` — Current state of everything

## Note

The `CLAUDE.md` file is automatically loaded by Claude Code on every conversation.
The memory file needs to be manually read once on a new machine.
