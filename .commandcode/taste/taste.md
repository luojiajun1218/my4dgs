# Taste — Command Code

- Communicates in Chinese and expects replies, status updates, and explanations in Chinese. Confidence: 0.8
- On handoff ("读取下…的进度，然后继续"), expects the agent to reconstruct prior state on its own — git log, docs/experiment records, tmp/ task-state files, even other agents' session logs — and then continue the work autonomously instead of re-planning from scratch or asking what to do first. Confidence: 0.65
- When asking for an explanation, wants "what was actually done + what it accomplished": concrete experiment outcomes, measured numbers, and an explicit list of what was ruled out and what is still open — not a narration of tool calls. Confidence: 0.6
