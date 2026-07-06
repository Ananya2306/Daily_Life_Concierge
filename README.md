# 🧭 Daily Life Concierge - A Multi Agent Personal Assistant

**Kaggle AI Agents: Intensive Vibe Coding Capstone - Concierge Agents Track**

## Problem

Everyone's daily life is different a student, a founder, a salaried employee,
a freelancer, and a daily-wage worker all need help with budgeting, food, and
staying on track, but in completely different ways. Most budgeting/habit apps
hard-code one persona (usually "young urban professional") and don't adapt.

## Solution

**Daily Life Concierge** is a single conversational agent system that first
asks who you are and what you actually want help with your life situation,
dietary preference, currency, and which domains matter to you then routes
every message through an **orchestrator agent** to a **specialist sub-agent**
that tailors its answer to your stated context, not an assumed one.

Uncheck "Study / Work Productivity" if you're not a student. Pick "Non-vegetarian"
if you're not vegan. Set your own currency and budget. The agents only speak to
what you told them.

## Architecture

```
                     ┌─────────────────────┐
   User message ───▶ │  Orchestrator Agent  │
                     │  (intent classifier) │
                     └──────────┬───────────┘
                                │
        ┌───────────────┬──────┴───────┬────────────────┐
        ▼               ▼              ▼                 ▼
 ┌─────────────┐ ┌─────────────┐ ┌───────────────┐ ┌─────────────┐
 │Budget Agent │ │ Meal Agent  │ │Fitness/Product│ │General Agent│
 │(expense log,│ │(diet-aware  │ │Agent (habit   │ │  (fallback) │
 │ tracking)   │ │ suggestions)│ │logging, coach)│ │             │
 └──────┬──────┘ └──────┬──────┘ └───────┬───────┘ └──────┬──────┘
        └───────────────┴──────────────┴────────────────┘
                                │
                     ┌──────────▼───────────┐
                     │   call_llm() router   │
                     │  1. try local Ollama  │
                     │  2. else Gemini API   │
                     └────────────────────────┘
```

- **Orchestrator**: deterministic keyword-based router (fast, free, no extra
  LLM call swappable for an LLM-based classifier).
- **Sub-agents**: each builds a narrow, role-specific prompt and only the
  context it needs (data minimization), then calls the shared `call_llm()`
  dispatcher.
- **Hybrid LLM backend**: `call_llm()` checks if a local Ollama server is
  running if so, uses it (free, unlimited, fully offline). If not, it falls
  back to the Gemini API using a key the user supplies. This means the app
  works immediately for anyone (via Gemini), while rewarding users who set up
  Ollama with zero limits and zero API keys.
- **Tools**: the Budget Agent has a lightweight regex "tool" that parses
  natural language ("spent 200 on groceries") and logs a structured expense
  entry a simple example of an agent taking action, not just chatting.

<img width="2179" height="1480" alt="architecture_diagram" src="https://github.com/user-attachments/assets/4664539b-ff20-4d31-b8ae-41cb07b77e83" />


## Concepts demonstrated (rubric mapping)

| Concept | Where |
|---|---|
| Multi-agent system | `app.py` - orchestrator + 4 specialist agents |
| Security features | `sanitize_input()`, session-only Gemini key (only needed as fallback), no server-side persistence |
| Deployability | Two working paths: fully local/offline via Ollama (zero cost, zero limits), or Streamlit Cloud + free Gemini key (zero setup for the end user) |

## Setup

**Option A - Ollama (free, unlimited, fully offline, one-time setup):**
```bash
# 1. Install Ollama: https://ollama.com/download
# 2. Pull a model
ollama pull llama3.1      # ~4.7GB, best quality/speed balance
# 3. Start the server (leave running in a terminal)
ollama serve
```

**Option B — Gemini fallback (works instantly, no install, free tier limits apply):**
Get a free key at https://aistudio.google.com/app/apikey and paste it into
the sidebar when Ollama isn't detected.

**Then, either way:**
```bash
git clone <your-repo-url>
cd daily-concierge
pip install -r requirements.txt
streamlit run app.py
```

The sidebar shows which backend is active. If Ollama is running, it's used
automatically no key needed. If not, you'll be prompted for a Gemini key.

## Switching Between Ollama and Gemini (manual test)
 
The app picks a backend automatically — but if you want to force-test the
Gemini fallback path (e.g. to confirm both backends genuinely work), you
need to manually stop Ollama first.
 
**Windows:**
```bash
# Stop Ollama
taskkill /IM ollama.exe /F
taskkill /IM "ollama app.exe" /F
 
# Confirm it's stopped — this should fail/refuse a connection
curl http://localhost:11434
```
Then refresh the app in your browser. The sidebar will switch to
"🟡 Ollama not detected — using Gemini fallback" and show a field to paste
a Gemini API key.
 
**Mac/Linux:**
```bash
pkill ollama
```
 
**To switch back to Ollama** (for free/unlimited daily use):
```bash
ollama serve
```
Then refresh the app again — it'll detect Ollama and switch back
automatically, no restart of the app itself needed.

## Deployment

- **Fully local**: run on your own machine with Ollama zero cost, zero
  limits, zero cloud dependency. Best for personal daily use.
- **Streamlit Community Cloud**: push this repo to GitHub, deploy on
  [share.streamlit.io](https://share.streamlit.io) since Ollama can't run
  there, it'll automatically use the Gemini fallback, so end users just paste
  their own free key. Zero server cost, zero billing risk either way.

## Security notes

- **Ollama path**: no API key at all nothing to leak, all inference local.
- **Gemini fallback path**: key entered via a password-masked field, held
  only in `st.session_state` for the browser tab, never written to disk or
  logged.
- All user input passed through `sanitize_input()` before being interpolated
  into any prompt (strips control characters, caps length, neutralizes fake
  role markers to reduce prompt-injection risk).
- **Profile & tracked data persistence**: preferences (life situation, diet,
  currency, focus areas, goal note) and tracked data (expenses, meals, habit
  logs) are saved to a local `user_profile.json` file next to `app.py`, so a
  returning user doesn't have to re-onboard every session. A "Reset my
  profile & data" button in the sidebar clears it.
  - This is safe and appropriate for the app's intended use **single-user,
    local deployment** (you running it on your own machine via Ollama).
  - It is **not** appropriate for a shared multi-user cloud deployment,
    since every visitor would read/write the same file. If you deploy this
    to Streamlit Community Cloud for multiple people to use, either add
    proper per-user accounts, or intentionally revert to session-only state
    (remove the `save_profile()`/`load_profile()` calls) so each visitor's
    data stays private to their browser tab and is discarded on refresh.
  - `user_profile.json` is included in `.gitignore` so your personal data
    never gets committed to the public repo.

## Future extensions

- Swap the keyword router for an LLM/ADK-based orchestrator for fuzzier intent
  classification.
- Add an MCP server connector (e.g., Google Calendar) so the Fitness/Productivity
  Agent can actually create calendar reminders instead of just logging locally.
- Persist history opt-in via encrypted local storage for returning users.
- Add more local model options and auto-detect which are already pulled via
  `ollama list`, instead of a fixed dropdown.
