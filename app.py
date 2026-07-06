"""
Daily Life Concierge — Multi-Agent System
==========================================
A conversational AI concierge that routes your daily-life requests to
specialist sub-agents: Budget, Meals, and Fitness/Productivity. Built for
the Kaggle "AI Agents: Intensive Vibe Coding" Capstone (Concierge Agents
track).

Architecture
------------
User message --> Orchestrator Agent (intent classification)
                    |--> Budget Agent
                    |--> Meal Agent
                    |--> Fitness/Productivity Agent
                    |--> General Fallback Agent

Each agent has its own system prompt and only receives the context it
needs (data minimization -> security).

LLM backend: HYBRID. Tries a LOCAL Ollama server first (http://localhost:11434)
for free, unlimited, fully offline inference. If Ollama isn't running on the
machine, falls back to the Gemini API (free tier) using a key the user enters
at runtime — so the app works out-of-the-box for anyone, while still rewarding
users who've set up Ollama with zero limits and zero API keys.

Security notes:
 - Ollama path: no API key at all — nothing to leak, all inference local.
 - Gemini fallback path: key entered via st.text_input(type="password"),
   held only in st.session_state, never written to disk or logged.
 - All user input is sanitized before being interpolated into prompts.
 - No personal data leaves the session; there is no external database.

Deployability:
 - Works two ways: (1) fully local via Ollama, zero cloud dependency, or
   (2) deployed to Streamlit Community Cloud with users supplying their own
   free Gemini key. Either path is free and carries no billing risk.
 - See README.md for both deployment paths.
"""

import re
import json
import os
from datetime import datetime

import requests
import streamlit as st
import google.generativeai as genai

# ----------------------------- CONFIG ------------------------------------

st.set_page_config(
    page_title="Daily Life Concierge",
    page_icon="🧭",
    layout="centered",
)

OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODELS = ["llama3.1", "phi3", "mistral"]  # must be `ollama pull`-ed first
GEMINI_MODEL_NAME = "gemini-2.5-flash"

AGENT_ROUTES = {
    "budget": "N · Budget Agent",
    "meal": "E · Meal Agent",
    "study_gym": "S · Fitness & Productivity Agent",
    "general": "W · General Agent",
}

AGENT_AVATARS = {
    "budget": "💰",
    "meal": "🥗",
    "study_gym": "🏋️",
    "general": "🧭",
}

# ----------------------------- SECURITY HELPERS ---------------------------

def sanitize_input(text: str, max_len: int = 800) -> str:
    """Strip control characters, cap length, prevent basic prompt-injection
    patterns from escaping their role as user content."""
    if not text:
        return ""
    text = text[:max_len]
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)  # control chars
    # neutralize attempts to inject fake role markers
    text = text.replace("SYSTEM:", "").replace("ASSISTANT:", "")
    return text.strip()


def check_ollama_available() -> bool:
    """Ping the local Ollama server. Returns False (never throws) if it's
    not running or unreachable — callers fall back to Gemini."""
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        return resp.status_code == 200
    except requests.exceptions.RequestException:
        return False


def call_ollama(prompt: str) -> str:
    """Send a prompt to the local Ollama server and return the generated text.
    num_predict caps response length so replies come back fast even on a
    CPU-only laptop — long unbounded generations are the main cause of the
    app feeling 'stuck' with no feedback."""
    model_name = st.session_state.get("ollama_model", OLLAMA_MODELS[0])
    resp = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": model_name, "prompt": prompt, "stream": False,
            "options": {"num_predict": 180, "temperature": 0.7},
        },
        timeout=120,
    )
    resp.raise_for_status()
    text = resp.json().get("response", "").strip()
    return text or "⚠️ Got an empty response from the model — try again, or switch models in the sidebar."


def call_gemini(prompt: str) -> str:
    """Send a prompt to the Gemini API using the session-scoped key.
    Key is never persisted to disk or logged."""
    api_key = st.session_state.get("api_key", "")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(GEMINI_MODEL_NAME)
    resp = model.generate_content(prompt)
    return resp.text


def llm_backend_status() -> str:
    """Returns 'ollama', 'gemini', or 'none' — which backend is actually usable
    right now, checked fresh each call so status stays accurate as things change."""
    if check_ollama_available():
        return "ollama"
    if st.session_state.get("api_key", "").strip():
        return "gemini"
    return "none"


def call_llm(prompt: str) -> str:
    """Unified dispatcher: try local Ollama first (free, unlimited, offline),
    fall back to Gemini if Ollama isn't running and a key is set. Never
    raises — always returns a user-facing string."""
    backend = llm_backend_status()
    if backend == "ollama":
        try:
            return call_ollama(prompt)
        except requests.exceptions.RequestException as e:
            return f"⚠️ Ollama error: {e}"
    if backend == "gemini":
        try:
            return call_gemini(prompt)
        except Exception as e:
            return f"⚠️ Gemini error: {e}"
    return ("⚠️ No LLM backend available. Either start Ollama (`ollama serve`) "
            "or add a free Gemini API key in the sidebar.")


# ----------------------------- PROFILE PERSISTENCE --------------------------
# Saves preferences + tracked data to a local JSON file next to app.py, so a
# returning user doesn't have to re-onboard every time they restart the app.
# This is safe for the intended single-user local deployment (Ollama on your
# own machine); it is NOT meant for a shared multi-user cloud deployment,
# where session-only state (the old behavior) is actually the correct,
# privacy-preserving default. See README for this trade-off.

PROFILE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_profile.json")
PROFILE_FIELDS = [
    "life_situation", "diet_pref", "currency_symbol", "focus_areas",
    "goal_note", "monthly_budget", "ollama_model",
    "expenses", "logged_meals", "tasks",
]


def load_profile() -> dict:
    if not os.path.exists(PROFILE_FILE):
        return {}
    try:
        with open(PROFILE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save_profile():
    data = {k: st.session_state.get(k) for k in PROFILE_FIELDS}
    try:
        with open(PROFILE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except OSError:
        pass  # non-fatal — app still works, just won't persist this run


def reset_profile():
    if os.path.exists(PROFILE_FILE):
        try:
            os.remove(PROFILE_FILE)
        except OSError:
            pass
    for k in PROFILE_FIELDS + ["chat_history"]:
        if k in st.session_state:
            del st.session_state[k]


# ----------------------------- STATE ---------------------------------------

DIET_OPTIONS = ["No preference", "Vegan", "Vegetarian", "Eggetarian", "Non-vegetarian", "Keto", "Gluten-free"]
CURRENCY_OPTIONS = {"₹ INR": "₹", "$ USD": "$", "€ EUR": "€", "£ GBP": "£"}
FOCUS_OPTIONS = ["Budget & Expenses", "Meals & Nutrition", "Fitness", "Study / Work Productivity"]
LIFE_SITUATION_OPTIONS = [
    "Student", "Working professional / employee", "Business owner / founder",
    "Freelancer", "Daily-wage / manual worker", "Between jobs / preparing for something",
    "Homemaker", "Retired", "Other",
]


def init_state():
    saved = load_profile()
    defaults = {
        "api_key": "",
        "ollama_model": OLLAMA_MODELS[0],
        "onboarded": False,
        "life_situation": "Student",
        "diet_pref": "No preference",
        "currency_symbol": "₹",
        "focus_areas": FOCUS_OPTIONS.copy(),  # which domains this user actually wants
        "goal_note": "",                       # free-text: their own words on what they want help with
        "chat_history": [],       # list of {role, content, agent}
        "expenses": [],           # list of {item, amount, date}
        "monthly_budget": 6000.0, # user-editable, in their chosen currency
        "logged_meals": [],
        "tasks": [],              # fitness/study habit log
    }
    # overlay any previously saved profile fields on top of the hard defaults
    for k in PROFILE_FIELDS:
        if k in saved:
            defaults[k] = saved[k]
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()

# ----------------------------- ORCHESTRATOR AGENT ---------------------------

def classify_intent(user_msg: str) -> str:
    """Lightweight rule-based + keyword orchestrator.
    Falls back to 'general' agent if nothing matches, OR if the matched
    agent's domain wasn't selected by the user during onboarding (respects
    each user's own chosen focus areas rather than forcing all domains on
    everyone).
    (Kept rule-based for speed/determinism/zero extra API cost; can be
    swapped for an LLM-based router by calling call_ollama() here instead.)"""
    msg = user_msg.lower()
    active = st.session_state.focus_areas

    budget_kw = ["spent", "spend", "expense", "budget", "money", "cost", "price", "save money", "afford"]
    meal_kw = ["eat", "meal", "food", "recipe", "cook", "breakfast", "lunch", "dinner", "diet", "snack"]
    fitness_study_kw = ["gym", "workout", "exercise", "study", "leetcode", "gre", "cgpa", "habit", "schedule", "routine", "reminder", "work", "productivity"]

    if any(k in msg for k in budget_kw) and "Budget & Expenses" in active:
        return "budget"
    if any(k in msg for k in meal_kw) and "Meals & Nutrition" in active:
        return "meal"
    if any(k in msg for k in fitness_study_kw) and (
        "Fitness" in active or "Study / Work Productivity" in active
    ):
        return "study_gym"
    return "general"


# ----------------------------- SUB-AGENTS -----------------------------------

def budget_agent(user_msg: str, backend_available: bool) -> str:
    sym = st.session_state.currency_symbol
    total_spent = sum(e["amount"] for e in st.session_state.expenses)
    remaining = st.session_state.monthly_budget - total_spent

    match = re.search(r"(?:spent|spend)\s*(?:[₹$€£]|rs\.?)?\s*(\d+(?:\.\d+)?)\s*(?:on|for)?\s*(.*)", user_msg, re.I)
    logged_note = ""
    if match:
        amount = float(match.group(1))
        item = match.group(2).strip() or "unspecified"
        st.session_state.expenses.append({
            "item": item, "amount": amount,
            "date": datetime.now().strftime("%Y-%m-%d")
        })
        remaining -= amount
        logged_note = f"\n\n(Logged {sym}{amount:.0f} for '{item}'.)"

    context = (
        f"Monthly budget: {sym}{st.session_state.monthly_budget:.0f}. "
        f"Spent so far: {sym}{total_spent:.0f}. Remaining: {sym}{remaining:.0f}."
    )

    if not backend_available:
        return (f"[Budget Agent — no LLM backend]\n{context}{logged_note}\n"
                f"Start Ollama or add a Gemini key in the sidebar for personalized advice.")

    prompt = (
        "You are a concise, practical personal budget assistant. Be encouraging but "
        "honest. Keep replies under 80 words. Use the currency symbol given, don't "
        "assume any other currency.\n"
        f"User's stated goal (if any): {sanitize_input(st.session_state.goal_note) or 'not specified'}\n"
        f"Context: {context}\n"
        f"User: {sanitize_input(user_msg)}"
    )
    reply = call_llm(prompt)
    return reply + logged_note


def meal_agent(user_msg: str, backend_available: bool) -> str:
    if not backend_available:
        return "[Meal Agent — no LLM backend] Start Ollama or add a Gemini key in the sidebar."

    sym = st.session_state.currency_symbol
    diet = st.session_state.diet_pref
    remaining = st.session_state.monthly_budget - sum(e["amount"] for e in st.session_state.expenses)
    prompt = (
        "You are a meal-planning assistant. Respect the user's stated dietary preference "
        "strictly — do not suggest anything outside it. Suggest simple, budget-friendly, "
        "reasonably healthy meals. Keep it under 100 words, use bullet points if listing items.\n"
        f"User's dietary preference: {diet}.\n"
        f"User's remaining monthly budget: {sym}{remaining:.0f}.\n"
        f"User: {sanitize_input(user_msg)}"
    )
    reply = call_llm(prompt)
    st.session_state.logged_meals.append({"query": user_msg, "date": datetime.now().strftime("%Y-%m-%d %H:%M")})
    return reply


def study_gym_agent(user_msg: str, backend_available: bool) -> str:
    st.session_state.tasks.append({"note": user_msg, "date": datetime.now().strftime("%Y-%m-%d %H:%M")})

    if not backend_available:
        return "[Fitness/Productivity Agent — no LLM backend] Start Ollama or add a Gemini key in the sidebar."

    prompt = (
        "You are a disciplined but supportive accountability coach. The user's own words "
        "about their fitness/study/work goals are given below (may be blank) — tailor your "
        "advice to what they've actually told you rather than assuming a fixed routine. "
        "Their life situation is also given — a founder's 'productivity' looks different from "
        "a student's or a daily-wage worker's, so adapt accordingly. "
        "Give short, actionable, motivating replies under 80 words.\n"
        f"User's life situation: {st.session_state.life_situation}.\n"
        f"User's stated goal (if any): {sanitize_input(st.session_state.goal_note) or 'not specified'}\n"
        f"User: {sanitize_input(user_msg)}"
    )
    return call_llm(prompt)


def general_agent(user_msg: str, backend_available: bool) -> str:
    if not backend_available:
        return "[General Agent — no LLM backend] Start Ollama or add a Gemini key in the sidebar to chat."
    prompt = (
        "You are a warm, concise daily-life concierge assistant, useful to anyone regardless "
        "of profession — student, business owner, employee, freelancer, or manual worker. "
        "Don't assume a specific lifestyle beyond what the user tells you. "
        "Keep replies under 80 words.\n"
        f"User's life situation: {st.session_state.life_situation}.\n"
        f"User: {sanitize_input(user_msg)}"
    )
    return call_llm(prompt)


AGENT_FUNCS = {
    "budget": budget_agent,
    "meal": meal_agent,
    "study_gym": study_gym_agent,
    "general": general_agent,
}

# ----------------------------- UI -------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500&display=swap');

html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }

/* subtle depth instead of a flat fill — cinematic rather than dashboard-flat */
.stApp {
    background: radial-gradient(ellipse 90% 60% at 50% -10%, #E3F0EA 0%, #F1F7F4 55%, #F1F7F4 100%);
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes glowPulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(212,160,23,0.0); }
    50%      { box-shadow: 0 0 0 5px rgba(212,160,23,0.14); }
}

.hero-eyebrow {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem;
    letter-spacing: 0.18em; text-transform: uppercase; color: #D4A017;
    animation: fadeInUp 0.5s ease-out both;
}
.concierge-header {
    display: flex; align-items: baseline; gap: 0.7rem; flex-wrap: wrap;
    margin: 0.15rem 0 0.2rem 0;
    animation: fadeInUp 0.6s ease-out both;
}
.concierge-header h1 {
    font-family: 'Fraunces', serif; font-weight: 700; font-size: 2.65rem;
    color: #0D3B36; margin: 0; letter-spacing: -0.02em; line-height: 1.05;
}
.concierge-header .bearing {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.82rem;
    color: #D4A017; background: #FFF7E0; padding: 0.2rem 0.6rem;
    border-radius: 999px; border: 1px solid #EEDCA0;
    animation: glowPulse 3.5s ease-in-out infinite;
}
.concierge-tagline {
    color: #4A5D5C; font-size: 1.02rem; margin-bottom: 1.4rem; max-width: 640px;
    animation: fadeInUp 0.7s ease-out both;
}

.waypoint-row { display: flex; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 1.1rem; }
.waypoint-chip {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem;
    color: #0D3B36; background: #E8F3EF; border: 1px solid #C9E4DA;
    padding: 0.28rem 0.7rem; border-radius: 999px;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.waypoint-chip:hover { transform: translateY(-1px); box-shadow: 0 3px 10px rgba(13,59,54,0.10); }

.snapshot-grid { display: flex; gap: 0.7rem; margin: 0.4rem 0 1rem 0; flex-wrap: wrap; }
.snapshot-card {
    flex: 1; min-width: 140px; background: #FFFFFF; border: 1px solid #E3ECE8;
    border-radius: 14px; padding: 0.75rem 0.95rem;
    transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
}
.snapshot-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(13,59,54,0.10);
    border-color: #D4A017;
}
.snapshot-card .label {
    font-size: 0.72rem; color: #6B8280; text-transform: uppercase;
    letter-spacing: 0.05em; margin-bottom: 0.25rem;
}
.snapshot-card .value {
    font-family: 'IBM Plex Mono', monospace; font-size: 1.35rem;
    color: #0D3B36; font-weight: 500;
}
.snapshot-card .sub { font-size: 0.75rem; color: #8A9C99; margin-top: 0.15rem; }

/* chat bubbles: soft entrance + a touch more editorial than a generic dashboard */
div[data-testid="stChatMessage"] {
    border-radius: 16px;
    animation: fadeInUp 0.35s ease-out both;
    transition: box-shadow 0.15s ease;
}
div[data-testid="stChatMessage"]:hover { box-shadow: 0 4px 16px rgba(13,59,54,0.06); }

/* buttons: quick-actions + sidebar buttons get a lift on hover instead of a flat click */
.stButton button {
    border-radius: 10px !important;
    transition: transform 0.12s ease, box-shadow 0.12s ease !important;
}
.stButton button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 16px rgba(13,59,54,0.14);
}

/* ---- sidebar containers (st.container(border=True)) as colorful animated cards ---- */
[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 14px !important;
    animation: fadeInUp 0.5s ease-out both;
    transition: transform 0.18s ease, box-shadow 0.18s ease;
}
[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(13,59,54,0.08);
}
/* Settings card — gold accent, 1st card in the sidebar */
[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"]:nth-of-type(1) {
    background: #FFFBF0 !important; border: 1px solid #EEDCA0 !important;
    border-left: 4px solid #D4A017 !important;
    animation-delay: 0.05s;
}
/* Profile card — teal accent, 2nd card */
[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"]:nth-of-type(2) {
    background: #F3FAF7 !important; border: 1px solid #C9E4DA !important;
    border-left: 4px solid #0D3B36 !important;
    animation-delay: 0.15s;
}
/* Snapshot card — soft rose accent, 3rd card, for visual variety */
[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"]:nth-of-type(3) {
    background: #FBF5F6 !important; border: 1px solid #EBD3D7 !important;
    border-left: 4px solid #B5657A !important;
    animation-delay: 0.25s;
}
[data-testid="stSidebar"] {
    background: #FBFDFC !important;
    border-right: 1px solid #E3ECE8;
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    font-family: 'Fraunces', serif !important; color: #0D3B36 !important;
}
[data-testid="stSidebar"] label, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span {
    color: #3D5450 !important;
}

/* form widgets: selectbox / text input / textarea — consistent rounded, teal-on-focus look */
[data-baseweb="select"] > div,
.stTextInput input,
.stTextArea textarea,
.stNumberInput input {
    border-radius: 10px !important;
    border-color: #D3E4DC !important;
    background-color: #FFFFFF !important;
}
[data-baseweb="select"] > div:focus-within,
.stTextInput input:focus,
.stTextArea textarea:focus,
.stNumberInput input:focus {
    border-color: #D4A017 !important;
    box-shadow: 0 0 0 1px #D4A017 !important;
}

/* multiselect tags default to Streamlit's red — recolor to match the palette */
[data-baseweb="tag"] {
    background-color: #0D3B36 !important;
    border-radius: 8px !important;
}
[data-baseweb="tag"] span { color: #FFFFFF !important; }
[data-baseweb="tag"] svg { fill: #F0C955 !important; }

/* radio / selectbox dropdown menu items on hover */
[role="option"]:hover { background-color: #E8F3EF !important; }

/* progress bar accent already follows theme's primaryColor (gold) via config.toml */

/* empty-state card shown before the first message */
.empty-state {
    text-align: center; padding: 2.2rem 1.5rem; margin: 1.5rem 0;
    border: 1px dashed #C9E4DA; border-radius: 16px; background: rgba(255,255,255,0.5);
    color: #6B8280; font-size: 0.92rem;
}
.empty-state .big { font-size: 1.8rem; margin-bottom: 0.4rem; }

/* custom scrollbar for a slightly more considered feel */
::-webkit-scrollbar { width: 10px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #C9E4DA; border-radius: 8px; }
::-webkit-scrollbar-thumb:hover { background: #A8CFC0; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-eyebrow">CONCIERGE SYSTEM · MULTI-AGENT</div>
<div class="concierge-header">
  <h1>🧭 Daily Life Concierge</h1>
  <span class="bearing">N·E·S·W</span>
</div>
<div class="concierge-tagline">
  One assistant, four directions — Budget, Meals, Fitness, and General —
  routed automatically based on what you ask. Adapts to whoever you are.
</div>
""", unsafe_allow_html=True)

def bordered_container():
    """st.container(border=True) needs Streamlit >=1.32. Falls back gracefully
    on older installs instead of crashing the whole app."""
    try:
        return st.container(border=True)
    except TypeError:
        return st.container()


with st.sidebar:
    st.header("⚙️ Settings")
    settings_card = bordered_container()
    with settings_card:
        ollama_up = check_ollama_available()
        if ollama_up:
            st.success("🟢 Ollama connected (local, free, no limits)")
            st.session_state.ollama_model = st.selectbox(
                "Local model", OLLAMA_MODELS,
                index=OLLAMA_MODELS.index(st.session_state.ollama_model),
                help="Must be pulled first with `ollama pull <model>`."
            )
        else:
            st.warning("🟡 Ollama not detected — using Gemini fallback (needs a key below)")
            with st.expander("Want unlimited & offline? Set up Ollama"):
                st.caption("Install from ollama.com, run `ollama pull llama3.1`, "
                           "then `ollama serve`. Refresh this page once it's running.")
            st.session_state.api_key = st.text_input(
                "Gemini API Key (fallback)", type="password", value=st.session_state.api_key,
                help="Free at aistudio.google.com/app/apikey. Session-only, never saved to disk."
            )

        backend = llm_backend_status()
        backend_label = {"ollama": "🟢 Ollama (local)", "gemini": "🔵 Gemini (cloud)", "none": "🔴 none"}[backend]
        st.caption(f"Active backend: {backend_label}")

    st.subheader("👤 Tell it about you")
    profile_card = bordered_container()
    with profile_card:
        st.session_state.life_situation = st.selectbox(
            "Which best describes you?", LIFE_SITUATION_OPTIONS,
            index=LIFE_SITUATION_OPTIONS.index(st.session_state.life_situation),
            help="Advice is tailored to this — a founder's routine looks nothing like a student's."
        )
        st.session_state.goal_note = st.text_area(
            "In your own words, what do you want help with?", value=st.session_state.goal_note,
            placeholder="e.g. 'saving for a business loan', 'staying consistent at the gym after night shifts'...",
            height=70,
        )
        st.session_state.focus_areas = st.multiselect(
            "Which areas do you actually want this to cover?", FOCUS_OPTIONS,
            default=st.session_state.focus_areas,
            help="Uncheck anything irrelevant to you — e.g. skip 'Study' if you're not studying."
        )
        diet_choice = st.selectbox(
            "Dietary preference", DIET_OPTIONS,
            index=DIET_OPTIONS.index(st.session_state.diet_pref),
        )
        st.session_state.diet_pref = diet_choice
        currency_labels = list(CURRENCY_OPTIONS.keys())
        current_label = next(
            (lbl for lbl, sym in CURRENCY_OPTIONS.items() if sym == st.session_state.currency_symbol),
            currency_labels[0]
        )
        currency_label = st.selectbox("Currency", currency_labels, index=currency_labels.index(current_label))
        st.session_state.currency_symbol = CURRENCY_OPTIONS[currency_label]
        st.session_state.monthly_budget = st.number_input(
            f"Monthly budget ({st.session_state.currency_symbol})", min_value=0.0,
            value=st.session_state.monthly_budget, step=100.0
        )

    st.subheader("📊 Snapshot")
    snapshot_card = bordered_container()
    with snapshot_card:
        sym = st.session_state.currency_symbol
        total_spent = sum(e["amount"] for e in st.session_state.expenses)
        remaining = st.session_state.monthly_budget - total_spent
        st.markdown(f"""
        <div class="snapshot-grid">
          <div class="snapshot-card">
            <div class="label">Spent</div>
            <div class="value">{sym}{total_spent:.0f}</div>
            <div class="sub">{sym}{remaining:.0f} left of {sym}{st.session_state.monthly_budget:.0f}</div>
          </div>
          <div class="snapshot-card">
            <div class="label">Meals logged</div>
            <div class="value">{len(st.session_state.logged_meals)}</div>
          </div>
          <div class="snapshot-card">
            <div class="label">Check-ins</div>
            <div class="value">{len(st.session_state.tasks)}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
        if st.session_state.monthly_budget > 0:
            st.progress(min(max(total_spent / st.session_state.monthly_budget, 0.0), 1.0))

    if st.button("🗑️ Reset my profile & data"):
        reset_profile()
        st.rerun()

    st.divider()
    st.caption("💾 Your profile and tracked data are saved locally in "
               "`user_profile.json` next to app.py, so they persist across restarts.")
    st.caption("🔒 Security: Gemini key (if used) is never saved to disk, only kept in-session. "
               "This app is meant for single-user local use — see README for the multi-user caveat.")
    st.caption("🚀 Deployability: local Ollama for unlimited free use, or Gemini fallback anywhere.")

# persist any changes made via the sidebar widgets above, every rerun
save_profile()

# waypoint chips — show which domains are actually active for this user
active_chip_map = {
    "Budget & Expenses": "N · Budget", "Meals & Nutrition": "E · Meals",
    "Fitness": "S · Fitness", "Study / Work Productivity": "S · Productivity",
}
chips_html = "".join(
    f'<span class="waypoint-chip">{active_chip_map[a]}</span>'
    for a in st.session_state.focus_areas if a in active_chip_map
)
if chips_html:
    st.markdown(f'<div class="waypoint-row">{chips_html}</div>', unsafe_allow_html=True)

# chat history render
if not st.session_state.chat_history:
    st.markdown("""
    <div class="empty-state">
      <div class="big">🧭</div>
      Nothing logged yet — tap a quick action above, or type below to get started.
    </div>
    """, unsafe_allow_html=True)

for turn in st.session_state.chat_history:
    avatar = AGENT_AVATARS.get(turn.get("agent"), "🙂") if turn["role"] == "assistant" else "🙂"
    with st.chat_message(turn["role"], avatar=avatar):
        if turn["role"] == "assistant":
            st.caption(AGENT_ROUTES.get(turn.get("agent"), ""))
        st.write(turn["content"])


def handle_message(clean_msg: str):
    """Shared path for both typed chat input and quick-action buttons."""
    st.session_state.chat_history.append({"role": "user", "content": clean_msg})
    intent = classify_intent(clean_msg)
    backend_available = llm_backend_status() != "none"
    reply = AGENT_FUNCS[intent](clean_msg, backend_available)
    st.session_state.chat_history.append({"role": "assistant", "content": reply, "agent": intent})
    save_profile()


# quick-action buttons — only show ones relevant to the user's selected focus areas,
# so this stays useful whether they picked one domain or all four
quick_actions = []
if "Budget & Expenses" in st.session_state.focus_areas:
    quick_actions.append(("💰 Log an expense", "I want to log an expense"))
if "Meals & Nutrition" in st.session_state.focus_areas:
    quick_actions.append(("🥗 Suggest a meal", "Suggest a meal for me right now"))
if "Fitness" in st.session_state.focus_areas or "Study / Work Productivity" in st.session_state.focus_areas:
    quick_actions.append(("🏋️ Quick check-in", "I want to check in on my progress today"))
quick_actions.append(("💬 Something else", "I need some general help"))

st.session_state.setdefault("processing", False)

qa_cols = st.columns(len(quick_actions))
for col, (label, prompt_text) in zip(qa_cols, quick_actions):
    if col.button(label, use_container_width=True, disabled=st.session_state.processing, key=f"qa_{label}"):
        st.session_state.processing = True
        with st.spinner(f"{label} — thinking... (first response can take 10-30s on a laptop CPU)"):
            handle_message(prompt_text)
        st.session_state.processing = False
        st.rerun()

user_msg = st.chat_input("Ask about budget, meals, gym, study — anything daily-life...")

if user_msg:
    clean_msg = sanitize_input(user_msg)
    with st.chat_message("user", avatar="🙂"):
        st.write(clean_msg)
    intent_preview = classify_intent(clean_msg)
    with st.chat_message("assistant", avatar=AGENT_AVATARS.get(intent_preview, "🙂")):
        st.caption(AGENT_ROUTES[intent_preview])
        with st.spinner(f"{AGENT_ROUTES[intent_preview]} thinking..."):
            handle_message(clean_msg)
        st.write(st.session_state.chat_history[-1]["content"])
