# streamlit_tz_to_telegram_app.py
# -*- coding: utf-8 -*-
"""
Streamlit: Генератор ТЗ из идей → структурное ТЗ → отправка в Telegram

Зависимости: streamlit, openai>=1.0.0,<3, requests
Запуск:     streamlit run streamlit_tz_to_telegram_app.py

Секреты (.streamlit/secrets.toml) — поддерживаются оба варианта:

# ВАРИАНТ 1 (в корне):
OPENAI_API_KEY      = "sk-..."
OPENAI_MODEL        = "gpt-4o-mini"
TELEGRAM_BOT_TOKEN  = "8427...:AAG..."
TELEGRAM_CHAT_ID    = "489408957"

# ВАРИАНТ 2 (в секции [telegram]; как на вашем скриншоте):
[telegram]
TELEGRAM_BOT_TOKEN  = "8427...:AAG..."
TELEGRAM_CHAT_ID    = "489408957"

# (Также поддерживаются bot_token / default_chat_id / chat_id и др. эквиваленты.)
"""

from __future__ import annotations
import re
import json
from typing import List, Dict, Any, Optional

import streamlit as st
import requests

# ===== OpenAI SDK (v1+) =====
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # покажем ошибку ниже

# ===== UI =====
st.set_page_config(page_title="Генератор ТЗ → Telegram", page_icon="📝", layout="centered")
st.title("📝 Генератор ТЗ для промпт-инженера → Telegram")
st.caption("Вставьте идею или черновик ТЗ, ответьте на уточняющие вопросы, утвердите и отправьте в нужный отдел в Telegram.")

# ===== Secrets / Settings =====
OPENAI_API_KEY: Optional[str] = st.secrets.get("OPENAI_API_KEY")
OPENAI_MODEL_DEFAULT: str = st.secrets.get("OPENAI_MODEL", "gpt-4o-mini")

TELEGRAM_CONF = st.secrets.get("telegram", {}) or {}

def _get_secret_any(*names: str) -> Optional[str]:
    """Ищем ключи и в корне secrets, и в секции [telegram], без учёта регистра."""
    # 1) прямое совпадение
    for n in names:
        v = st.secrets.get(n)
        if v:
            return str(v)
        if isinstance(TELEGRAM_CONF, dict) and TELEGRAM_CONF.get(n):
            return str(TELEGRAM_CONF.get(n))
    # 2) case-insensitive
    root_lower = {k.lower(): v for k, v in dict(st.secrets).items() if not isinstance(v, dict)}
    tg_lower   = {k.lower(): v for k, v in dict(TELEGRAM_CONF).items()}
    for n in names:
        ln = n.lower()
        if ln in root_lower and root_lower[ln]:
            return str(root_lower[ln])
        if ln in tg_lower and tg_lower[ln]:
            return str(tg_lower[ln])
    return None

TG_TOKEN: Optional[str] = _get_secret_any("TELEGRAM_BOT_TOKEN", "BOT_TOKEN", "bot_token")
TG_DEFAULT_CHAT: Optional[str] = _get_secret_any("TELEGRAM_CHAT_ID", "TELEGRAM_DEFAULT_CHAT_ID", "CHAT_ID", "default_chat_id", "chat_id")
DEPT_MAP: Dict[str, str] = TELEGRAM_CONF.get("departments", {}) or {}

# ===== Session =====
if "_init" not in st.session_state:
    st.session_state._init = True
    st.session_state.stage = "input"   # input -> questions -> draft
    st.session_state.initial_text = ""
    st.session_state.questions: List[str] = []
    st.session_state.answers: Dict[int, str] = {}
    st.session_state.tz_markdown = ""
    st.session_state.selected_dept = None
    st.session_state.requester = ""

# ===== Config (без сайдбара) =====
model_name = OPENAI_MODEL_DEFAULT
TEMPERATURE = 0.2

# ===== Guards =====
if not OPENAI_API_KEY:
    st.error("Не найден OPENAI_API_KEY в secrets.")
if OpenAI is None:
    st.error("Пакет openai не установлен. Установите:  pip install openai")

# ===== OpenAI helpers =====
@st.cache_resource(show_spinner=False)
def _openai_client():
    return OpenAI(api_key=OPENAI_API_KEY)

def call_chat_completion(
    messages: List[Dict[str, Any]],
    temperature: float = TEMPERATURE,
    max_new_tokens: int = 1200,
) -> str:
    """
    Некоторые модели не принимают `max_tokens` (требуют `max_completion_tokens` или `max_output_tokens`).
    Делаем последовательные попытки и возвращаем контент при первом успехе.
    """
    client = _openai_client()
    last_err: Optional[Exception] = None
    for extra in (
        {"max_completion_tokens": max_new_tokens},
        {"max_output_tokens": max_new_tokens},
        {"max_tokens": max_new_tokens},
        {},  # на крайний — без лимита
    ):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                **extra,
            )
            content = resp.choices[0].message.content or ""
            return content.strip()
        except Exception as e:
            last_err = e
            continue
    if last_err:
        st.error(f"Ошибка OpenAI: {last_err}")
    return ""

# ===== Telegram helpers =====
def chunk_for_tg(text: str, limit: int = 4000) -> List[str]:
    text = text.strip()
    if len(text) <= limit:
        return [text]
    parts: List[str] = []
    current: List[str] = []
    size = 0
    for para in text.split("\n\n"):
        block = para.strip() + "\n\n"
        if size + len(block) > limit and current:
            parts.append("".join(current).rstrip())
            current, size = [block], len(block)
        else:
            current.append(block); size += len(block)
    if current:
        parts.append("".join(current).rstrip())
    fixed: List[str] = []
    for p in parts:
        if len(p) <= limit:
            fixed.append(p); continue
        buf: List[str] = []
        tally = 0
        for ln in p.splitlines(keepends=True):
            if tally + len(ln) > limit and buf:
                fixed.append("".join(buf).rstrip()); buf, tally = [ln], len(ln)
            else:
                buf.append(ln); tally += len(ln)
        if buf:
            fixed.append("".join(buf).rstrip())
    return fixed

def send_to_telegram(text: str, chat_id: Optional[str] = None):
    if not TG_TOKEN:
        st.error("Не найден telegram.bot_token / TELEGRAM_BOT_TOKEN / BOT_TOKEN в secrets.")
        return []
    target_chat = chat_id or TG_DEFAULT_CHAT
    if not target_chat:
        st.error("Не задан chat_id (telegram.default_chat_id | telegram.chat_id | TELEGRAM_CHAT_ID | TELEGRAM_DEFAULT_CHAT_ID | CHAT_ID).")
        return []
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    results = []
    for chunk in chunk_for_tg(text):
        r = requests.post(url, data={"chat_id": target_chat, "text": chunk, "disable_web_page_preview": True}, timeout=30)
        results.append(r)
        if r.status_code != 200:
            st.error(f"Telegram ошибка {r.status_code}: {r.text}")
            break
    return results

def reset_to_home():
    """Сброс состояния и возврат на главный экран."""
    st.session_state.stage = "input"
    st.session_state.initial_text = ""
    st.session_state.questions = []
    st.session_state.answers = {}
    st.session_state.tz_markdown = ""
    st.session_state.selected_dept = None
    st.session_state.requester = ""

# ===== Prompts =====
SYSTEM_PROMPT = (
    "Вы — опытный продакт-менеджер и промпт-инженер. Работаете по-русски, кратко и структурно. "
    "Если пользователь дал только идею — сначала задайте релевантные уточняющие вопросы (5–10, не банальные). "
    "Когда ответы получены — соберите подробное, но лаконичное ТЗ в практическом формате для промпт-инженера. "
    "ТЗ должно быть ориентировано на реализацию промптов в продуктах на базе LLM."
)
QUESTIONS_INSTRUCTION = (
    "Сформируй список из 5–10 уточняющих вопросов по введённой идее/черновику. "
    "Формат строго нумерованный список, каждый вопрос — в одну строку, без подсписков."
)
TZ_INSTRUCTION = (
    "На основе идеи/черновика и ответов на вопросы собери ТЗ (Markdown, без лишней воды) по шаблону:\n\n"
    "# Название\nКороткое и ёмкое.\n\n"
    "## Цель\n1–3 предложения про проблему и целевую метрику.\n\n"
    "## Контекст и ограничения\nКаналы, пользователи, языки, приватность/безопасность, юридические ограничения.\n\n"
    "## Пользовательские сценарии\nБуллет-список типичных сценариев (3–6).\n\n"
    "## Входные данные\nЧто получает модель (поля формы, файлы, контекст, системные инструкции).\n\n"
    "## Выход/результат\nФормат ответа модели, стиль, структура, требования к длине.\n\n"
    "## Критерии качества и приёмки\nЧёткие проверяемые критерии (bullet list).\n\n"
    "## Ограничения генерации\nЗапрещённые темы/стили, тональность, правила безопасности.\n\n"
    "## Технические детали промпта\nСистемное сообщение, переменные, few-shot (если нужны), temperature/top_p, длина контекста.\n\n"
    "## Телеметрия и логирование\nЧто логируем, как измеряем качество.\n\n"
    "## Риски и допущения\nОсновные риски и способы их снижения.\n\n"
    "## Чек-лист готовности\nКороткий список из 5–8 пунктов.\n\n"
    "Пиши по-русски, делай разделы информативными и прикладными."
)

# ===== Parsers =====
def parse_numbered_questions(text_block: str) -> List[str]:
    """Извлекаем '1. вопрос' и '- вопрос'."""
    lines = text_block.splitlines()
    questions: List[str] = []
    for s in (ln.strip() for ln in lines):
        if not s:
            continue
        m = re.match(r"(\d+)[\).]\s*(.+)", s)
        if m:
            q = m.group(2).strip().rstrip("?。．！!；;：:")
            if q:
                questions.append(q + "?")
            continue
        m2 = re.match(r"^[\-•]\s*(.+)", s)
        if m2:
            q = m2.group(1).strip().rstrip("?。．！!；;：:")
            if q:
                questions.append(q + "?")
    if not questions:  # превратим строки в вопросы
        for s in (ln.strip() for ln in lines if ln.strip()):
            questions.append(s.rstrip("?") + "?")
    return questions[:10]

def parse_json_questions(text_block: str) -> List[str]:
    try:
        data = json.loads(text_block)
        if isinstance(data, dict) and isinstance(data.get("questions"), list):
            return [str(x).strip().rstrip("?") + "?" for x in data["questions"] if str(x).strip()][:10]
    except Exception:
        pass
    return []

# ===== Question generation =====
def generate_questions(initial_text: str) -> List[str]:
    """Стойкая генерация: список → JSON → фолбэк."""
    msg1 = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"""Текст:

{initial_text}

{QUESTIONS_INSTRUCTION}"""},
    ]
    raw1 = call_chat_completion(msg1, temperature=TEMPERATURE)
    qs = parse_numbered_questions(raw1) if raw1 else []
    if qs:
        return qs

    # Вторая попытка — строго JSON без пояснений (без f-строк, чтобы не экранировать скобки)
    json_prompt = (
        'Сформируй 7 уточняющих вопросов строго в JSON без пояснений, '
        'формат: {"questions":["вопрос1","вопрос2","..."]}. Текст:\n\n' + str(initial_text)
    )
    msg2 = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json_prompt},
    ]
    raw2 = call_chat_completion(msg2, temperature=TEMPERATURE)
    qs = parse_json_questions(raw2) if raw2 else []
    if qs:
        return qs

    # Фолбэк-вопросы
    return [
        "Какова основная цель и целевая метрика?",
        "Кто целевая аудитория/персоны и их ключевые задачи?",
        "Какие входные данные/поля должен предоставить пользователь?",
        "Какие ограничения по стилю, тону, длине и языку ответа?",
        "Какие риски/нежелательные ответы нужно исключить?",
        "Есть ли примеры желаемого результата (few-shot)?",
        "Где и как это будет встроено (канал/интеграция)?",
        "Какие требования к логированию и метрикам качества?",
        "Какие юридические/безопасностные требования?",
    ]

# ===== Misc builders =====
def build_header_meta(dept: Optional[str], requester: Optional[str]) -> str:
    meta = []
    if dept:
        meta.append(f"Отдел: {dept}")
    if requester:
        meta.append(f"Постановщик: {requester}")
    return ("\n" + "\n".join(meta) + "\n\n") if meta else "\n"

def build_fallback_tz(initial_text: str, questions: List[str], answers: Dict[int, str]) -> str:
    md: List[str] = []
    md += ["# Название\n", "Черновик ТЗ (автосборка)\n\n"]
    md += ["## Исходный ввод\n", str(initial_text).strip() + "\n\n"]
    md += ["## Уточнения\n"]
    for i in range(len(questions)):
        q = str(questions[i]); ans = str(answers.get(i, "") or "—")
        md.append(f"- {i+1}. {q}\n  Ответ: {ans}\n")
    md += [
        "\n## Цель\n—\n\n",
        "## Контекст и ограничения\n—\n\n",
        "## Пользовательские сценарии\n- —\n- —\n- —\n\n",
        "## Входные данные\n- —\n\n",
        "## Выход/результат\n- —\n\n",
        "## Критерии качества и приёмки\n- —\n\n",
        "## Ограничения генерации\n- —\n\n",
        "## Технические детали промпта\n- Системное сообщение — заполнить\n- Параметры: temperature/top_p — уточнить\n\n",
        "## Телеметрия и логирование\n- —\n\n",
        "## Риски и допущения\n- —\n\n",
        "## Чек-лист готовности\n- —\n",
    ]
    return "".join(md)

# ===================== STAGES =====================
# ----- Stage: Input -----
if st.session_state.stage == "input":
    st.subheader("Шаг 1. Введите идею или черновик ТЗ")
    input_mode = st.radio("Формат ввода:", ["Идея (свободный текст)", "Черновик ТЗ"], index=0, horizontal=True)
    placeholder = (
        "Опишите проблему/цель, целевую аудиторию, желаемый результат, ограничения…"
        if input_mode.startswith("Идея")
        else "Вставьте черновик ТЗ — уточним детали и усилим структуру"
    )
    st.session_state.initial_text = st.text_area(placeholder, value=st.session_state.initial_text, height=220)

    col_a, col_b = st.columns([1, 1])
    with col_a:
        if st.button("Сгенерировать вопросы", type="primary", use_container_width=True, disabled=not bool(st.session_state.initial_text.strip())):
            with st.spinner("Генерируем уточняющие вопросы…"):
                qs = generate_questions(st.session_state.initial_text)
                if not qs:
                    st.error("Не удалось получить вопросы от модели. Попробуйте ещё раз.")
                else:
                    st.session_state.questions = qs
                    st.session_state.answers = {i: "" for i in range(len(qs))}
                    st.session_state.stage = "questions"
                    st.rerun()
    with col_b:
        if st.button("Очистить", use_container_width=True):
            st.session_state.initial_text = ""

# ----- Stage: Questions -----
elif st.session_state.stage == "questions":
    st.subheader("Шаг 2. Ответьте на вопросы")
    if not st.session_state.questions:
        st.warning("Сначала вернитесь и введите идею/черновик. Или попробуйте заново сгенерировать вопросы.")
        if st.button("Сгенерировать вопросы ещё раз"):
            with st.spinner("Генерируем уточняющие вопросы…"):
                qs = generate_questions(st.session_state.initial_text)
                st.session_state.questions = qs
                st.session_state.answers = {i: "" for i in range(len(qs))}
                st.rerun()
    else:
        for i, q in enumerate(st.session_state.questions, start=1):
            st.session_state.answers[i - 1] = st.text_area(f"{i}. {q}", value=st.session_state.answers.get(i - 1, ""), height=100)

        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("Сформировать ТЗ", type="primary", use_container_width=True):
                with st.spinner("Собираем структурное ТЗ…"):
                    answers_block = "\n\n".join(
                        [f"{i+1}. {st.session_state.questions[i]}\nОтвет: {st.session_state.answers.get(i, '').strip()}" for i in range(len(st.session_state.questions))]
                    )
                    msg = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"""Изначальный текст (идея/черновик):

{st.session_state.initial_text}

Ответы на уточняющие вопросы:

{answers_block}

{TZ_INSTRUCTION}"""},
                    ]
                    tz_md = call_chat_completion(msg, temperature=TEMPERATURE)
                    if not tz_md.strip():
                        tz_md = build_fallback_tz(st.session_state.initial_text, st.session_state.questions, st.session_state.answers)
                        st.warning("Модель вернула пустой ответ — собрали базовый черновик ТЗ автоматически.")
                    st.session_state.tz_markdown = tz_md
                    st.session_state.stage = "draft"
                    st.rerun()
        with col2:
            if st.button("Перегенерировать вопросы", use_container_width=True):
                with st.spinner("Генерируем уточняющие вопросы…"):
                    qs = generate_questions(st.session_state.initial_text)
                    st.session_state.questions = qs
                    st.session_state.answers = {i: "" for i in range(len(qs))}
                    st.rerun()
        with col3:
            if st.button("Назад", use_container_width=True):
                st.session_state.stage = "input"
                st.rerun()

# ----- Stage: Draft / Preview -----
elif st.session_state.stage == "draft":
    st.subheader("Шаг 3. Предпросмотр ТЗ")
    st.info("Вы можете подредактировать ТЗ перед отправкой.")
    st.session_state.tz_markdown = st.text_area("ТЗ (Markdown)", value=st.session_state.tz_markdown, height=420)

    dept_names = list(DEPT_MAP.keys())
    if dept_names:
        st.session_state.selected_dept = st.selectbox("Куда отправить (отдел)", options=dept_names, index=0)
    else:
        st.session_state.selected_dept = None

    st.session_state.requester = st.text_input("Постановщик (ФИО, ник, контакт)", value=st.session_state.requester)

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        st.download_button("Скачать .md", data=st.session_state.tz_markdown.encode("utf-8"), file_name="tz_prompt.md", mime="text/markdown")
    with c2:
        if st.button("Отправить в Telegram", type="primary", use_container_width=True):
            dept = st.session_state.selected_dept
            chat_id = DEPT_MAP.get(dept) if dept else None
            header = "ТЗ для промпт-инженера\n" + ("=" * 24) + build_header_meta(dept, st.session_state.requester)
            final_text = header + st.session_state.tz_markdown
            with st.spinner("Отправляем в Telegram…"):
                responses = send_to_telegram(final_text, chat_id=chat_id)
                if responses and all(r.status_code == 200 for r in responses):
                    st.toast("Отправлено в Telegram ✅", icon="✅")
                    reset_to_home()
                    st.rerun()
                else:
                    st.warning("Часть сообщений могла не отправиться. Проверьте настройки и логи выше.")
    with c3:
        if st.button("Назад к вопросам", use_container_width=True):
            st.session_state.stage = "questions"
            st.rerun()
