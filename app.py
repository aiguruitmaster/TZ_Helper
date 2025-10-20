# -*- coding: utf-8 -*-


from __future__ import annotations
import os
import re
import json
import textwrap
from typing import List, Dict, Any

import streamlit as st
import requests

# === OpenAI SDK (официальная библиотека v1+) ===
try:
    from openai import OpenAI
except Exception as e:  # pragma: no cover
    OpenAI = None

# ---------------------------- UI CONFIG ----------------------------
st.set_page_config(page_title="Генератор ТЗ → Telegram", page_icon="📝", layout="centered")

st.title("📝 Генератор ТЗ для промпт‑инженера")
st.caption("Вставьте идею или черновик ТЗ, ответьте на уточняющие вопросы, утвердите и отправьте в нужный отдел в Telegram.")

# ---------------------------- Secrets / Settings ----------------------------
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
OPENAI_MODEL_DEFAULT = st.secrets.get("OPENAI_MODEL", "gpt-4o-mini")  # можно переопределить в сайдбаре

TELEGRAM_CONF = st.secrets.get("telegram", {})
TG_TOKEN = TELEGRAM_CONF.get("bot_token") or st.secrets.get("TELEGRAM_BOT_TOKEN")
TG_DEFAULT_CHAT = TELEGRAM_CONF.get("default_chat_id") or st.secrets.get("TELEGRAM_CHAT_ID")
DEPT_MAP: Dict[str, str] = TELEGRAM_CONF.get("departments", {})

if "_init" not in st.session_state:
    st.session_state._init = True
    st.session_state.stage = "input"  # input -> questions -> draft -> send
    st.session_state.initial_text = ""
    st.session_state.questions: List[str] = []
    st.session_state.answers: Dict[int, str] = {}
    st.session_state.tz_markdown = ""
    st.session_state.confirmed = False
    st.session_state.selected_dept = None
    st.session_state.requester = ""

# ---------------------------- Config (без сайдбара) ----------------------------
model_name = OPENAI_MODEL_DEFAULT
TEMPERATURE = 0.2

# ---------------------------- Guards ----------------------------
if not OPENAI_API_KEY:
    st.error("Не найден OPENAI_API_KEY в secrets. Добавьте ключ в .streamlit/secrets.toml и перезапустите приложение.")

if OpenAI is None:
    st.error("Пакет openai не установлен. Установите `pip install openai` и перезапустите.")

# ---------------------------- Helpers ----------------------------

def get_openai_client() -> OpenAI:
    """Инициализация клиента OpenAI."""
    return OpenAI(api_key=OPENAI_API_KEY)


def call_chat_completion(messages: List[Dict[str, Any]], max_tokens: int = 2000, temperature: float = 0.2) -> str:
    """Вызов chat.completions с безопасным извлечением контента."""
    client = get_openai_client()
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = resp.choices[0].message.content or ""
        return content.strip()
    except Exception as e:
        st.error(f"Ошибка OpenAI: {e}")
        return ""


def chunk_for_tg(text: str, limit: int = 4000) -> List[str]:
    """Разбить длинный текст для Telegram (лимит ~4096 симв.)."""
    text = text.strip()
    if len(text) <= limit:
        return [text]
    parts = []
    current = []
    size = 0
    for para in text.split("\n\n"):
        para_block = (para.strip() + "\n\n")
        if size + len(para_block) > limit and current:
            parts.append("".join(current).rstrip())
            current = [para_block]
            size = len(para_block)
        else:
            current.append(para_block)
            size += len(para_block)
    if current:
        parts.append("".join(current).rstrip())
    # если всё равно какие-то куски большие — делим по строкам
    fixed = []
    for p in parts:
        if len(p) <= limit:
            fixed.append(p)
        else:
            buf = []
            tally = 0
            for line in p.splitlines(keepends=True):
                if tally + len(line) > limit and buf:
                    fixed.append("".join(buf).rstrip())
                    buf = [line]
                    tally = len(line)
                else:
                    buf.append(line)
                    tally += len(line)
            if buf:
                fixed.append("".join(buf).rstrip())
    return fixed


def send_to_telegram(text: str, chat_id: str | None = None) -> List[requests.Response]:
    if not TG_TOKEN:
        st.error("Не найден telegram.bot_token в secrets.")
        return []
    target_chat = chat_id or TG_DEFAULT_CHAT
    if not target_chat:
        st.error("Не задан chat_id для Telegram (telegram.default_chat_id или TELEGRAM_CHAT_ID).")
        return []
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    results = []
    for chunk in chunk_for_tg(text):
        payload = {
            "chat_id": target_chat,
            "text": chunk,
            # Без parse_mode, чтобы избежать экранирования. Хотите форматирование — смените на 'HTML'.
            # "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
        r = requests.post(url, data=payload, timeout=30)
        results.append(r)
        if r.status_code != 200:
            st.error(f"Telegram ошибка {r.status_code}: {r.text}")
            break
    return results


# ---------------------------- Prompt builders ----------------------------
SYSTEM_PROMPT = (
    "Вы — опытный продакт-менеджер и промпт-инженер. Работаете по-русски, кратко и структурно. "
    "Если пользователь дал только идею — сначала задайте релевантные уточняющие вопросы (5–10, не банальные). "
    "Когда ответы получены — соберите подробное, но лаконичное ТЗ в практическом формате для промпт‑инженера. "
    "ТЗ должно быть ориентировано на реализацию промптов в продуктах на базе LLM."
)

QUESTIONS_INSTRUCTION = (
    "Сформируй список из 5–10 уточняющих вопросов по введённой идее/черновику. "
    "Формат строго нумерованный список, каждый вопрос — в одну строку, без подсписков."
)

TZ_INSTRUCTION = (
    "На основе идеи/черновика и ответов на вопросы собери ТЗ (Markdown, без лишней воды) по шаблону:\n\n"
    "# Название\n"
    "Короткое и ёмкое.\n\n"
    "## Цель\n"
    "1–3 предложения про проблему и целевую метрику.\n\n"
    "## Контекст и ограничения\n"
    "Каналы, пользователи, языки, приватность/безопасность, юридические ограничения.\n\n"
    "## Пользовательские сценарии\n"
    "Буллет-список типичных сценариев (3–6).\n\n"
    "## Входные данные\n"
    "Что получает модель (поля формы, файлы, контекст, системные инструкции).\n\n"
    "## Выход/результат\n"
    "Формат ответа модели, стиль, структура, требования к длине.\n\n"
    "## Критерии качества и приёмки\n"
    "Чёткие проверяемые критерии (bullet list).\n\n"
    "## Ограничения генерации\n"
    "Запрещённые темы/стили, тональность, правила безопасности.\n\n"
    "## Технические детали промпта\n"
    "Системное сообщение, переменные, few-shot (если нужны), temperature/top_p, длина контекста.\n\n"
    "## Телеметрия и логирование\n"
    "Что логируем, как измеряем качество.\n\n"
    "## Риски и допущения\n"
    "Основные риски и способы их снижения.\n\n"
    "## Чек-лист готовности\n"
    "Короткий список из 5–8 пунктов.\n\n"
    "Пиши по-русски, делай разделы информативными и прикладными."
)


def parse_numbered_questions(text_block: str) -> List[str]:
    """Вытащить строки вида '1. вопрос' из текста."""
    lines = text_block.splitlines()
    questions = []
    for ln in lines:
        m = re.match(r"\s*(\d+)[\).]\s*(.+)", ln.strip())
        if m:
            q = m.group(2).strip()
            if q:
                questions.append(q)
    # fallback — если модель дала просто абзацы
    if not questions:
        for ln in lines:
            ln = ln.strip().rstrip("?。．！!；;：:")
            if ln:
                questions.append(ln + "?")
    # ограничим 10
    return questions[:10]


def build_header_meta(dept: str | None, requester: str | None) -> str:
    meta = []
    if dept:
        meta.append(f"Отдел: {dept}")
    if requester:
        meta.append(f"Постановщик: {requester}")
    if meta:
        return "\n" + "\n".join(meta) + "\n\n"
    return "\n"

# ---------------------------- Stage: Input ----------------------------
if st.session_state.stage == "input":
    st.subheader("Шаг 1. Введите идею или черновик ТЗ")
    input_mode = st.radio(
        "Формат ввода:",
        ["Идея (свободный текст)", "Черновик ТЗ"],
        index=0,
        horizontal=True,
    )
    placeholder = (
        "Опишите проблему/цель, целевую аудиторию, желаемый результат, ограничения…"
        if input_mode.startswith("Идея")
        else "Вставьте любой ваш черновик ТЗ — мы уточним детали и усилим структуру"
    )
    st.session_state.initial_text = st.text_area(
        placeholder,
        value=st.session_state.initial_text,
        height=220,
    )

    col_a, col_b = st.columns([1, 1])
    with col_a:
        if st.button("Сгенерировать вопросы", type="primary", use_container_width=True, disabled=not bool(st.session_state.initial_text.strip())):
            with st.spinner("Генерируем уточняющие вопросы…"):
                msg = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Текст:\n\n{st.session_state.initial_text}\n\n{QUESTIONS_INSTRUCTION}"},
                ]
                raw = call_chat_completion(msg, temperature=TEMPERATURE)
                st.session_state.questions = parse_numbered_questions(raw)
                st.session_state.answers = {i: "" for i in range(len(st.session_state.questions))}
                st.session_state.stage = "questions"
                st.rerun()
    with col_b:
        st.button("Очистить", use_container_width=True, on_click=lambda: st.session_state.update(initial_text=""))

# ---------------------------- Stage: Questions ----------------------------
elif st.session_state.stage == "questions":
    st.subheader("Шаг 2. Ответьте на вопросы")
    if not st.session_state.questions:
        st.warning("Сначала вернитесь и введите идею/черновик.")
    else:
        for i, q in enumerate(st.session_state.questions, start=1):
            st.session_state.answers[i - 1] = st.text_area(f"{i}. {q}", value=st.session_state.answers.get(i - 1, ""), height=100)

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Сформировать ТЗ", type="primary", use_container_width=True):
                with st.spinner("Собираем структурное ТЗ…"):
                    # Собираем блок с ответами
                    answers_block = "\n\n".join([f"{i+1}. {st.session_state.questions[i]}\nОтвет: {st.session_state.answers.get(i, '').strip()}" for i in range(len(st.session_state.questions))])
                    msg = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": (
                            f"Изначальный текст (идея/черновик):\n\n{st.session_state.initial_text}\n\n"
                            f"Ответы на уточняющие вопросы:\n\n{answers_block}\n\n{TZ_INSTRUCTION}"
                        )},
                    ]
                    tz_md = call_chat_completion(msg, temperature=TEMPERATURE)
                    st.session_state.tz_markdown = tz_md
                    st.session_state.stage = "draft"
                    st.rerun()
        with col2:
            if st.button("Назад", use_container_width=True):
                st.session_state.stage = "input"
                st.rerun()

# ---------------------------- Stage: Draft / Preview ----------------------------
elif st.session_state.stage == "draft":
    st.subheader("Шаг 3. Предпросмотр ТЗ")

    st.info("Вы можете подредактировать ТЗ перед отправкой.")
    st.session_state.tz_markdown = st.text_area("ТЗ (Markdown)", value=st.session_state.tz_markdown, height=420)

    # Метаданные для Telegram
    dept_names = list(DEPT_MAP.keys())
    if dept_names:
        st.session_state.selected_dept = st.selectbox("Куда отправить (отдел)", options=dept_names, index=0)
    else:
        st.session_state.selected_dept = None
        st.caption("Маршрутизация по отделам не настроена в secrets — будет использован default_chat_id.")

    st.session_state.requester = st.text_input("Постановщик (ФИО, ник, контакт)", value=st.session_state.requester)

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        st.download_button("Скачать .md", data=st.session_state.tz_markdown.encode("utf-8"), file_name="tz_prompt.md", mime="text/markdown")
    with c2:
        if st.button("Отправить в Telegram", type="primary", use_container_width=True):
            # Собираем финальный текст
            dept = st.session_state.selected_dept
            chat_id = None
            if dept and dept in DEPT_MAP:
                chat_id = DEPT_MAP[dept]
            header = "ТЗ для промпт‑инженера\n" + ("=" * 24) + build_header_meta(dept, st.session_state.requester)
            final_text = header + st.session_state.tz_markdown
            with st.spinner("Отправляем в Telegram…"):
                responses = send_to_telegram(final_text, chat_id=chat_id)
                if responses and all(r.status_code == 200 for r in responses):
                    st.success("Отправлено в Telegram ✅")
                else:
                    st.warning("Часть сообщений могла не отправиться. Проверьте настройки и логи выше.")
    with c3:
        if st.button("Назад к вопросам", use_container_width=True):
            st.session_state.stage = "questions"
            st.rerun()


