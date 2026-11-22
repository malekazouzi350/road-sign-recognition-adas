"""Small UI helpers so every page looks the same."""

from __future__ import annotations

from html import escape
from pathlib import Path

import streamlit as st

CSS_FILE = Path(__file__).resolve().parents[1] / "assets" / "style.css"


def inject_css() -> None:
    st.html(f"<style>{CSS_FILE.read_text(encoding='utf-8')}</style>")


def page_header(number: str, label: str, title: str, subtitle: str | None = None) -> None:
    sub = f'<p class="rs-subtitle">{escape(subtitle)}</p>' if subtitle else ""
    st.html(
        f'<div class="rs-kicker"><span class="rs-tag">{escape(number)}</span>'
        f'<span class="rs-label">{escape(label)}</span></div>'
        f'<h1 class="rs-title">{escape(title)}</h1>{sub}'
    )


def kpi(value: str, label: str, detail: str = "", tone: str = "teal") -> str:
    """HTML for one key-number card; tone in teal | green | coral | grey."""
    return (
        f'<div class="rs-kpi rs-{tone}"><div class="rs-kpi-value">{escape(value)}</div>'
        f'<div class="rs-kpi-label">{escape(label)}</div>'
        f'<div class="rs-kpi-detail">{escape(detail)}</div></div>'
    )


def kpi_row(cards: list[str]) -> None:
    st.html(f'<div class="rs-kpi-row">{"".join(cards)}</div>')


def status_line(ok: bool, text: str) -> str:
    cls = "rs-ok" if ok else "rs-missing"
    mark = "●" if ok else "○"
    return f'<div class="rs-status {cls}"><span>{mark}</span> {escape(text)}</div>'


def not_evaluated(what: str, command: str) -> None:
    st.html(
        f'<div class="rs-empty"><b>Not evaluated yet</b> · {escape(what)}<br>'
        f'<span>Pour produire ce résultat : <code>{escape(command)}</code></span></div>'
    )
