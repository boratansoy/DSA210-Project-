#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
import unicodedata
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    from icalendar import Calendar
except ImportError as exc:
    raise ImportError(
        "The 'icalendar' package is required. Please run: python3 -m pip install icalendar pandas"
    ) from exc


BASE_DIR = Path.home() / "Desktop" / "DSA210 TERM PROJECT"
INPUT_ICS_PATH = BASE_DIR / "Takvim .ics"
FILTERED_EVENTS_OUTPUT_PATH = BASE_DIR / "filtered_calendar_events.csv"
SPECIAL_DATES_OUTPUT_PATH = BASE_DIR / "special_dates.csv"

EXAM_KEYWORDS = [
    "midterm", "final", "quiz", "sinav", "sınav", "vize", "but", "bütünleme", "butunleme"
]

DEADLINE_KEYWORDS = [
    "homework", "assignment", "odev", "ödev", "proposal", "project", "proje",
    "objection", "itiraz", "review", "deadline", "due", "submission", "submit",
    "teslim", "son gun", "son gün", "basvuru", "başvuru", "application",
    "mulakat", "mülakat", "interview", "form", "mail", "email"
]

ACADEMIC_KEYWORDS = [
    "ders", "lecture", "class", "lab", "recitation", "office hour", "seminar",
    "workshop", "presentation", "sunum", "meeting", "toplanti", "toplantı", "course"
]

ACADEMIC_COURSE_PATTERNS = [
    r"\bdsa\b", r"\bcs\s*\d*\b", r"\bie\s*\d*\b", r"\bmath\s*\d*\b", r"\bhum\s*\d*\b",
    r"\bens\s*\d*\b", r"\becon\s*\d*\b", r"\bhist\s*\d*\b", r"\btll\s*\d*\b", r"\bsps\s*\d*\b"
]

PERSONAL_KEYWORDS = [
    "dogum gunu", "doğum günü", "birthday", "anneler gunu", "anneler günü",
    "beril", "bulusma", "buluşma", "meetup", "date", "yemek"
]

NOISE_KEYWORDS = [
    "kargo", "trendyol", "kavanoz", "tras", "traş", "temizlik", "havlu",
    "siparis", "sipariş", "buzluk", "buzdolabi", "buzdolabı", "terzi", "hisse al"
]


@dataclass(frozen=True)
class CalendarEventRow:
    summary: str
    normalized_summary: str
    start: pd.Timestamp
    end: pd.Timestamp | None
    date: object
    category: str
    event_group: str


def normalize_turkish_text(value: object) -> str:
    if value is None:
        return ""

    text = str(value).lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))

    replacements = {
        "ç": "c",
        "ğ": "g",
        "ı": "i",
        "ö": "o",
        "ş": "s",
        "ü": "u",
        "â": "a",
        "î": "i",
        "û": "u",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"\s+", " ", text).strip()
    return text


def contains_any_keyword(normalized_text: str, keywords: Iterable[str]) -> bool:
    return any(normalize_turkish_text(keyword) in normalized_text for keyword in keywords)


def contains_any_pattern(normalized_text: str, patterns: Iterable[str]) -> bool:
    return any(re.search(pattern, normalized_text) is not None for pattern in patterns)


def classify_event(summary: str) -> str | None:
    normalized_summary = normalize_turkish_text(summary)

    if not normalized_summary:
        return None

    if contains_any_keyword(normalized_summary, NOISE_KEYWORDS):
        return None

    if contains_any_keyword(normalized_summary, EXAM_KEYWORDS):
        return "exam"

    if contains_any_keyword(normalized_summary, DEADLINE_KEYWORDS):
        return "deadline"

    if contains_any_keyword(normalized_summary, ACADEMIC_KEYWORDS) or contains_any_pattern(
        normalized_summary, ACADEMIC_COURSE_PATTERNS
    ):
        return "academic_event"

    if contains_any_keyword(normalized_summary, PERSONAL_KEYWORDS):
        return "personal"

    return None


def infer_event_group(category: str, normalized_summary: str) -> str:
    if category == "exam":
        if "final" in normalized_summary:
            return "final"
        if "midterm" in normalized_summary or "vize" in normalized_summary:
            return "midterm"
        if "quiz" in normalized_summary:
            return "quiz"
        return "exam"

    if category == "deadline":
        if any(keyword in normalized_summary for keyword in ["proposal", "project", "proje"]):
            return "project_deadline"
        if any(keyword in normalized_summary for keyword in ["homework", "odev", "assignment", "ödev"]):
            return "homework_deadline"
        if any(keyword in normalized_summary for keyword in ["basvuru", "başvuru", "application", "form"]):
            return "application_deadline"
        if any(keyword in normalized_summary for keyword in ["mulakat", "mülakat", "interview"]):
            return "interview_related"
        return "deadline"

    if category == "academic_event":
        return "academic_event"

    if category == "personal":
        return "personal"

    return "other"


def safe_to_datetime(value: object) -> pd.Timestamp | None:
    if value is None:
        return None

    try:
        result = pd.to_datetime(value, errors="coerce")
        if pd.isna(result):
            return None
        return result
    except Exception:
        return None


def read_calendar(input_path: Path) -> Calendar:
    if not input_path.exists():
        raise FileNotFoundError(f"Calendar file not found: {input_path}")

    raw_bytes = input_path.read_bytes()
    if not raw_bytes.strip():
        raise ValueError(f"Calendar file is empty: {input_path}")

    return Calendar.from_ical(raw_bytes)


def extract_component_value(component, key: str) -> object:
    value = component.get(key)
    if value is None:
        return None
    if hasattr(value, "dt"):
        return value.dt
    return value


def extract_filtered_events(calendar: Calendar) -> tuple[pd.DataFrame, int]:
    components = [component for component in calendar.walk() if component.name == "VEVENT"]
    filtered_rows: list[CalendarEventRow] = []

    for component in components:
        summary = str(extract_component_value(component, "SUMMARY") or "").strip()
        normalized_summary = normalize_turkish_text(summary)
        category = classify_event(summary)

        if category is None:
            continue

        start = safe_to_datetime(extract_component_value(component, "DTSTART"))
        end = safe_to_datetime(extract_component_value(component, "DTEND"))

        if start is None:
            continue

        filtered_rows.append(
            CalendarEventRow(
                summary=summary,
                normalized_summary=normalized_summary,
                start=start,
                end=end,
                date=start.date(),
                category=category,
                event_group=infer_event_group(category, normalized_summary),
            )
        )

    dataframe = pd.DataFrame([asdict(row) for row in filtered_rows])

    if dataframe.empty:
        dataframe = pd.DataFrame(
            columns=["date", "summary", "category", "event_group", "start", "end", "normalized_summary"]
        )
    else:
        dataframe = dataframe[
            ["date", "summary", "category", "event_group", "start", "end", "normalized_summary"]
        ].sort_values(["date", "start", "summary"]).reset_index(drop=True)

    return dataframe, len(components)


def create_event_day_rows(filtered_events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, event in filtered_events.iterrows():
        category = event["category"]
        rows.append(
            {
                "date": event["date"],
                "is_exam": int(category == "exam"),
                "is_deadline": int(category == "deadline"),
                "is_academic_event": int(category == "academic_event"),
                "is_personal": int(category == "personal"),
                "is_stress_period": 0,
                "source_events": event["summary"],
                "categories": category,
            }
        )
    return pd.DataFrame(rows)


def create_stress_period_rows(filtered_events: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, event in filtered_events.iterrows():
        category = event["category"]

        if category == "exam":
            stress_days = 7
        elif category == "deadline":
            stress_days = 3
        else:
            continue

        event_date = pd.to_datetime(event["date"]).date()

        for offset in range(1, stress_days + 1):
            stress_date = (pd.Timestamp(event_date) - pd.Timedelta(days=offset)).date()
            rows.append(
                {
                    "date": stress_date,
                    "is_exam": 0,
                    "is_deadline": 0,
                    "is_academic_event": 0,
                    "is_personal": 0,
                    "is_stress_period": 1,
                    "source_events": event["summary"],
                    "categories": f"stress_before_{category}",
                }
            )

    return pd.DataFrame(rows)


def concatenate_unique(values: pd.Series) -> str:
    unique_values = sorted({str(value) for value in values.dropna() if str(value).strip()})
    return "; ".join(unique_values)


def create_special_dates(filtered_events: pd.DataFrame) -> pd.DataFrame:
    output_columns = [
        "date",
        "is_exam",
        "is_deadline",
        "is_academic_event",
        "is_personal",
        "is_stress_period",
        "source_events",
        "categories",
    ]

    if filtered_events.empty:
        return pd.DataFrame(columns=output_columns)

    event_day_rows = create_event_day_rows(filtered_events)
    stress_period_rows = create_stress_period_rows(filtered_events)
    combined = pd.concat([event_day_rows, stress_period_rows], ignore_index=True)

    binary_columns = [
        "is_exam",
        "is_deadline",
        "is_academic_event",
        "is_personal",
        "is_stress_period",
    ]

    special_dates = (
        combined.groupby("date", as_index=False)
        .agg(
            {
                "is_exam": "max",
                "is_deadline": "max",
                "is_academic_event": "max",
                "is_personal": "max",
                "is_stress_period": "max",
                "source_events": concatenate_unique,
                "categories": concatenate_unique,
            }
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    special_dates[binary_columns] = special_dates[binary_columns].astype("int64")
    return special_dates.loc[:, output_columns]


def save_outputs(filtered_events: pd.DataFrame, special_dates: pd.DataFrame) -> None:
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    filtered_events.to_csv(FILTERED_EVENTS_OUTPUT_PATH, index=False, encoding="utf-8-sig")
    special_dates.to_csv(SPECIAL_DATES_OUTPUT_PATH, index=False, encoding="utf-8-sig")


def print_run_summary(total_events_found: int, filtered_events: pd.DataFrame, special_dates: pd.DataFrame) -> None:
    print(f"Total calendar events found: {total_events_found}")
    print(f"Number of filtered meaningful events: {len(filtered_events)}")
    print(f"Number of rows in special_dates.csv: {len(special_dates)}")
    print(f"Saved filtered events to: {FILTERED_EVENTS_OUTPUT_PATH}")
    print(f"Saved special dates to: {SPECIAL_DATES_OUTPUT_PATH}")

    print("\nFirst 10 filtered events:")
    if filtered_events.empty:
        print("No filtered events found.")
    else:
        print(filtered_events.head(10).to_string(index=False))

    print("\nFirst 10 rows of special_dates:")
    if special_dates.empty:
        print("No special dates generated.")
    else:
        print(special_dates.head(10).to_string(index=False))


def main() -> None:
    try:
        calendar = read_calendar(INPUT_ICS_PATH)
        filtered_events, total_events_found = extract_filtered_events(calendar)
        special_dates = create_special_dates(filtered_events)
        save_outputs(filtered_events, special_dates)
        print_run_summary(total_events_found, filtered_events, special_dates)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
