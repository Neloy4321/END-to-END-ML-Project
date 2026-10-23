"""Optional, non-diagnostic survey insights built from released cohort totals.

All V2 predictor inputs and inference weights stay unchanged. Additional form
answers are never included in model scoring, and no medical diagnosis is inferred.
"""
import json
from pathlib import Path

from .behavioral_assessment import InvalidAssessmentInput

SINGLE = (
    ("age", "Age group", "Background"),
    ("gender", "Gender (optional)", "Background"),
    ("occupation", "Occupation", "Background"),
    ("weekend_bed", "Usual bedtime on weekends", "Weekend schedule"),
    ("weekend_wake", "Usual wake-up time on weekends", "Weekend schedule"),
    ("sleep_latency", "Reported time to fall asleep", "Weekend schedule"),
    ("medical_self_report", "Existing medical condition (self-reported, optional)", "Reported experiences"),
    ("concentration", "Difficulty concentrating due to lack of sleep", "Reported experiences"),
)
MULTI = (
    ("late_reasons", "Reasons for sleeping late", "Habits & effects"),
    ("side_effects", "Effects you have experienced", "Habits & effects"),
    ("coping", "How you cope with late sleeping", "Habits & effects"),
)
CHRONOLOGY = {
    "bed_time": ("Before 10 PM", "10 PM - 12 AM", "12 AM - 2 AM", "After 2 AM"),
    "wake_time": ("Before 6 AM", "6 AM - 8 AM", "8 AM - 10 AM", "After 10 AM"),
}
DISPLAY_ORDER = {
    "age": ("Under 18", "18-30", "31-50", "Above 50"),
    "weekend_bed": CHRONOLOGY["bed_time"],
    "weekend_wake": CHRONOLOGY["wake_time"],
    "concentration": ("Never", "Rarely", "Sometimes", "Often", "Always"),
    "sleep_latency": (
        "Less than 15 minutes", "15-30 minutes",
        "31-60 minutes", "More than 60 minutes",
    ),
}
COHORT_TYPES = (
    ("age", "by_age", "Age group"),
    ("gender", "by_gender", "Gender"),
    ("occupation", "by_occupation", "Occupation"),
    ("medical_self_report", "by_medical_self_report", "Reported medical condition"),
)
PREVALENCE_TYPES = (
    ("late_reasons", "reason_prevalence", "Late-sleeping reasons"),
    ("side_effects", "effect_prevalence", "Self-reported effects"),
    ("coping", "coping_prevalence", "Coping choices"),
)


class SurveyInsightsV3:
    def __init__(self, file_path):
        data = json.loads(Path(file_path).read_text(encoding="utf-8"))
        if data.get("schema_version") != 3 or data.get("task") != "descriptive_non_diagnostic_survey_cohorts":
            raise ValueError("Invalid survey cohort artifact")
        if data.get("source_record_count") != data["overall"]["n"]:
            raise ValueError("Cohort count mismatch")
        if data.get("minimum_cohort_size", 0) < 100:
            raise ValueError("Small cohort disclosure risk")
        for _, group, _ in COHORT_TYPES:
            if any(item["n"] < data["minimum_cohort_size"] for item in data[group].values()):
                raise ValueError("Subthreshold medical/demographic cohort")
        self.data = data

    def sections(self, selected=None):
        selected = selected or {}
        sections = []
        for title in ("Background", "Weekend schedule", "Reported experiences", "Habits & effects"):
            single = []
            multi = []
            for key, label, group in SINGLE:
                if group != title:
                    continue
                choices = self.data["single_select_options"][key]
                order = DISPLAY_ORDER.get(key)
                if order:
                    choices = [answer for answer in order if answer in choices]
                single.append({
                    "key": key, "label": label, "selected": selected.get(key, ""),
                    "options": choices,
                })
            for key, label, group in MULTI:
                if group != title:
                    continue
                if hasattr(selected, "getlist"):
                    picked = selected.getlist(key)
                else:
                    picked = selected.get(key, [])
                if not isinstance(picked, list):
                    picked = []
                multi.append({
                    "key": key, "label": label,
                    "options": [
                        {"value": value, "checked": value in picked}
                        for value in self.data["multi_select_options"][key]
                    ],
                })
            sections.append({"title": title, "single": single, "multi": multi})
        return sections

    def parse(self, form):
        result = {}
        for key, label, _ in SINGLE:
            original = form.get(key, "")
            value = original.strip() if isinstance(original, str) else ""
            if value and value not in self.data["single_select_options"][key]:
                raise InvalidAssessmentInput("Invalid answer for " + key)
            result[key] = value
        for key, label, _ in MULTI:
            values = form.getlist(key) if hasattr(form, "getlist") else form.get(key, [])
            if not isinstance(values, list):
                raise InvalidAssessmentInput("Invalid answer format for " + key)
            allowed = self.data["multi_select_options"][key]
            if len(values) > len(allowed) or len(values) != len(set(values)):
                raise InvalidAssessmentInput("Duplicate or excess responses for " + key)
            if not all(isinstance(value, str) and value in allowed for value in values):
                raise InvalidAssessmentInput("Invalid checkbox option for " + key)
            result[key] = values
        return result

    @staticmethod
    def _band_shift(weekday, weekend, order):
        if not weekday or not weekend:
            return None
        old_index = order.index(weekday)
        new_index = order.index(weekend)
        difference = new_index - old_index
        if difference == 0:
            description = "Same answer band"
        else:
            direction = "later" if difference > 0 else "earlier"
            description = str(abs(difference)) + " time band(s) " + direction
        return description

    def report(self, six_answers, extra):
        bed_shift = self._band_shift(
            six_answers["bed_time"], extra["weekend_bed"], CHRONOLOGY["bed_time"]
        )
        wake_shift = self._band_shift(
            six_answers["wake_time"], extra["weekend_wake"], CHRONOLOGY["wake_time"]
        )
        summaries = [
            {"label": label, "answer": extra[key]}
            for key, label, _ in SINGLE
            if extra.get(key)
        ]
        prevalence = []
        for key, catalog_key, section_label in PREVALENCE_TYPES:
            for answer in extra[key]:
                item = self.data[catalog_key][answer]
                prevalence.append({
                    "group": section_label,
                    "answer": answer,
                    "survey_percent": item["percent_of_answers"],
                    "count": item["n"],
                })
        cohorts = []
        for key, group, name in COHORT_TYPES:
            value = extra[key]
            if value and value in self.data[group]:
                item = self.data[group][value]
                cohorts.append({
                    "title": name, "answer": value,
                    "n": item["n"],
                    "distribution": [
                        {"label": cls, "percent": item["percent"][cls]}
                        for cls in self.data["class_labels"]
                    ],
                })
        return {
            "selected": extra,
            "bed_shift": bed_shift,
            "wake_shift": wake_shift,
            "summaries": summaries,
            "selected_behaviours": prevalence,
            "cohorts": cohorts,
            "overall_distribution": [
                {"label": cls, "percent": self.data["overall"]["percent"][cls]}
                for cls in self.data["class_labels"]
            ],
            "supplied_optional_answers": any(bool(value) for value in extra.values()),
        }
