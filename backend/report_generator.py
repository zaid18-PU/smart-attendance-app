"""
Generates a natural-language attendance summary using a small instruction-tuned
transformer (FLAN-T5). Runs fully offline/local -- no API key required, which
keeps the whole capstone self-contained. If you have an Anthropic API key and
want richer report writing, swap `generate_summary` to call the Claude API
instead (see the commented alternative at the bottom).
"""
from transformers import pipeline
import torch

DEVICE = 0 if torch.cuda.is_available() else -1

_summarizer = None


def load_report_model():
    global _summarizer
    print("[report_generator] Loading FLAN-T5 summarizer ...")
    _summarizer = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        device=DEVICE,
    )
    print("[report_generator] Ready.")


def generate_summary(present_names, absent_names, total_roster):
    present_pct = (len(present_names) / total_roster * 100) if total_roster else 0

    prompt = (
        "Write a short, professional attendance summary paragraph (3-4 sentences) "
        f"for a class session. Total students on roster: {total_roster}. "
        f"Present ({len(present_names)}): {', '.join(present_names) if present_names else 'none'}. "
        f"Absent ({len(absent_names)}): {', '.join(absent_names) if absent_names else 'none'}. "
        f"Attendance rate: {present_pct:.1f}%. "
        "Mention the attendance rate and name the absent students."
    )

    if _summarizer is None:
        # graceful fallback if model isn't loaded yet
        return _fallback_summary(present_names, absent_names, total_roster, present_pct)

    try:
        result = _summarizer(prompt, max_new_tokens=120, do_sample=False)
        text = result[0]["generated_text"].strip()
        if len(text) < 10:  # model sometimes returns near-empty output
            return _fallback_summary(present_names, absent_names, total_roster, present_pct)
        return text
    except Exception as e:
        print(f"[report_generator] Generation failed: {e}")
        return _fallback_summary(present_names, absent_names, total_roster, present_pct)


def _fallback_summary(present_names, absent_names, total_roster, present_pct):
    absent_str = ", ".join(absent_names) if absent_names else "none"
    return (
        f"Attendance recorded for {total_roster} student(s) on the roster. "
        f"{len(present_names)} were present and {len(absent_names)} were absent "
        f"({present_pct:.1f}% attendance rate). "
        f"Absent: {absent_str}."
    )


# ---------------------------------------------------------------------------
# Optional upgrade: call Claude for a higher-quality written report instead.
# Requires an ANTHROPIC_API_KEY environment variable.
#
# import anthropic
# def generate_summary_claude(present_names, absent_names, total_roster):
#     client = anthropic.Anthropic()
#     msg = client.messages.create(
#         model="claude-sonnet-5",
#         max_tokens=200,
#         messages=[{"role": "user", "content": (
#             f"Write a short professional attendance summary. Roster: {total_roster}. "
#             f"Present: {present_names}. Absent: {absent_names}."
#         )}],
#     )
#     return msg.content[0].text
# ---------------------------------------------------------------------------
