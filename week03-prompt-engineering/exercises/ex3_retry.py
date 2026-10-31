from pydantic import ValidationError

from ex1_weak_vs_strong import TEST_CASES
from ex2_json_output import (
    MODEL, SYSTEM, InquiryResult, client, get_text, has_devanagari,
)

MAX_ATTEMPTS = 3


def check_meaning(result: InquiryResult) -> str | None:
    """Return an error message, or None if the result is OK."""
    if result.language in ("ne", "hi") and not has_devanagari(result.suggested_reply):
        return "suggested_reply must be written in Nepali or Hindi (Devanagari script)."
    return None


def analyze(message: str) -> InquiryResult:
    messages = [{"role": "user", "content": f"<message>{message}</message>"}]

    for attempt in range(1, MAX_ATTEMPTS + 1):
        response = client.messages.create(
            model=MODEL,
            max_tokens=500,
            thinking={"type": "disabled"},
            system=SYSTEM,
            messages=messages,
        )
        text = get_text(response)

        # 1. Check the shape, then 2. check the meaning
        try:
            result = InquiryResult.model_validate_json(text)
            error = check_meaning(result)
        except ValidationError as e:
            details = "; ".join(
                f"{err['loc'][0] if err['loc'] else 'json'}: {err['msg']}"
                for err in e.errors()
            )
            error = f"Your JSON was invalid: {details}"

        if error is None:
            return result  # success

        print(f"  attempt {attempt} failed: {error[:80]}")

        # Add the bad answer and the error to the history, then try again
        messages += [
            {"role": "assistant", "content": text},
            {"role": "user", "content": f"{error}\nReturn the corrected JSON only."},
        ]

    raise ValueError(f"Failed after {MAX_ATTEMPTS} attempts")


if __name__ == "__main__":
    ok = 0
    for message, expected in TEST_CASES:
        print(f"\n=== {message[:40]}")
        try:
            result = analyze(message)
            ok += 1
            print(f"intent: {result.intent} (expected: {expected}) | reply: {result.suggested_reply[:50]}")
        except ValueError as e:
            print("FAILED:", e)

    print(f"\nSuccess: {ok}/{len(TEST_CASES)}")