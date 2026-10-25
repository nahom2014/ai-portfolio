from datetime import date
from anthropic.types import MessageParam
from hello_claude import client, cost_yen, is_truncated

MODEL = "claude-haiku-5-5"


SYSTEM_PROMPT = f"""Today is {date.today().strftime('%Y-%m-%d (%A)')}.
You are the front-desk assistant of Sakura Hotel in Tokyo.

Rules:
- Answer in the same language the customer uses.
- Keep answers short: 3 sentences or less.
- If you don't know something about the hotel, say so. Never invent details.
- You cannot check availability or make bookings. If asked, tell the customer to call the front desk at 03-1234-5678.
- Only help with questions about the hotel and the customer's stay. Politely decline other requests.

Hotel facts:
- Check-in: 15:00, check-out: 10:00
- Parking: available, ¥1,500 per night, 10 spaces, no reservation needed
- Breakfast: 7:00–9:30, ¥2,000 per person"""


def main() -> None:
    messages: list[MessageParam] = []   # the whole conversation, kept by YOU
    total_yen = 0.0
    turn = 0

    print("Chat started. Type 'exit' to quit.")
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if user_input == "exit":
            break
        if not user_input:
            print("empty input")
            continue

        messages.append({"role": "user", "content": user_input})

        # NEW: stream the answer instead of waiting for all of it
        print("Claude: ", end="", flush=True)
        with client.messages.stream(
            model=MODEL,
            max_tokens=1024,          # ceiling for thinking + answer together
            system=SYSTEM_PROMPT,
            messages=messages,        # the WHOLE history
        ) as stream:
            for text in stream.text_stream:
                print(text, end="", flush=True)
            response = stream.get_final_message()
        print()                       # new line after the streamed answer

        # count the cost first: you pay even when the answer is empty
        turn += 1
        inp = response.usage.input_tokens
        out = response.usage.output_tokens
        thinking = response.usage.output_tokens_details.thinking_tokens
        yen = cost_yen(MODEL, inp, out)
        total_yen += yen

        reply = "".join(b.text for b in response.content if b.type == "text")
        if is_truncated(response.stop_reason):
            print("[WARN] answer was cut off")

        if not reply:
            print("[ERROR] no answer received, please ask again")
            messages.pop()            # remove the unanswered question, keep history valid
        else:
            messages.append({"role": "assistant", "content": reply})   # already printed above

        print(f"[turn {turn}] {inp} in / {out} out (thinking {thinking}) | ¥{yen:.4f} | total ¥{total_yen:.4f}")

    print(f"\nBye! {turn} turns, total cost ¥{total_yen:.4f}")


if __name__ == "__main__":
    main()