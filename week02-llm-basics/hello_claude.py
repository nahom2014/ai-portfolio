from dotenv import load_dotenv
import anthropic


load_dotenv()                    # puts ANTHROPIC_API_KEY into the environment
client = anthropic.Anthropic()   # the SDK reads the key from the environment

def is_truncated(reason: str) -> bool:
    return reason == "max_tokens"


def ask(question: str, model: str = "claude-haiku-5-5", max_tokens: int = 500) -> tuple[str, int, int]:  # NEW return type
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": question}],
    )

    if is_truncated(message.stop_reason):
        print("[WARN] answer was cut off")

    answer = ""
    for block in message.content:
        if block.type == "text":
            answer += block.text
    return answer, message.usage.input_tokens, message.usage.output_tokens  # NEW: 3 values

USD_JPY = 150.0   # replace with today's rate

PRICES = {   # USD per 1 million tokens: (input, output)
    "claude-haiku-5-5":  (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
}

def cost_yen(model: str, input_tokens: int, output_tokens: int) -> float:
    input_price, output_price = PRICES[model]                  # step 1: look up prices
    input_cost = input_tokens * input_price / 1_000_000        # step 2: tokens × price
    output_cost = output_tokens * output_price / 1_000_000     #         (prices are per 1 million)
    usd = input_cost + output_cost
    return usd * USD_JPY          

if __name__ == "__main__":
    assert round(cost_yen("claude-haiku-5-5", 25, 89), 5) == 0.00705

    questions = [
        "Explain what an API rate limit is in two sentences.",
        "APIのレート制限とは何か、2文で説明してください。",
    ]
    # for model in ["claude-haiku-5-5", "claude-sonnet-5-5"]:
    #     for question in questions:
    #         answer, inp, out = ask(question, model=model)
    #         print(f"{model} | {question[:15]} | {inp} in / {out} out | ¥{cost_yen(model, inp, out):.4f}")
    calls_per_month = 1000 * 30                       # step 2
    for model in ["claude-haiku-5-5", "claude-sonnet-5-5"]:
        one_call = cost_yen(model, 500, 300)          # step 1
        monthly = one_call * calls_per_month          # step 3
        print(f"{model}: ¥{one_call:.4f} per call, ¥{monthly:,.0f} per month")