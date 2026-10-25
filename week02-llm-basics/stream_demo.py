import time

from hello_claude import client

MODEL = "claude-haiku-5-5"
messages = [{"role": "user", "content": "Explain in about 150 words why hotels use online reservation systems."}]

# 1. Without streaming
start = time.perf_counter()
response = client.messages.create(model=MODEL, max_tokens=1024, messages=messages)
total = time.perf_counter() - start
print("".join(b.text for b in response.content if b.type == "text"))
print(f"\n[no stream] first text after {total:.2f}s, finished after {total:.2f}s\n")

# 2. With streaming
start = time.perf_counter()
first = None
with client.messages.stream(model=MODEL, max_tokens=1024, messages=messages) as stream:
    for text in stream.text_stream:
        if first is None:
            first = time.perf_counter() - start    # time until the FIRST piece arrives
        print(text, end="", flush=True)
total = time.perf_counter() - start
print(f"\n[stream] first text after {first:.2f}s, finished after {total:.2f}s")