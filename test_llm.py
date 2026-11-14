
from ollama import Client
import time

client = Client(host="http://127.0.0.1:11434")

start = time.perf_counter()

response = client.chat(
    model="qwen3:8b",
    messages=[
        {
            "role": "system",
            "content": (
                "Ты корпоративный ассистент. "
                "Отвечай кратко и на русском языке."
            ),
        },
        {
            "role": "user",
            "content": "Что такое RAG?"
        },
    ],
    options={
        "temperature": 0.1,
        "num_ctx": 4096,
        "num_predict": 300,
    },
    think=False,
)

elapsed = time.perf_counter() - start

print(response.message.content)
print(f"\nLatency: {elapsed:.2f} sec")