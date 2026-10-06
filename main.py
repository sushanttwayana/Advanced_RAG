import os
from dotenv import load_dotenv
from openai import OpenAI


# Load variables from .env
load_dotenv()

# Get OpenRouter API key
api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise ValueError(
        "OPENROUTER_API_KEY not found. "
        "Please add it to your .env file."
    )


# Create OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)


# Model you want to use
MODEL = "anthropic/claude-haiku-4.5"


def ask_model(user_query: str) -> str:

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": user_query,
            }
        ],
    )

    return response.choices[0].message.content


def main():

    print("=" * 60)
    print("OpenRouter - Claude Haiku 4.5")
    print("=" * 60)
    print("Type 'exit' to quit.\n")

    while True:

        user_query = input("You: ")

        if user_query.lower() == "exit":
            print("Goodbye!")
            break

        if not user_query.strip():
            continue

        try:
            answer = ask_model(user_query)

            print("\nClaude Haiku 4.5:")
            print(answer)
            print()

        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()