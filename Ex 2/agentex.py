import os
from dotenv import load_dotenv
from groq import Groq

# -------------------------------------------------
# Setup
# -------------------------------------------------
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("GROQ_API_KEY not found in .env")

client = Groq(api_key=api_key)

# -------------------------------------------------
# Tools
# -------------------------------------------------
def calculator(expression: str) -> str:
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

TOOLS = {
    "calculator": calculator
}

# -------------------------------------------------
# System Prompt
# -------------------------------------------------
SYSTEM_PROMPT = """
You are an intelligent AI agent.

You may decide to use tools.

Available tools:
- calculator(expression)

When you want to use a tool, respond EXACTLY like:

TOOL: calculator
INPUT: <expression>

When you receive a tool result, use it and continue.
When finished, respond with the final answer only.
"""

# -------------------------------------------------
# LLM Call
# -------------------------------------------------
def call_llm(messages):
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        temperature=0
    )
    return completion.choices[0].message.content.strip()

# -------------------------------------------------
# Agent Loop
# -------------------------------------------------
def run_agent(user_question, max_steps=5):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_question}
    ]

    for step in range(max_steps):
        reply = call_llm(messages)
        print(f"\n🤖 LLM:\n{reply}")

        if reply.startswith("TOOL:"):
            lines = reply.splitlines()
            tool_name = lines[0].split(":")[1].strip()
            tool_input = lines[1].replace("INPUT:", "").strip()

            if tool_name in TOOLS:
                result = TOOLS[tool_name](tool_input)
            else:
                result = "Unknown tool"

            print(f"\n🔧 Tool result:\n{result}")

            messages.append({"role": "assistant", "content": reply})
            messages.append({
                "role": "assistant",
                "content": f"Tool result: {result}"
            })
        else:
            break

# -------------------------------------------------
# Run (User Input)
# -------------------------------------------------
if __name__ == "__main__":
    print("🤖 Groq Agent (type 'exit' to quit)\n")

    while True:
        question = input("🧑 You: ")
        if question.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        run_agent(question)
