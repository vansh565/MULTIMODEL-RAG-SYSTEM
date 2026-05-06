from src.llm.llm_client import generate

def generate_answer(query, local_data, global_data):
    context = "\n".join(local_data) + "\n" + "\n".join(global_data)

    prompt = f"""
You MUST answer in ONLY 3 lines.
Do NOT explain.
Keep answer short.
donot use page number ,use only context of the pdf
Context:
{context}

Question:
{query}

Answer:
"""

    response = generate(prompt)

    return " ".join(response.split()[:25])