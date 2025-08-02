from langchain_community.llms import Cohere



def call_llm(prompt):
    llm = Cohere(
    cohere_api_key="d8pxfJhPz2HBHKZdiZBxGzsn5Bo17efkFJvW9hly",
    model="command",           # or "command-light", "command-nightly", etc.
    temperature=0.1,
    )

    # `prompts` should be a list of strings
    response = llm.generate(prompts=[prompt])
    return response.generations[0][0].text


def get_default_message():
    return """
    يبدو أنني لم أتمكن من العثور على إجابة لسؤالك. يُرجى المحاولة بسؤال آخر.
    """
