import json
from corpus_utils import Agent

def build_prompt(history):
    prompt = ""
    for query, response in history:
        prompt += f"\n\n用户：{query}"
        prompt += f"\n\nChatGLM2-6B：{response}"
    return prompt

if __name__ == "__main__":
    agent = Agent()

    with open("corpus_gen/dataset/fact_b.json", 'rb') as file:
        data = json.load(file)
    items = data["items"]

    test_time = 1
    turn_time = 2

    for i in range(test_time):
        history = []
        history_re = []
        for _ in range(turn_time):
            question, history = agent.gen_question(items[0], history_re)
            response, history_re = agent.gen_resposne(history)
            print(question)
            print(response)