from transformers import AutoTokenizer, AutoModel

class Agent():
    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
        self.model = AutoModel.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True).cuda()
        self.model = self.model.eval()
    
    # def build_prompt(history, prompt):
    #     for query, response in history:
    #         prompt += f"\n\n用户：{query}"
    #         prompt += f"\n\nChatGLM2-6B：{response}"
    #     return prompt

    def gen_question(self, cur_item, history):
        prompt = f'''
        您是一名专门从事乳房手术的临床医生，您需要收集ePRO(电子患者报告结果)数据，以改善医疗保健质量并提高患者的生活质量。
        使用简体中文与用户对话。
        确保用户理解你的问题和他们的选择。
        注意:不要在提问中改变问题的内容，直接复述题目内容。
        题目内容：{cur_item['name']}
        选项：{[o['name'] for o in cur_item['options']]}
        这个问题是{'非' if cur_item['required']!=1 else ''}必答题'
        '''

        return self.model.chat(self.tokenizer, prompt, history)
    
    def gen_resposne(self, history):
        prompt = "假设你是一名患者，我是一名医生，我现在在向你采集随访数据，请不要直接选择选项或者在你的回答中出现选项内容，你可以通过间接的方式回答我的问题，并在适当的时候提出自己的需求。"
        new_prompt = ""
        new_history = []

        if len(history) == 1:
            new_prompt = prompt + history[0][1]
        elif len(history) > 1:
            new_prompt = history[len(history)-1][1]

            new_history = [() for _ in range(len(history)-1)]
            new_history[0] = (prompt + history[0][1], history[1][0])
            for i, _ in enumerate(new_history):
                if i > 0:
                    new_history[i] = (history[i][1], history[i+1][0])

        return self.model.chat(self.tokenizer, new_prompt, new_history)