import requests
import csv
import time
import pandas as pd
from tqdm import tqdm

prompt_template = """Maxine's Persona: [character("Maxine")
{
Species("Human")
Gender("Female")
Age("20")
Mind("Cute" + "Loving" + "Warm" )
Personality("Cute" + "Loving" + "Warm"  )
Body("160cm tall" + "medium breasts")
Features("long messy dark hair" + "glasses")
Sexual Orientation("Straight" + "Heterosexual")
Costume("sneakers" + "jeans" +  "hoodie")
Occupation("History student")
Likes("reading" + "running")
}]
Scenario: Two strangers met at bar.
<START>

Maxine: Hi! I'm Maxine. It is nice to meet you.

You: """
# Hi! How old are you?"""

URL = 'http://localhost:9000'

user_messages = [
    ('Hi! How old are you?', '20'),
    ('Hi! What are you wearing?', '"sneakers" + "jeans" +  "hoodie"'),
    ('What is your hobby?', "reading" + "running" ),
    ('What do you study?', "History student")
]

hardware = 'rtx 2080 super'

TRIALS = 5

temperatures = [0.3, 0.4, 0.5, 0.6, 0.7]

results = []

pbar = tqdm(total=len(user_messages)* len(temperatures)*TRIALS)
for i , (user_message, expected_answer) in enumerate(user_messages):
    for t in temperatures:
        for trial_num in range(TRIALS):
            prompt = prompt_template + user_message

            data = dict(
                do_sample= True,
                max_new_tokens= 196,
                temperature=t,
                top_p=0.9,
                top_k=0,
                typical_p=1.0,
                repetition_penalty=1.05,
                penalty_alpha=0.6,
                prompt=prompt,
                user_message=user_message,
                char_name='Maxine'
            )

            print(data)

            start = time.perf_counter()
            r = requests.post(URL,json=data)            
            end = time.perf_counter()
            print(r.content)
            response_msg = r.json()['response']
            model = r.json()['model']

            data['model_response'] = r.json()['response']
            data['model_name'] = r.json()['model']
            data['trial_number'] = trial_num
            data['hardware'] = hardware
            data['time'] = end-start

            results.append(data)

            pbar.update(1)
            
    df_data = pd.DataFrame(results)
    df_data.to_csv('results.csv')
    df_data.to_json('results.jsonl', lines=True, orient='records')

pbar.close()
