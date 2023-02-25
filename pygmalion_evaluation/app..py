import logging

import uvicorn

from pygmalion_evaluation.model import build_model_and_tokenizer_for, run_raw_inference
from pydantic import BaseModel

import csv

from pygmalion_evaluation.parsing import parse_messages_from_str

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

model_name = 'PygmalionAI/pygmalion-350m'

model, tokenizer = build_model_and_tokenizer_for(model_name)

from fastapi import FastAPI

app =FastAPI()

class Request(BaseModel):
    do_sample: bool
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int
    typical_p: float
    repetition_penalty: float
    penalty_alpha: float
    prompt: str
    user_message: str
    char_name: str

@app.post("/")
def generate(request:Request):
    model_output = run_raw_inference(model, tokenizer, request.prompt, request.user_message, 
                           do_sample = request.do_sample,
                            max_new_tokens = request.max_new_tokens,
                            temperature = request.temperature,
                            top_p = request.top_p,                            
                            top_k = request.top_k,
                            typical_p = request.typical_p,
                            repetition_penalty = request.repetition_penalty,
                            penalty_alpha = request.penalty_alpha

                                     )
    generated_messages = parse_messages_from_str(model_output,
                                                     ["You", request.char_name])
    logger.debug("Parsed model response is: `%s`", generated_messages)
    bot_message = generated_messages[0]

    return {'response': bot_message, 'model':model_name}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)