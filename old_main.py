from pydantic import BaseModel
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
from transformers import pipeline, AutoModelForCausalLM, LlamaTokenizerFast
import json
class AnswerFormat(BaseModel):
    first_name: str
    last_name: str
    year_of_birth: int
    num_seasons_in_nba: int

# Create a transformers pipeline
mdl = AutoModelForCausalLM.from_pretrained("princeton-nlp/Sheared-LLaMA-1.3B", device_map="auto", offload_folder="offload")

hf_pipeline = pipeline('text-generation', model=mdl, tokenizer=LlamaTokenizerFast.from_pretrained("princeton-nlp/Sheared-LLaMA-1.3B"))
prompt = f'Here is information about Michael Jordan in the following json schema: {json.dumps(AnswerFormat.model_json_schema(), indent=4)} :\n'

# Create a character level parser and build a transformers prefix function from it
parser = JsonSchemaParser(AnswerFormat.model_json_schema())
prefix_function = build_transformers_prefix_allowed_tokens_fn(hf_pipeline.tokenizer, parser)

# Call the pipeline with the prefix function
output_dict = hf_pipeline(prompt)#, prefix_allowed_tokens_fn=prefix_function)

# Extract the results
result = output_dict[0]['generated_text'][len(prompt):]
print(result)
# {'first_name': 'Michael', 'last_name': 'Jordan', 'year_of_birth': 1963, 'num_seasons_in_nba': 15}