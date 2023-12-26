from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from pydantic import BaseModel
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
import json


class AnswerFormat(BaseModel):
    score: int
    tags: list[int]
    summary: str

model=AutoModelForCausalLM.from_pretrained("princeton-nlp/Sheared-LLaMA-1.3B")
tokenizer=AutoTokenizer.from_pretrained("princeton-nlp/Sheared-LLaMA-1.3B")

pp=pipeline("text-generation",model=model,tokenizer=tokenizer)

parser = JsonSchemaParser(AnswerFormat.model_json_schema())
prefix_function = build_transformers_prefix_allowed_tokens_fn(pp.tokenizer, parser)
prompt = """Task: Detection of Dark Patterns in UI/UX

Dark Pattern Flags:

Major Flags:
  Subscription trap (subscription_trap) [ Websites redirect you to subscriptions pages or makes it hard for you to cancel a subscription ]
  Drip pricing (drip_pricing) [ Additional fees added upon checkout ]
  False Urgency  (false_urgency) [ False information about an item being low in stock to create urgency ]
  Forced action (forced_action) [ Forcefully making a customer sign up for subscriptions ]
  Basket sneaking (basket_sneaking) [ Sneakily adding subscriptions/warranties to user's cart ]
  Confirm shaming (confirm_shaming) [ Guilting user to opt into something ]
  Bait and switch (bait_and_switch) [ Showing misleading price on a homepage and chaging it on the product's page ]

Minor Flags:
  Interface interference (interface_interference) [ Obscuring relevant information in a webpage ] 
  Disguised advertisement (disguised_ad) [ Making it difficult to identify between fake and real ads ] 
  Nagging (nagging) [ Constant requests to do something ]

Input Type: Raw HTML or Plain text
Output Schema(strict):

{
    "output": {
        "score": 1 to 10,
        "tags": ["tag 1", "tag 2", ...] 
        "summary": "explanation" 
    }
}

Scoring System(strict):
    10: No Flags Detected
    Between 7 to 9: 1 minor or 1 major flag detected
    Between 4 to 6: More than 1 minor flags detected or 1 major flag detected
    Between 1-3: Several Major and Minor flags detected


Examples:

Example Input 1: <div><h1>Buy Amazon Prime to get the following benfits</h1><ul><li>Easy Discounts</li><li>One Day Delivery</li></ul></div>
Example Output 1: 
{ 
    "output": {
        "score":4,
        "tags": ["basket_sneaking", "interface_interference", "nagging"],
        "summary":"Basket Sneaking: Website makes attempt to sell subscription before order placement. Interface Interference: Popup is shown interrupting customer action. Nagging: Website constantly prompts user to buy their subscription"
    }
}

Example Input 2: <div><h1>Place Order</h1><span>Order Total: 500RS</span><button>Checkout</button></div>
Example Output 2:  
{ 
    "output": {
        "score":10,
        "tags": [],
        "summary":"No Popups or Flags Detected"
    }
}, Now answer the response here: """ + AnswerFormat.schema_json() + ":\n"
output_dict = pp(prompt, prefix_allowed_tokens_fn=prefix_function)
print(output_dict)
result = output_dict[0]['generated_text'][len(prompt):]
print(result)