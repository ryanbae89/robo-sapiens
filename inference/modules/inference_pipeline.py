# inference/modules/inference_pipeline.py
import random
import time
import torch

from inference.modules.prompt_utils import format_instructions, format_prompt

def generate_conversation(model, 
                          tokenizer, 
                          context, 
                          inference_params, 
                          generation_params, 
                          prompt_params,
                          verbose=False):

    # Grab conversation generation params
    speakers = generation_params["speakers"]
    n_rounds = generation_params.get("n_rounds", 10)
    time_delay = generation_params.get("time_delay", 2)
    
    # Grab prompt params
    group_description = prompt_params["group_description"]
    
    # Debugging output
    if verbose: print(context)
        
    # Loop thru n_rounds
    for i in range(n_rounds):
        
        # Time delay for typing realism
        time.sleep(time_delay)
        
        # Randomly sample from list of speakers
        next_speaker = speakers[random.randint(0, len(speakers)-1)]
        
        # Format instructions and prompt
        instruction = format_instructions(group_description, next_speaker, speakers)
        prompt = format_prompt(sample={"instruction": instruction, 
                                       "input": context})
        
        # Make next message inference and concatenate to existing context
        response = generate_response(model, tokenizer, prompt, inference_params)
        context += response
        
        # Debugging output
        if verbose: print(response)
        
    return context

def generate_response(model, 
                      tokenizer, 
                      prompt, 
                      inference_params):
    """
    """
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

    temperature = inference_params.get("temperature", 1.0)
    max_new_tokens = inference_params.get("max_new_tokens", 256)
    top_p = inference_params.get("top_p", 0.9)
    repetition_penalty = inference_params.get("repetition_penalty", 1.0)
    do_sample = inference_params.get("do_sample", True)

    with torch.inference_mode() and torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample
        )
    message_start_ind = len(input_ids[0])
    return tokenizer.decode(output_ids[0][message_start_ind:], skip_special_tokens=True)