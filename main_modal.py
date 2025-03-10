import modal

# create a shared volume to store weights
# volume = modal.NetworkFileSystem.persisted("robo-lads-vol")

# create a modal "stub" to handle config for functions
stub = modal.Stub(
    "robo-lads-predict",
    image=modal.Image.debian_slim().pip_install("firebase-admin",
                                                "numpy",
                                                "torch",
                                                "transformers",
                                                "peft",
                                                "scipy",
                                                "accelerate",
                                                "bitsandbytes").apt_install('git').run_commands('pip install git+https://github.com/huggingface/transformers')
)

@stub.cls(
    gpu=modal.gpu.T4(count=1), 
    secrets=[
        modal.Secret.from_name("my-huggingface-secret"),
        modal.Secret.from_name("my-firebase-secret"),
        ],
    allow_concurrent_inputs=1, 
    container_idle_timeout=60 * 10, 
    timeout=60 * 100)
class Model:
    def __enter__(self):
        
        # import libraries
        import os
        import json
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel
        
        import firebase_admin
        from firebase_admin import credentials
        from firebase_admin import firestore
        
        # firebase stuff
        service_account_info = json.loads(os.environ["SERVICE_ACCOUNT_JSON"])
        cred = credentials.Certificate(service_account_info)
        app = firebase_admin.initialize_app(cred)

        # Create a Firestore client
        self.db = firestore.client()
        
        # set device to GPU
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.cuda.empty_cache()
        device_map = {"": 0}
        
        # load model and tokenizer
        print("Downloading model and tokenizer from Hugging Face...")
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", 
                                                     use_auth_token=os.environ["HF_TOKEN"],
                                                     load_in_8bit=True,
                                                     device_map=device_map)
        self.model = PeftModel.from_pretrained(model, 
                                               "ryanbae89/robo-lads", 
                                               device_map=device_map)
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", 
                                                       padding_side="left",
                                                       trust_remote_code=True)
        print("Successfully loaded model and tokenizer!")   
        
    @modal.method(keep_warm=1)
    def generate_conversation(self, context, last_speaker, n_rounds, memory_min, wake):
        
        # import statements
        import random
        import traceback
        
        # just a way to wake up this function
        if wake:
            return
        
        # conditionally get 'background' context on chat if desired, helpful to keep conversations going across multiple prompts.
        full_context = ""
        background_context = self.get_firestore_context(memory_min)
        if len(background_context) > 0:
            full_context = background_context + '\n' + context
        else:
            full_context = context
        print(f"FULL CONTEXT:\n{full_context}")
        
        # get possible speakers
        speakers = ["Ryan Bae", "Eric Wu", "Aladin Corhodzic", "Kyle Matsuo", "Lyle Johnson", "Andrija Perovic"]
        speakers.remove(last_speaker)
        
        # if n_rounds is None, randomly pick # of responses
        if not n_rounds:
            n_rounds = random.randint(5, 10)
            
        # loop thru and generate conversation
        for i in range(n_rounds):
            try:
                next_speaker_ind = random.randint(0, len(speakers)-1)
                next_speaker = speakers[next_speaker_ind]
                response = self.predict_message(full_context, next_speaker)
                print(response.rstrip('\n'))
                full_context += response
                print(response.split(f"{next_speaker}: "))
                message = response.split(f"{next_speaker}: ")[1]
                m = self.dispatch_msg_to_firestore(message, next_speaker)
            except Exception:
                print('failed message', traceback.format_exc())
                break
    
    def predict_message(self, context: str, speaker: str):
        
        # import statements
        import torch
        
        # formatting function
        def format_test_instruction(sample):
	        return f"""### Instruction:
Your name is {sample['responder_name']}. You are in a group chat with 5 of your best friends: Ryan Bae, Eric Wu, Aladin Corhodzic, Kyle Matsuo, Lyle Johnson, Andrija Perovic. 
Most of you went to college together at UCLA and lived and hungout together in California, and talk to each other with no filter. 
You are encouraged to say generally inappropriate and hilarious things, make fun of each other, and talk freely. 
Everything is in good fun, so remember to joke and laugh, and be funny, but also answer all questions clearly. 
You will be presented with the most recent messages in the group chat. Write a response to the conversation as {sample['responder_name']}.

### Context:
{sample['input']}

### Response:"""

        prompt = format_test_instruction({"input": context, "responder_name": speaker})
        # test_prompt = context
        input_ids = self.tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
        with torch.inference_mode() and torch.no_grad():
            outputs = self.model.generate(input_ids=input_ids, 
                                          max_new_tokens=128,
                                          top_p=0.85,
                                          temperature=0.95,
                                          repetition_penalty=1.00)
            message_start_ind = len(input_ids[0])
            message = self.tokenizer.decode(outputs[0][message_start_ind:], skip_special_tokens=True)
        return message
    
    def dispatch_msg_to_firestore(self, message, sender):
        
        # imports
        from datetime import datetime, timezone
        import time

        # I delay to make the conversation more realistic on the front-end.
        time.sleep(0.25)
        senders = {
            'Aladin Corhodzic': {
                'uid': 'fake-aladin',
                'photo': 'https://imgur.com/GrPSmTe.jpg',
                'email': 'fake@email.com',
                'displayName': 'Aladin Corhodzic'
            },
            'Eric Wu': {
                'uid': 'fake-eric',
                'photo': 'https://imgur.com/xGGcktY.jpg',
                'email': 'fake@email.com',
                'displayName': 'Eric Wu'
            },
            'Kyle Matsuo': {
                'uid': 'fake-kyle',
                'photo': 'https://imgur.com/VbRn7Nc.jpg',
                'email': 'fake@email.com',
                'displayName': 'Kyle Matsuo'
            },
            'Lyle Johnson': {
                'uid': 'fake-lyle',
                'photo': 'https://imgur.com/Cd6C57M.jpg',
                'email': 'fake@email.com',
                'displayName': 'Lyle Johnson'
            },
            'Andrija Perovic': {
                'uid': 'fake-andrija',
                'photo': 'https://imgur.com/3JlJ7QG.jpg',
                'email': 'fake@email.com',
                'displayName': 'Andrija Perovic'
            },
            'Ryan Bae': {
                'uid': 'fake-ryan',
                'photo': 'https://imgur.com/DacQdmJ.jpg',
                'email': 'fake@email.com',
                'displayName': 'Ryan Bae'
            }
        }
        # get sender info
        sender = senders[sender]
        
        chat_doc_ref = self.db.collection('chats').document('<chatdb>')
        chat_messages_ref = chat_doc_ref.collection('messages')
        create_time, doc_ref = chat_messages_ref.add({
            'timestamp': datetime.now(timezone.utc),
            'message': message,
            'uid': sender['uid'],
            'photo': sender['photo'],
            'email': sender['email'],
            'displayName': sender['displayName'],
        })
        return create_time
    
    def get_firestore_context(self, forget_time_min=30):
        
        # imports
        from firebase_admin import firestore
        from datetime import datetime, timedelta,timezone

        # query chats collection in firestone database
        chat_doc_ref = self.db.collection('chats').document('<chatdb>')
        chat_messages_ref = chat_doc_ref.collection('messages')
        
        # get context of all messages previously within the last 5 min
        messages = chat_messages_ref.order_by('timestamp', direction=firestore.Query.DESCENDING).limit(15).get()
        context = ''
        prev = ''
        for i in messages[::-1]:
            raw = i.to_dict()
            if prev == raw['message']:
                return ''
            # check timestamp
            message_timestamp = raw['timestamp']
            current_time = datetime.now(timezone.utc)
            time_diff = current_time - message_timestamp
            # print(time_diff)
            if time_diff <= timedelta(minutes=forget_time_min):
                message = f"{raw['displayName']} : {raw['message']}"
                context += message
                context += '\n'
                prev = raw['message']
        return context
    
    # exists only to wake the container
    @modal.method(keep_warm=1)
    def wake():
        print('waking up')
    
    
# just for testing
# @stub.function()
# @modal.web_endpoint(label="response-test")
# def get_completion(context: str, speaker="Ryan Bae"):
#     from fastapi.responses import HTMLResponse
#     convo = Model().generate_conversation.call(context=context, last_speaker=speaker, n_rounds=None, wake=False)
#     to_render = convo.replace("\n", "<br />")
#     return HTMLResponse(to_render)

@stub.function()
@modal.web_endpoint(label="wake")
def wake():
   Model().generate_conversation.spawn('wake', "Ryan Bae", 1, memory_min=30, wake=True)
   print('waking up container')
   
@stub.function()
@modal.web_endpoint(label="alive")
def check_alive():
    print('Checking status of GPU container')
    status = Model().generate_conversation.get_current_stats()
    return status

@stub.function()
@modal.web_endpoint(label="response")
def get_completion():
    speaker = "Ryan Bae"
    print(f"Generating conversation...")
    Model().generate_conversation.remote(context="", 
                                         last_speaker=speaker, 
                                         n_rounds=None, 
                                         memory_min=1, 
                                         wake=False)


@stub.local_entrypoint()
def main():
    print("Waking up container...")
    # Model().wake().remote()
    Model.generate_conversation.spawn('wake', "Ryan Bae", 1, wake=True)
    
    speaker = "Ryan Bae"
    STARTING_TEXT = f"""{speaker}: where my bunion enthusiasts at?? @artdames"""
    n_rounds = 10
    Model().generate_conversation.remote(STARTING_TEXT, speaker, n_rounds, wake=False)
    
    # speaker = "Ryan Bae"
    # STARTING_TEXT = f"""{speaker}: who's better at tennis, me or @artdames??"""
    # n_rounds = 10
    # Model().generate_conversation.remote(STARTING_TEXT, speaker, n_rounds, wake=False)
    
    