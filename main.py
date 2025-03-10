import os
import random
import torch
import time
from datetime import datetime, timezone, timedelta

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from transformers import AutoTokenizer, LlamaForCausalLM
from peft import PeftModel

import firebase_admin
from firebase_admin import credentials, firestore


# Initialize FastAPI app
app = FastAPI()

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000",
                   "https://robo-lads.web.app",
                   "https://bb66-174-21-171-28.ngrok-free.app"],  # Use ["http://localhost:3000"] to restrict to your React app during development
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers (e.g., Content-Type, Authorization)
)

# Initialize Firebase Admin SDK
service_account_info = os.environ.get("SERVICE_ACCOUNT_JSON")
if not service_account_info:
    raise HTTPException(status_code=500, detail="Firebase service account JSON not set.")
cred = credentials.Certificate(service_account_info)
firebase_admin.initialize_app(cred)

# Globals to hold the model and tokenizer
model = None
tokenizer = None
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the model and tokenizer (GPU-enabled)
MODEL_NAME = "robo-lads"
MODEL_ID = "meta-llama/Llama-2-7b-chat-hf"
REPOSITORY_ID = f"{MODEL_ID.split('/')[1]}-{MODEL_NAME}"  # the id of your huggingface repository where the model will be stored


# Define input schema for the API request
class ConversationRequest(BaseModel):
    chat_info: dict
    speaker: str
    n_rounds: int
    forget_time: int
    dispatch: bool
    

# # GPU Check Endpoint
# @app.get("/gpu-check")
# async def gpu_check():
#     return {"gpu_available": torch.cuda.is_available()}

# # Model/tokenizer check Endpoint
# @app.get("/model-check")
# async def model_check():
#     global model, tokenizer
#     if model and tokenizer:        
#         return {"model_loaded": True}
#     else:
#         return {"model_loaded": False}

# Container Status Endpoint
@app.get("/container-status-check")
async def check_status():
    res = {"gpu_available": torch.cuda.is_available(),
           "model_available": False}
    # check if model is loaded
    global model, tokenizer
    if model and tokenizer:
        res["model_available"] = True
    else:
        res["model_available"] = False
    return res

# Wake Endpoint (Load Model and Tokenizer)
@app.post("/load-model")
async def wake():
    global model, tokenizer
    try:
        # Load the model and tokenizer into GPU memory
        if torch.cuda.is_available():
            # empty GPU cache
            torch.cuda.empty_cache()
            device_map = {"": 0}
            print("Loading model and tokenizer to GPU...")
            # download from hugging face
            # config = PeftConfig.from_pretrained("ryanbae89/robo-lads")
            # model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", 
            #                                              load_in_8bit=True,
            #                                              device_map=device_map)
            # model = PeftModel.from_pretrained(model, "ryanbae89/robo-lads", device_map=device_map)
            # load finetuned llama-2 model
            model = LlamaForCausalLM.from_pretrained(
                MODEL_ID,
                load_in_8bit=True,
                device_map=device_map,
            )
            # load the pre-trained LoRA model from the lora-alpaca folder
            model = PeftModel.from_pretrained(model, os.path.join(REPOSITORY_ID, "checkpoint-5000"), torch_dtype=torch.float16)
            model.eval()  # Ensure model is in evaluation mode
            # load tokenizer
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", 
                                          trust_remote_code=True,
                                        #   pad_token=-100,
                                          padding_side="left")
            return {"status": "Model and tokenizer loaded successfully"}
        else:
            return {"status": "GPU not available. Unable to load model to GPU."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {e}")
    
# Post message to firebase
def post_message(chat_info: dict, sender: str, message: str):
    """
    Post a message to Firebase, which will function as adding to the existing context.
    """
    # short delay to make the conversation more realistic on the front-end
    time.sleep(0.25)
    try:
        # Connect to Firestore
        db = firestore.client()

        # Define chat and message structure
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

        if sender not in senders:
            raise HTTPException(status_code=400, detail="Invalid sender. Please use a valid sender name.")
        sender_info = senders[sender]

        # Add message to Firestore
        chat_doc_ref = db.collection(chat_info["collection"]).document(chat_info["document"])
        chat_messages_ref = chat_doc_ref.collection('messages')
        chat_messages_ref.add({
            'timestamp': datetime.now(timezone.utc),
            'message': message,
            'uid': sender_info['uid'],
            'photo': sender_info['photo'],
            'email': sender_info['email'],
            'displayName': sender_info['displayName'],
        })

        return {"status": "Message posted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error posting message: {e}")
    
# Conversation generation endpoint
@app.post("/generate-conversation")
async def generate_conversation(request: ConversationRequest):
    
    # check if model and tokenizer are loaded on GPU
    global model, tokenizer
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Model and tokenizer are not loaded. Use the /wake endpoint first.")

    # get context from current chat
    context = get_context(chat_info = request.chat_info,
                          forget_time_min=request.forget_time,
                          max_lookback=25)
    
     # Default random rounds if not provided
    if request.n_rounds < 1:
        n_rounds = random.randint(3, 10)
    else:
        n_rounds = request.n_rounds

    # get list of possible speakers
    speakers = ["Ryan Bae", "Eric Wu", "Aladin Corhodzic", "Kyle Matsuo", "Lyle Johnson", "Andrija Perovic"]
    if request.speaker in speakers:
        speakers.remove(request.speaker)

    # loop n_rounds times to generate conversation
    full_context = context
    for _ in range(n_rounds):
        # randomly pick next speaker
        next_speaker = random.choice(speakers)
        # generate prompt and inference
        prompt = format_test_instruction({"input": full_context, "responder_name": next_speaker})
        response = predict_message(prompt)
        try:
            message = response.split(f"{next_speaker}: ")[1]
            # add to existing context
            full_context += f"\n{next_speaker}: {message.strip()}"
            # full_context += f"\n{message.strip()}"
            # dispatch message to firebase
            if request.dispatch:
                post_message(chat_info=request.chat_info, sender=next_speaker, message=message)
        except Exception as e:
            # Print the invalid message and the exception details
            print(f"Invalid message generated, discarded. Message: {response}")
            print(f"Error: {e}")
            continue  # Skip to the next iteration of the loop
    return {"conversation": full_context}

# Model inference function
def predict_message(prompt: str):
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
    with torch.inference_mode() and torch.no_grad():
        outputs = model.generate(input_ids=input_ids, 
                                        max_new_tokens=128,
                                        top_p=0.85,
                                        temperature=0.95,
                                        repetition_penalty=1.00)
        message_start_ind = len(input_ids[0])
        message = tokenizer.decode(outputs[0][message_start_ind:], skip_special_tokens=True)
    return message

# Helper function to format instruction
def format_test_instruction(sample):
    return f"""### Instruction:
Your name is {sample['responder_name']}. You are in a group chat with 5 of your best friends: Ryan Bae, Eric Wu, Aladin Corhodzic, Kyle Matsuo, Lyle Johnson, Andrija Perovic. 
Most of you went to college together at UCLA and lived and hung out together in California, and talk to each other with no filter. 
You are encouraged to say generally inappropriate and hilarious things, make fun of each other, and talk freely. 
Everything is in good fun, so remember to joke and laugh, and be funny, but also answer all questions clearly. 
You will be presented with the most recent messages in the group chat. Write a response to the conversation as {sample['responder_name']}.

### Context:
{sample['input']}

### Response:"""

# Get context from existing conversation
def get_context(chat_info: dict, forget_time_min: int, max_lookback: int):
    try:
        # Connect to Firestore
        db = firestore.client()
        chat_doc_ref = db.collection(chat_info["collection"]).document(chat_info["document"])
        chat_messages_ref = chat_doc_ref.collection('messages')

        # Get recent messages
        messages = chat_messages_ref.order_by('timestamp', direction=firestore.Query.DESCENDING).limit(max_lookback).get()
        context = ""
        for msg in messages[::-1]:
            raw = msg.to_dict()
            # check timestamp
            message_timestamp = raw['timestamp']
            current_time = datetime.now(timezone.utc)
            time_diff = current_time - message_timestamp
            if time_diff <= timedelta(minutes=forget_time_min):
                message = f"{raw['displayName']}: {raw['message']}"
                context += message + "\n"
        return context
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching Firestore context: {e}")
