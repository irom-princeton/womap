import gc
import numpy as np
import torch
from transformers import AutoProcessor, AutoTokenizer, AutoModelForZeroShotObjectDetection, CLIPTextModelWithProjection
from sentence_transformers import SentenceTransformer

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# reward - grounding dino
# setup GroundingDINO
g_dino_model_id = "IDEA-Research/grounding-dino-base"
g_dino_processor = AutoProcessor.from_pretrained(g_dino_model_id)
g_dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(g_dino_model_id).to(
    device
)

# object to query
target_object: str = "mug"

# query text
g_dino_query_text = f"{target_object}. table. random. object. background."

def compute_reward(
    raw_image,
    input_processor=g_dino_processor,
    input_model=g_dino_model,
    query_text=g_dino_query_text,
    target_object=None,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    """
    Compute the reward based on the object detection confidence score.
    """

    # set the target object
    if target_object is None:
        target_object = query_text.split(".")[0]

    # process inputs
    inputs = input_processor(images=raw_image, text=query_text, return_tensors="pt").to(
        device
    )

    # run the model
    with torch.no_grad():
        outputs = input_model(**inputs)

    # post-process the ouputs
    results = input_processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.2,
        text_threshold=0.2,
        target_sizes=[raw_image.size[::-1]],
    )

    # extract the outputs
    bboxes = results[0]["boxes"]  # top-left, bottom-right
    areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
    labels = results[0]["labels"]
    scores = results[0]["scores"]
    
    detected_outputs =  list(
            (idx, score.item())
            for idx, (score, label, area) in enumerate(zip(scores, labels, areas))
            if label.replace(" ", "") == target_object
    )
    
    if len(detected_outputs) == 0:
        return 0.0, [0, 0, 0, 0]
        
    detected_outputs = sorted(detected_outputs, key=lambda x: x[1], reverse=True)
    max_detection_id = detected_outputs[0][0]
    target_score = scores[max_detection_id].item()
    box = bboxes[max_detection_id]

    return target_score, box.detach().cpu().tolist()

def compute_text_embeddings(
    text,
    model_arch="sentence_transformer",
):
    """
    Compute the text embeddings of the target objects.
    """
    
    if model_arch.lower() == "clip":
        # model's name
        model_name = "openai/clip-vit-base-patch32"
        
        # model (with projection the shared image-text embedding space)
        model = CLIPTextModelWithProjection.from_pretrained(model_name)
        
        # tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # tokenize the texts
        text_inputs = tokenizer(text, padding=True, return_tensors="pt")
        
        with torch.no_grad():
            # compute the text embeddings
            outputs = model(**text_inputs)
            
            # embeddings
            text_embeds = outputs.text_embeds
        
    elif model_arch.lower() == "sentence_transformer":
        # model
        model = SentenceTransformer("all-MiniLM-L6-v2")

        # compute language embeddings
        text_embeds = torch.tensor(model.encode(text))
        
        # tokenizer
        tokenizer = None
        
    # free the memory
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    
    return text_embeds