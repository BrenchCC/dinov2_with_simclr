"""
Doubao-vison inference code, should add ARK_API_KEY for model calling.
"""
import argparse
import base64
import json
import re
import os

from pyarrow.fs import FileSystem
from volcenginesdkarkruntime import Ark

HDFS_CLIENT, _ = FileSystem.from_uri('hdfs://haruna/home/')

current_dir = os.path.dirname(os.path.abspath(__file__))
AI_RESOURCE_RECOGNITION_SP = open(os.path.join(current_dir, "./prompts/resource_recognition_sp.md")).read()


def parse_args():
    parser = argparse.ArgumentParser(description="dinvov2 inference")
    
    # 必须参数
    parser.add_argument(
        "--model_ep", type=str, required=True, 
        help="doubao vision model endpoint"
    )
    parser.add_argument(
        "--image_path", type=str, required=True, 
        help="image paths, splitted by comma."
    )
    
    args = parser.parse_args()
    return args


def vision_inference(user_prompt, images, ep, system_prompt=None, image_type="jpeg"):
    client = Ark(base_url="https://ark.cn-beijing.volces.com/api/v3")

    if system_prompt:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": []}
        ]
    else:
        messages = [
            {"role": "user", "content": []}
        ]
    
    if not images: return None
    if user_prompt:
        messages[-1]['content'].append({"type": "text", "text": f"{user_prompt}"})
    for idx, image in enumerate(images):
        if image.startswith("hdfs"):
            with HDFS_CLIENT.open_input_file(image) as f:
                img_bytes = f.read()
        else:
            img_bytes = open(image, 'rb').read()
        base64_image = base64.b64encode(img_bytes).decode()
        if not user_prompt:
            messages[-1]['content'].append({"type": "text", "text": f"第 {idx} 帧图"})
        messages[-1]['content'].extend([
                {
                    "type": "image_url",
                    "image_url": {
                        "url":  f"data:image/{image_type};base64,{base64_image}"
                        # "detail": "low"
                    },
                },
            ]
        )
    try:
        result = client.chat.completions.create(
            model=ep, 
            messages=messages, 
            temperature=0.8
        )
        prompt_tok = result.usage.prompt_tokens
        completion_tok = result.usage.completion_tokens
        result = result.choices[0].message.content
        return result, prompt_tok, completion_tok
    except Exception as e:
        print(f"Model calling error: {e}")
        return "", "", ""

def recognize_entity_with_doubao(image_path_list, model_ep, system_prompt, num_images=3):
    #TODO: Update system prompt for low-quality filtering.

    if len(image_path_list) == 0:
        raise ValueError("image_path_list is empty.")
    if len(image_path_list) > num_images:
        image_path_list = image_path_list[::len(image_path_list) // num_images]
    image_type = image_path_list[0].rsplit(".")[-1]
    pred, _, _ = vision_inference(
        user_prompt=None, 
        images=image_path_list, 
        ep=model_ep, 
        system_prompt=system_prompt,
        image_type=image_type,
    )
    # Parse model raw prediction.
    print("Doubao raw pred:", pred)
    try:
        # 提取JSON中的作品名称
        pred_str = pred.replace("```json", "").replace("```", "")
        pred_data = json.loads(pred_str)
        valid_flag = pred_data.get("商业短剧", "是")
        entity = pred_data.get("作品名称", "").strip("《》").strip()
    except:
        entity = ""
        valid_flag = "是" # by default, input frame is from valid video.
    
    return entity, valid_flag


if __name__ == "__main__":
    args = parse_args()
    image_path_list = args.image_path.split(",")
    model_ep = args.model_ep
    entity, valid_flag = recognize_entity_with_doubao(
        image_path_list, 
        model_ep, 
        AI_RESOURCE_RECOGNITION_SP
    )
    print(f"Resource extraction result: {entity}")
