import json
import re
import string


import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

ID2ENTITY_TYPE_ZH = {
    "10": "小说",
    "20": "电视剧",
    "30": "电影",
    "40": "动漫",
    "50": "纪录片",
    "60": "综艺",
    "70": "小说",
    "80": "短剧",
    "81": "短剧",
    "110": "音乐",
    "111": "音乐"
}

IN_HOUSE_TYPE = {
    "70": {"authority_site": "番茄小说", "url": "https://fanqienovel.com", "re_pattern": r'book_id%3D([^%]+)%'},
    "81": {"authority_site": "红果短剧", "url": "https://novelquickapp.com", "re_pattern": r'series_id=([^&]+)&'}
}

SERIAL_STATUS_DICT = {0: "未知", 1: "连载中/更新中", 2: "已完结", 3: "未上映"}


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def extract_first_json(text):
    # 使用正则匹配第一个完整的JSON对象
    match = re.search(r'\{[\s\S]*?\}(?=\s*第\s*\d+\s*帧图分析结果|$)', text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            print(text)
            return None
    return None

def normalize_str(input_str):
    norm_str = re.sub(r"\s+", "", input_str)
    punctuation = """《》！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘'‛“”„‟…‧﹏"""
    punctuation += string.punctuation
    translator = str.maketrans('', '', punctuation)

    return norm_str.translate(translator)
