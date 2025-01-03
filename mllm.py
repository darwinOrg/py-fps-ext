import torch
from loguru import logger
from modelscope import AutoModelForCausalLM, AutoTokenizer, snapshot_download
from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor


def chat_completion(model_name: str, prompt: str) -> str:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=8192
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    logger.info(f'chat_completion({model_name}) success')
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


def batch_chat_completion(model_name: str, prompts: list[str]) -> list[str]:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

    messages_list = [[{"role": "user", "content": prompt}] for prompt in prompts]
    texts = [tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for messages in
             messages_list]
    model_inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=128
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    logger.info(f'batch_chat_completion({model_name}) success')
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)


def vl_ocr(model_name: str, prompt: str, image_url: str) -> str:
    model_dir = snapshot_download(model_name)

    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
    min_pixels = 256 * 28 * 28
    max_pixels = 1280 * 28 * 28
    processor = AutoProcessor.from_pretrained(model_dir, min_pixels=min_pixels, max_pixels=max_pixels,
                                              padding_side="left")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_url,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=8192)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    logger.info(f'vl_ocr({model_name}, {image_url}) success')
    return processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )


def batch_vl_ocr(model_name: str, prompt: str, image_urls: list[str]) -> list[str]:
    model_dir = snapshot_download(model_name)

    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
    min_pixels = 256 * 28 * 28
    max_pixels = 1280 * 28 * 28
    processor = AutoProcessor.from_pretrained(model_dir, min_pixels=min_pixels, max_pixels=max_pixels,
                                              padding_side="left")

    messages_list = []
    for image_url in image_urls:
        messages_list.append([
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_url,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ])

    # Preparation for inference
    texts = [processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for messages in
             messages_list]
    image_inputs, video_inputs = process_vision_info(messages_list)

    assert len(image_inputs) == len(messages_list), "Number of image inputs does not match the number of messages"

    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=8192)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    logger.info(f'vl_ocr({model_name}, {image_urls}) success')
    return processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
