from utils.prompt_llava import run_prompt 

if __name__ == "__main__":
    image_id = "00000_b"
    level_id = "level_0"

    results = run_prompt(image_id, level_id, prompt_strategy="visual")

    print(results)
    