import pandas as pd
import json
import os
import argparse
from tqdm import tqdm

# 动作映射
ACTION_MAP = {
    0: "Move Forward", 1: "Turn Left Small", 2: "Turn Right Small",
    3: "Turn Left Large", 4: "Turn Right Large", 5: "Look Up",
    6: "Look Down", 7: "Zoom In", 8: "Zoom Out"
}

def main():
    # 读取你的 GPT 生成的数据
    # 请确保这里的文件名和你之前生成的一致 (train.csv 或 train_gpt.csv)
    df = pd.read_csv("dataset_train/train.csv") 
    img_root = os.path.abspath("dataset_train/images") # 必须是绝对路径
    
    output_data = []
    
    print(f"Converting {len(df)} samples to ShareGPT format...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_path = os.path.join(img_root, row['image_id'])
        
        if not os.path.exists(img_path):
            continue
            
        action_str = ACTION_MAP.get(row['label'], "Stop")
        
        # === 关键修改 ===
        # LLaMA-Factory ShareGPT 格式要求:
        # 1. 键名必须是 "from" 和 "value"
        # 2. 角色通常建议用 "human" 和 "gpt"
        output_data.append({
            "images": [img_path],
            "messages": [
                {
                    "from": "human",  # 原 role: user -> 改为 human
                    # Qwen2-VL 模板需要 <image> 占位符
                    "value": f"<image>Instruction: {row['instruction']}" # 原 content -> 改为 value
                },
                {
                    "from": "gpt",    # 原 role: assistant -> 改为 gpt
                    "value": action_str
                }
            ]
        })

    # 保存
    with open("geoguessr_sft.json", "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"Saved {len(output_data)} samples to geoguessr_sft.json")

if __name__ == "__main__":
    main()