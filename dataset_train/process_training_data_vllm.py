import os
import json
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image

from vllm import LLM, SamplingParams
from transformers import AutoProcessor

# === 配置 ===
IMG_DIR = "dataset_panos/images"
OUTPUT_CSV = "dataset_train/train.csv"
# 确保你的模型路径正确，如果是 Qwen2.5-VL 或 Qwen2-VL，通常建议用官方名称或检查路径
MODEL_PATH = "models/Qwen3-VL-30B-A3B-Instruct" 
FOV = 90
SLICE_W, SLICE_H = 512, 512

# 系统 Prompt
SYSTEM_PROMPT = """
You are an expert navigation data generator for an autonomous street-view agent.
Your task is to analyze the input image and generate valid training pairs mapping "User Instructions" to "Discrete Actions".

### Action Space Definition
The agent operates with the following discrete actions. You must choose the BEST action based on where the target is relative to the current view.

**0: Move Forward**
   - Condition: The target path/object is **CENTERED** in the view (approx. middle 20% of width).
   - Use cases: "Follow the road", "Go straight", "Head towards the [central object]".

**1: Turn Left Small** (approx. 15-30 degrees)
   - Condition: The target is **slightly left** of center.
   - Use cases: "Adjust left to follow the path", "Head towards the [object on left-center]".

**2: Turn Right Small** (approx. 15-30 degrees)
   - Condition: The target is **slightly right** of center.
   - Use cases: "Adjust right", "Go to the [object on right-center]".

**3: Turn Left Large** (approx. 45-90 degrees)
   - Condition: The target is near the **left edge** or off-screen left.
   - Use cases: "Turn left at the intersection", "Look at the building on the far left".

**4: Turn Right Large** (approx. 45-90 degrees)
   - Condition: The target is near the **right edge** or off-screen right.
   - Use cases: "Turn right", "Check the shop on the far right".

**5: Look Up**
   - Condition: The target is above the current line of sight.
   - Use cases: "Look up at the sign", "Check the top of the building".

**6: Look Down**
   - Condition: The target is on the ground/floor.
   - Use cases: "Look down at the road markings", "Check the sidewalk".

**7: Zoom In**
   - Condition: The target is centered but **far away/small**.
   - Use cases: "Zoom in on the sign", "Get a closer look at the distant car".

**8: Zoom Out**
   - Condition: The view is too zoomed in, or the user asks for context.
   - Use cases: "Zoom out", "Show me the surroundings", "Back to wide view".

---

### Generation Task
Look at the image. Generate **3 distinct scenarios** covering different types of Model C instructions:

1.  **Geometric Instruction:** Explicit directional commands (e.g., "Turn left", "Align with the road").
2.  **Semantic Instruction:** Landmark-based commands (e.g., "Go to the red brick building", "Find the Starbucks sign", "Head towards the white truck").
3.  **Atomic/Functional Instruction:** Camera adjustments (e.g., "Zoom in on the crosswalk", "Look up at the street light", "Zoom out to see the corner").

### Strict Rules
1.  **Identify the Target First:** Before generating the label, decide WHERE the target object/path is in the image.
    - If Target is Centered -> Label **0** or **7**.
    - If Target is Left-Center -> Label **1**.
    - If Target is Far Left -> Label **3**.
    - If Target is Right-Center -> Label **2**.
    - If Target is Far Right -> Label **4**.
2.  **No Ambiguity:** Do not assign "Move Forward" if the road is blocked or facing a wall. If facing a wall, force a Turn (3 or 4).
3.  **JSON Output:** Output ONLY a raw JSON list. No markdown blocks.

### Example Output
[
  {"instruction": "Follow the main road", "label": 0},
  {"instruction": "Turn towards the blue shop on the far right", "label": 4},
  {"instruction": "Zoom in to read the street sign", "label": 7}
]
"""

BATCH_SIZE_PANOS = 32

class RectilinearProjector:
    """
    预计算投影映射表，大幅加速切图过程。
    """
    def __init__(self, fov, height, width):
        self.fov = fov
        self.height = height
        self.width = width
        self.f = 0.5 * width / np.tan(0.5 * fov * np.pi / 180)
        self.cx = width / 2
        self.cy = height / 2
        
        # 预先生成基础网格
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        self.x_3d = (x - self.cx) / self.f
        self.y_3d = -(y - self.cy) / self.f
        self.z_3d = np.ones_like(self.x_3d)
        
        # 缓存每个角度的 map_x 和 map_y
        self.maps = {}

    def get_maps(self, yaw, pitch, h_pano, w_pano):
        key = (yaw, pitch, h_pano, w_pano)
        if key in self.maps:
            return self.maps[key]

        yaw_rad = np.radians(yaw)
        pitch_rad = np.radians(pitch)

        # 旋转矩阵逻辑
        x_rot = self.x_3d
        y_rot = self.y_3d * np.cos(pitch_rad) - self.z_3d * np.sin(pitch_rad)
        z_rot = self.y_3d * np.sin(pitch_rad) + self.z_3d * np.cos(pitch_rad)

        x_final = x_rot * np.cos(yaw_rad) + z_rot * np.sin(yaw_rad)
        y_final = y_rot
        z_final = -x_rot * np.sin(yaw_rad) + z_rot * np.cos(yaw_rad)

        lon = np.arctan2(x_final, z_final)
        lat = np.arctan2(y_final, np.sqrt(x_final**2 + z_final**2))

        u = (lon / (2 * np.pi) + 0.5) * w_pano
        v = (-lat / np.pi + 0.5) * h_pano
        
        map_x = u.astype(np.float32)
        map_y = v.astype(np.float32)
        
        self.maps[key] = (map_x, map_y)
        return map_x, map_y

def main():
    print(f"Initializing VLLM with model: {MODEL_PATH}...")
    
    # 1. 初始化 VLLM
    # max_model_len: 根据你的显存和需求调整，Qwen2-VL 支持很长，但设小点可以省显存
    # limit_mm_per_prompt: 限制每个 prompt 的图片数量
    llm = LLM(
        model=MODEL_PATH,
        trust_remote_code=True,
        tensor_parallel_size=1, # 单卡 H100 设为 1；如果是多卡比如 2张 4090，设为 2
        gpu_memory_utilization=0.95, # 吃满显存以最大化 Batch
        max_model_len=4096, 
        limit_mm_per_prompt={"image": 1}
    )

    # 采样参数：temperature=0 保证确定性
    sampling_params = SamplingParams(temperature=0.0, max_tokens=512)

    # 我们仍然需要 processor 来生成正确的 Prompt 文本格式 (添加 <|vision_start|> 等 token)
    # 注意：这里我们只用它来处理文本模板，不让它处理 tensors
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

    projector = RectilinearProjector(FOV, SLICE_H, SLICE_W)
    os.makedirs("dataset_train_simple/images", exist_ok=True)
    
    files = [f for f in os.listdir(IMG_DIR) if f.endswith(".jpg")]
    data_list = []
    
    print(f"Processing {len(files)} panoramas with Batch Size {BATCH_SIZE_PANOS}...")

    # 分块处理 (Chunking) 以利用 VLLM 的并发优势
    for i in tqdm(range(0, len(files), BATCH_SIZE_PANOS)):
        chunk_files = files[i : i + BATCH_SIZE_PANOS]
        
        inputs_batch = []     # 存放给 vllm 的输入对象
        metadata_batch = []   # 存放文件名，用于后续匹配结果

        # --- 步骤 A: 准备数据 (CPU密集) ---
        for img_name in chunk_files:
            img_path = os.path.join(IMG_DIR, img_name)
            pano_img = cv2.imread(img_path)
            if pano_img is None: continue
            
            h_pano, w_pano = pano_img.shape[:2]

            for yaw in [0, 90, 180, 270]:
                # 1. 切图
                map_x, map_y = projector.get_maps(yaw, 0, h_pano, w_pano)
                perspective = cv2.remap(pano_img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
                
                slice_filename = f"{img_name.split('.')[0]}_{yaw}.jpg"
                save_path = f"dataset_train_simple/images/{slice_filename}"
                cv2.imwrite(save_path, perspective)
                
                # 2. 转换为 PIL
                pil_img = Image.fromarray(cv2.cvtColor(perspective, cv2.COLOR_BGR2RGB))
                
                # 3. 构建 Prompt 文本
                # Qwen2-VL 的模板会自动插入 <|vision_start|>...
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": pil_img}, # 这里占位，为了生成含 image token 的文本
                            {"type": "text", "text": SYSTEM_PROMPT}
                        ],
                    }
                ]
                # 生成纯文本 Prompt，不要生成 tensors
                prompt_text = processor.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )

                # 4. 加入 VLLM 任务列表
                # VLLM 接收 prompt 文本 + multi_modal_data 字典
                inputs_batch.append({
                    "prompt": prompt_text,
                    "multi_modal_data": {"image": pil_img},
                })
                metadata_batch.append(slice_filename)

        if not inputs_batch:
            continue

        # --- 步骤 B: VLLM 批量推理 (GPU密集) ---
        # vllm 会自动处理 padding 和 continuous batching，速度极快
        outputs = llm.generate(inputs_batch, sampling_params)

        # --- 步骤 C: 解析结果 ---
        for j, output in enumerate(outputs):
            filename = metadata_batch[j]
            generated_text = output.outputs[0].text

            try:
                # 鲁棒的 JSON 提取
                json_str = generated_text.strip()
                if "```json" in json_str:
                    json_str = json_str.split("```json")[-1].split("```")[0]
                elif "```" in json_str: # 有时候它只写 ```
                    json_str = json_str.split("```")[1]
                
                # 尝试找到列表起止
                s = json_str.find("[")
                e = json_str.rfind("]")
                if s != -1 and e != -1:
                    json_str = json_str[s : e+1]

                scenarios = json.loads(json_str)
                
                if isinstance(scenarios, list):
                    for item in scenarios:
                        data_list.append({
                            "image_id": filename,
                            "instruction": item.get('instruction', ''),
                            "label": item.get('label', 0)
                        })
            except Exception as e:
                # print(f"Parse Error {filename}: {e}")
                # print(f"Output was: {generated_text}")
                pass

        # 可选：每批次保存一次，防止程序崩溃丢失数据
        if i % (BATCH_SIZE_PANOS * 5) == 0:
             pd.DataFrame(data_list).to_csv(OUTPUT_CSV, index=False)

    # 最后保存
    df = pd.DataFrame(data_list)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Done! Generated {len(df)} training pairs.")

if __name__ == "__main__":

    import multiprocessing
    # 强制将多进程启动方式设置为 spawn，解决 CUDA 初始化冲突
    multiprocessing.set_start_method('spawn', force=True)
    # === 新增修复代码结束 ===
    
    main()