import pandas as pd
import streetlevel.streetview as streetview
from PIL import Image
import os
import json
import time
from tqdm import tqdm

# ================= 配置区域 =================
INPUT_CSV = "dataset/coords.csv"          # 输入文件，需包含 lat, lon 列
OUTPUT_DIR = "dataset_panos"        # 输出总目录
Meta_DIR = os.path.join(OUTPUT_DIR, "meta")
IMG_DIR = os.path.join(OUTPUT_DIR, "images")
ZOOM_LEVEL = 3                    # 3级缩放约 3328x1664，足够 CLIP 使用且下载快
SEARCH_RADIUS = 50                # 在坐标附近 50米内搜索最近的全景图
# ===========================================

def setup_directories():
    if not os.path.exists(Meta_DIR):
        os.makedirs(Meta_DIR)
    if not os.path.exists(IMG_DIR):
        os.makedirs(IMG_DIR)

def save_metadata(pano):
    """
    保存元数据，这是生成 Discrete Labels 的核心。
    我们需要知道：这张图的北在哪 (Heading)，以及它连接着哪些路口 (Links)。
    """
    
    # 提取“邻居节点”信息，用于训练 Path Following
    neighbors = []
    if pano.links:
        for link in pano.links:
            # print(link)
            neighbors.append({
                "pano_id": link.pano.id,
                "heading": link.direction,  # 关键：通往这个邻居的角度（Ground Truth）
                "description": getattr(link, 'text', '') # 有时会有路名
            })

    meta_data = {
        "pano_id": pano.id,
        "lat": pano.lat,
        "lon": pano.lon,
        "north_heading": pano.heading, # 图片本身的正北方向偏移量
        "date": str(pano.date),
        "links": neighbors             # 拓扑连接图
    }

    # 保存为 JSON
    file_path = os.path.join(Meta_DIR, f"{pano.id}.json")
    with open(file_path, 'w') as f:
        json.dump(meta_data, f, indent=4)
    
    return True

def download_image(pano):
    """
    下载全景图并拼接。
    streetlevel 库会自动处理 tiles 的下载和拼接。
    """
    file_path = os.path.join(IMG_DIR, f"{pano.id}.jpg")
    
    # 如果已经存在，跳过（断点续传）
    if os.path.exists(file_path):
        return False

    try:
        # 下载全景图 (Equirectangular)
        image = streetview.get_panorama(pano, zoom=ZOOM_LEVEL)
        image.save(file_path, "jpeg", quality=90)
        return True
    except Exception as e:
        print(f"\n[Error] Failed to download image for {pano.id}: {e}")
        return False

def main():
    setup_directories()
    
    # 1. 读取 CSV
    # 假设 CSV 有 'lat' 和 'lon' 两列
    try:
        df = pd.read_csv(INPUT_CSV)
        print(f"Loaded {len(df)} coordinates from {INPUT_CSV}")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # 用于去重 (不同坐标可能指向同一个最近的全景图)
    processed_ids = set()
    
    print(f"Already collected {len(processed_ids)} Panoramas. Starting collection...")
    existing_files = [f.split('.')[0] for f in os.listdir(Meta_DIR) if f.endswith('.json')]
    processed_ids.update(existing_files)

    n = 10000

    # 2. 遍历坐标
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):

        if index >= n:
            break

        lat, lon = row[0], row[1]
        
        try:
            # A. 搜索最近的全景图
            pano = streetview.find_panorama(lat, lon, radius=SEARCH_RADIUS)
                
            if pano.id in processed_ids:
                # 这是一个已知的路口，跳过
                continue

            # B. 保存元数据 (Graph Oracle 的基础)
            save_metadata(pano)
            
            # C. 下载图片 (Visual Backbone 的基础)
            download_image(pano)
            
            # 记录已处理
            processed_ids.add(pano.id)
            
            # 礼貌性延时，防止被 streetview 封 IP
            time.sleep(1.0)

        except Exception as e:
            print(f"\n[Error] Processing row {index}: {e}")
            continue

    print("\nCollection Complete!")
    print(f"Total Panoramas: {len(processed_ids)}")
    print(f"Data saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()