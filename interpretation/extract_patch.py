

import os
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import openslide

# =========================
# 1. 配置区
# =========================
WSI_ROOTDIR = " "
IG_ROOTDIR = " "
CSV_ROOTDIR = " "

CASES = [
    {
        "case_id": "123",
        "ig_npy": " ",
        "wsi_path": " ",
        "superpatch_csv": "123_0.75_4.3_artifact_sophis_final.csv",
        "location_csv": "123_node_location_list.csv",
    },
    {
        "case_id": "123",
        "ig_npy": " ",
        "wsi_path": " ",
        "superpatch_csv": "123_0.75_4.3_artifact_sophis_final.csv",
        "location_csv": "123_node_location_list.csv",
    },
    {
        "case_id": "123",
        "ig_npy": " ",
        "wsi_path": " ",
        "superpatch_csv": "123_0.75_4.3_artifact_sophis_final.csv",
        "location_csv": "123_node_location_list.csv",
    },
]

OUT_DIR = r" "
TOPK = 100                 # 每个区域（tumor/normal）分别取 TOPK 张
WSI_LEVEL = 1
RISKS = ["high", "low"]


# ========================= 2. 工具函数 =========================
def ensure_dir(path):
    """递归创建目录，存在时也不报错"""
    os.makedirs(path, exist_ok=True)

def load_ig_scores(npy_path):
    """加载 IG 分数（一维数组）"""
    arr = np.load(npy_path, allow_pickle=True)
    arr = np.asarray(arr).squeeze()
    if arr.ndim != 1:
        raise ValueError(f"IG npy 需要是一维数组，当前 shape={arr.shape}")
    return arr.astype(float)

def point_in_polygon(px, py, poly):
    """射线法判断点 (px,py) 是否在多边形 poly 内部（poly 为 [(x,y), ...]）"""
    n = len(poly)
    inside = False
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        # 仅当射线穿过边时改变状态
        if ((y1 > py) != (y2 > py)) and (px < (x2 - x1) * (py - y1) / (y2 - y1) + x1):
            inside = not inside
    return inside

def load_xml_polygons(xml_path):
    """
    解析 ASAP 风格 XML，提取每个 Region 的多边形顶点（Level 0 像素坐标）
    返回多边形列表，每个多边形为 [(x1,y1), (x2,y2), ...]
    """
    if not os.path.isfile(xml_path):
        return []
    tree = ET.parse(xml_path)
    root = tree.getroot()
    polygons = []
    # 遍历所有 Annotation 下的 Region
    for annotation in root.iter('Annotation'):
        for region in annotation.findall('Regions/Region'):
            vertices = []
            for vertex in region.findall('Vertices/Vertex'):
                x = float(vertex.get('X'))
                y = float(vertex.get('Y'))
                vertices.append((x, y))
            if len(vertices) >= 3:
                polygons.append(vertices)
    return polygons

def classify_superpatches(superpatch_df, location_df, xml_polygons):
    """
    根据 superpatch 中心点是否在任一肿瘤多边形内，返回每个 superpatch 的区域标签
    """
    n = superpatch_df.shape[0]
    regions = []
    # 预计算所有子节点的 Level 0 坐标（X/Y 为 tile 索引，乘 256）
    node_x = np.array(location_df["X"]) * 256
    node_y = np.array(location_df["Y"]) * 256

    for idx in range(n):
        row = superpatch_df.iloc[idx]
        subnodes = row.iloc[2:].dropna().values.astype(float).astype(int)
        if len(subnodes) == 0:
            regions.append("normal")
            continue

        center_x = np.mean(node_x[subnodes])
        center_y = np.mean(node_y[subnodes])

        in_tumor = any(point_in_polygon(center_x, center_y, poly) for poly in xml_polygons)
        regions.append("tumor" if in_tumor else "normal")
    return regions

def select_indices_by_risk(ig_scores, risk, topk=100):
    """
    按风险选择索引：high 取最高分，low 取最低分（忽略 NaN/Inf）
    返回 (全局索引, 分数)
    """
    valid_mask = np.isfinite(ig_scores)
    valid_idx = np.where(valid_mask)[0]
    valid_scores = ig_scores[valid_mask]

    if len(valid_idx) == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    if risk == "high":
        order = np.argsort(-valid_scores)   # 降序
    elif risk == "low":
        order = np.argsort(valid_scores)    # 升序
    else:
        raise ValueError(f"risk 必须为 'high' 或 'low'，当前={risk}")

    picked = order[: min(topk, len(order))]
    picked_idx = valid_idx[picked]
    picked_scores = valid_scores[picked]
    return picked_idx, picked_scores

def crop_superpatch(wsi, wsi_level, superpatch_node, superpatch_df, location_df):
    """
    从 WSI 中裁剪指定 superpatch 对应的图像区域
    参数：
        wsi: OpenSlide 对象
        wsi_level: 读取级别
        superpatch_node: superpatch 的索引（在 superpatch_df 中的行号）
        superpatch_df: 已读取的 superpatch DataFrame
        location_df: 已读取的 node location DataFrame
    返回：
        img: PIL Image
        (min_x, min_y, max_x, max_y): tile 索引边界
    """
    if superpatch_node >= superpatch_df.shape[0]:
        raise IndexError(f"superpatch_node={superpatch_node} 超出范围")

    # 提取子节点列表（忽略 NaN）
    subnodes = list(superpatch_df.iloc[int(superpatch_node)].iloc[2:].dropna())
    subnodes = [int(float(x)) for x in subnodes]
    if len(subnodes) == 0:
        raise ValueError(f"superpatch_node={superpatch_node} 没有子节点")

    subnode_loc = location_df.iloc[subnodes]
    min_x = int(np.min(subnode_loc["X"]))
    min_y = int(np.min(subnode_loc["Y"]))
    max_x = int(np.max(subnode_loc["X"]))
    max_y = int(np.max(subnode_loc["Y"]))

    downsample = int(wsi.level_downsamples[wsi_level])
    patch_dim = int(256 / downsample)  # 每个 tile 在目标级别上的像素尺寸
    width = int(max_x - min_x + 1) * patch_dim
    height = int(max_y - min_y + 1) * patch_dim

    # Level 0 起始像素坐标
    start_x = min_x * 256
    start_y = min_y * 256

    # 检查裁剪区域是否在图像内
    level_w, level_h = wsi.level_dimensions[wsi_level]
    if (start_x < 0 or start_y < 0 or
        start_x + width > level_w * downsample or
        start_y + height > level_h * downsample):
        raise ValueError(f"裁剪区域超出图像边界: start=({start_x},{start_y}), size=({width},{height})")

    img = wsi.read_region(
        (start_x, start_y),
        wsi_level,
        (width, height)
    ).convert("RGB")

    return img, (min_x, min_y, max_x, max_y)

# ========================= 3. 主处理函数 =========================
def export_case(case_cfg, out_dir, topk=100, wsi_level=1, risks=["high", "low"]):
    case_id = case_cfg["case_id"]
    ig_npy = case_cfg["ig_npy"]
    wsi_path = case_cfg["wsi_path"]
    superpatch_csv = case_cfg["superpatch_csv"]
    location_csv = case_cfg["location_csv"]

    # 检查文件存在性
    for p in [ig_npy, wsi_path, superpatch_csv, location_csv]:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"文件不存在: {p}")

    print(f"\n[INFO] 处理病例: {case_id}")

    # 读取数据（仅一次）
    ig_scores = load_ig_scores(ig_npy)
    superpatch_df = pd.read_csv(superpatch_csv)
    location_df = pd.read_csv(location_csv)

    if len(ig_scores) != superpatch_df.shape[0]:
        raise ValueError(
            f"{case_id}: IG 数组长度 ({len(ig_scores)}) 与 superpatch 行数 ({superpatch_df.shape[0]}) 不一致"
        )

    # 读取 XML 标注（与 SVS 同名）
    xml_path = os.path.splitext(wsi_path)[0] + ".xml"
    xml_polygons = load_xml_polygons(xml_path)
    if not xml_polygons:
        print(f"[WARN] 未找到有效标注多边形 {xml_path}，所有区域将视为 'normal'")
    else:
        print(f"[INFO] 载入 {len(xml_polygons)} 个肿瘤多边形")

    # 分类所有 superpatch
    regions = classify_superpatches(superpatch_df, location_df, xml_polygons)
    tumor_indices = [i for i, r in enumerate(regions) if r == "tumor"]
    normal_indices = [i for i, r in enumerate(regions) if r == "normal"]
    print(f"[INFO] tumor 区域 {len(tumor_indices)} 个 superpatch, normal 区域 {len(normal_indices)} 个")

    # 打开 WSI（仅一次）
    wsi = openslide.open_slide(wsi_path)

    for region_name, region_idx_list in [("tumor", tumor_indices), ("normal", normal_indices)]:
        if len(region_idx_list) == 0:
            print(f"[SKIP] {case_id} 的 {region_name} 区域无 superpatch，跳过")
            continue

        region_ig_scores = ig_scores[region_idx_list]

        for risk in risks:
            sub_idx, sub_scores = select_indices_by_risk(region_ig_scores, risk, topk=topk)
            selected_full_idx = np.array(region_idx_list)[sub_idx]   # 映射回全局索引

            # 输出子目录
            out_subdir = os.path.join(out_dir, case_id, region_name, risk)
            ensure_dir(out_subdir)

            print(f"[INFO] {case_id} | {region_name} | {risk} : 选中 {len(selected_full_idx)} 个 patch")

            # 保存排名 CSV
            rank_path = os.path.join(out_subdir, f"{case_id}_{region_name}_{risk}_selected_patch_rank.csv")
            try:
                rank_df = pd.DataFrame({
                    "rank": np.arange(1, len(selected_full_idx) + 1),
                    "superpatch_node": selected_full_idx,
                    "ig_score": sub_scores
                })
                rank_df.to_csv(rank_path, index=False)
            except Exception as e:
                print(f"[ERROR] 保存排名 CSV 失败: {rank_path}, err={e}")
                continue

            success = 0
            failed = 0
            records = []

            for rank, (sp_idx, score) in enumerate(zip(selected_full_idx, sub_scores), start=1):
                try:
                    img, (min_x, min_y, max_x, max_y) = crop_superpatch(
                        wsi=wsi,
                        wsi_level=wsi_level,
                        superpatch_node=int(sp_idx),
                        superpatch_df=superpatch_df,
                        location_df=location_df
                    )
                    # 命名：左上角像素坐标 + IG 分数
                    pixel_x = int(min_x * 256)
                    pixel_y = int(min_y * 256)
                    save_name = f"x{pixel_x}_y{pixel_y}_ig{score:.4f}.png"
                    save_path = os.path.join(out_subdir, save_name)
                    img.save(save_path)

                    records.append({
                        "rank": rank,
                        "superpatch_node": int(sp_idx),
                        "ig_score": float(score),
                        "pixel_x": pixel_x,
                        "pixel_y": pixel_y,
                        "min_tile_x": min_x,
                        "min_tile_y": min_y,
                        "max_tile_x": max_x,
                        "max_tile_y": max_y,
                        "save_path": save_path
                    })
                    success += 1
                except Exception as e:
                    print(f"[WARN] 裁图失败: case={case_id}, region={region_name}, risk={risk}, "
                          f"sp={sp_idx}, err={e}")
                    failed += 1

            # 保存元数据 CSV
            if records:
                meta_path = os.path.join(out_subdir, f"{case_id}_{region_name}_{risk}_patch_metadata.csv")
                try:
                    meta_df = pd.DataFrame(records)
                    meta_df.to_csv(meta_path, index=False)
                except Exception as e:
                    print(f"[ERROR] 保存元数据 CSV 失败: {meta_path}, err={e}")

            print(f"[INFO] 完成: {case_id} | {region_name} | {risk} | 成功={success} | 失败={failed}")

    wsi.close()

def main():
    ensure_dir(OUT_DIR)
    for case_cfg in CASES:
        try:
            export_case(case_cfg, OUT_DIR, topk=TOPK, wsi_level=WSI_LEVEL, risks=RISKS)
        except Exception as e:
            print(f"[FATAL] 处理病例 {case_cfg['case_id']} 时发生致命错误: {e}")

if __name__ == "__main__":
    main()