import numpy as np
import os
from scipy import spatial
# 导入你的类
from noise_gen_patch_straight import GenPatchTallinn # 请确保文件名正确

def debug_diagnostic(file_path):
    print(f"\n{'='*20} 开始诊断 {'='*20}")
    print(f"文件: {file_path}")
    
    # 1. 模拟初始化（不执行完整逻辑，只拿数据）
    # 这里直接手动模拟你的类内部关键步骤
    data_root = "/geogfs1/groups/hkurs/u3666068mgh/Tallinn"
    pc = np.loadtxt(file_path)
    print(f"原始点云点数: {pc.shape[0]}")
    
    # 模拟你的 down_sample
    import MinkowskiEngine.utils as ME_utils
    coords = np.floor(pc / 0.0075)
    inds = ME_utils.sparse_quantize(coords, return_index=True)
    pc_down = pc[inds]
    print(f"下采样后点数: {pc_down.shape[0]}")
    
    # 模拟 build_graph 关键件
    nbrs = spatial.cKDTree(pc_down)
    
    # 2. 模拟训练集种子点产生（这里以 vert_gt 为例，因为它是最常见的报错源）
    # 手动读取一次 GT
    gt_path = file_path.replace('/xyz/', '/gt/').replace('.xyz', '.obj')
    vert_gt = []
    if os.path.exists(gt_path):
        with open(gt_path, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    vert_gt.append([float(x) for x in line.strip().split()[1:4]])
    seed_vert = np.array(vert_gt)
    print(f"GT 顶点(种子)数量: {len(seed_vert)}")

    if len(seed_vert) == 0:
        print("警告: 该文件没有 GT 顶点，跳过顶点 Patch 测试")
    else:
        # 3. 核心诊断：模拟 gen_patch 内部的 query
        print("\n--- 执行 gen_patch 内部逻辑诊断 ---")
        k_val = 10
        dist, seed_idx = nbrs.query(seed_vert, k=k_val)
        
        print(f"dist 形状: {dist.shape}, dtype: {dist.dtype}")
        print(f"seed_idx 形状: {seed_idx.shape}, dtype: {seed_idx.dtype}")

        # 尝试复现 i=0 的操作
        try:
            i = 0
            d_i = dist[i]
            idx_i = seed_idx[i]
            mask = d_i < 0.05
            
            print(f"第 {i} 个点的 dist[i] 形状: {d_i.shape}")
            print(f"第 {i} 个点的 seed_idx[i] 形状: {idx_i.shape}")
            print(f"掩码 mask 形状: {mask.shape}")
            
            # 这里的报错就是你遇到的
            res = idx_i[mask]
            print("结果：索引操作成功！")
            
        except Exception as e:
            print(f"\n[!!!] 复现错误!")
            print(f"错误类型: {type(e).__name__}")
            print(f"错误信息: {e}")
            
            if "broadcast" in str(e):
                print("\n原因分析：")
                print(f"你的 seed_idx[{i}] 维度和 mask 维度不匹配。")
                if seed_vert.shape[0] == 1:
                    print("检测到种子点只有一个，导致 scipy.query 返回了 1D 数组而非 2D，触发了维度塌陷。")
                elif dist.dtype == 'O':
                    print("检测到 dist 是 Object 类型，说明不同点的邻居数量不一致（点云太稀疏）。")

if __name__ == '__main__':
    # 填入你报错的具体文件
    test_file = "/geogfs1/groups/hkurs/u3666068mgh/Tallinn/noise_sigma0.01clip0.01/train/xyz/22096.xyz"
    debug_diagnostic(test_file)