import os
import random
import shutil
from glob import glob
from tqdm import tqdm

def split_test_to_val(data_root, val_count=2000):
    # 1. 定义路径
    test_xyz_dir = os.path.join(data_root, 'train/xyz')
    test_wf_dir = os.path.join(data_root, 'train/wireframe')
    
    val_xyz_dir = os.path.join(data_root, 'validation/xyz')
    val_wf_dir = os.path.join(data_root, 'validation/wireframe')

    # 2. 创建目标文件夹
    os.makedirs(val_xyz_dir, exist_ok=True)
    os.makedirs(val_wf_dir, exist_ok=True)

    # 3. 获取 test 中所有的 xyz 文件
    all_test_files = glob(os.path.join(test_xyz_dir, '*.xyz'))
    
    if len(all_test_files) < val_count:
        print(f"错误：测试集文件总数({len(all_test_files)})少于要求的验证集数量")
        return

    # 4. 随机打乱并选取
    random.seed(42) # 固定随机种子，保证结果可复现
    selected_xyz_files = random.sample(all_test_files, val_count)

    print(f"正在移动 {val_count} 个文件到 validation 目录...")

    # 5. 执行移动
    move_count = 0
    for xyz_path in tqdm(selected_xyz_files):
        file_name = os.path.basename(xyz_path)
        base_name = file_name.replace('.xyz', '')
        
        # 对应 wireframe 的路径 (假设后缀是 .obj)
        wf_name = base_name + '.obj'
        wf_path = os.path.join(test_wf_dir, wf_name)
        
        # 目标路径
        dst_xyz = os.path.join(val_xyz_dir, file_name)
        dst_wf = os.path.join(val_wf_dir, wf_name)

        # 核心逻辑：只有当线框文件也存在时才移动 (因为 validation 必须有标签)
        if os.path.exists(wf_path):
            shutil.move(xyz_path, dst_xyz)
            shutil.move(wf_path, dst_wf)
            move_count += 1
        else:
            # 如果对应的 wireframe 不存在，则不移动，继续找下一个
            continue

    print(f"划分完成！成功移动了 {move_count} 对文件到 validation 目录。")

if __name__ == '__main__':
    # 修改为你的 Tallinn 路径
    my_data_root = "/geogfs1/groups/hkurs/u3666068mgh/Tallinn"
    # 你想分出多少个作为验证集？建议 2000
    split_test_to_val(my_data_root, val_count=2000)