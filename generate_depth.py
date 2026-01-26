import os
from PIL import Image
from transformers import pipeline


def load_midas_model():
    """
    加载Depth-Anything-V2-Small深度估计模型
    
    Returns:
        model: 加载好的Depth-Anything-V2-Small模型
        transform: 图像预处理变换
    """
    
    # 加载模型
    pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
    
    return pipe


def generate_depth_map(pipe, img_path):
    """
    为单张图像生成深度图
    
    Args:
        pipe: Depth-Anything-V2-Small模型管道
        img_path: 输入图像路径
    
    Returns:
        depth_map: 生成的深度图
    """
    # 加载图像
    image = Image.open(img_path)
    
    
    # 生成深度图
    depth = pipe(image)["depth"]
    
    return depth


def process_agd20k_dataset(agd20k_root, output_root):
    """
    处理AGD20K数据集，为所有图像生成深度图
    
    Args:
        agd20k_root: AGD20K数据集根目录
        output_root: 深度图输出根目录
    """
    # 加载Depth-Anything-V2-Small模型
    print("Loading Depth-Anything-V2-Small model...")
    pipe = load_midas_model()
    print("Model loaded successfully!")
    
    # 遍历数据集目录结构
    divide_dirs = ["Seen", "Unseen"]
    
    for divide in divide_dirs:
        divide_path = os.path.join(agd20k_root, divide)
        if not os.path.exists(divide_path):
            continue
        
        # 处理训练集和测试集
        for set_type in ["trainset", "testset"]:
            set_path = os.path.join(divide_path, set_type)
            if not os.path.exists(set_path):
                continue
            
            # 处理ego-centric和exo-centric图像
            for view in ["egocentric", "exocentric"]:
                view_path = os.path.join(set_path, view)
                if not os.path.exists(view_path):
                    continue
                
                # 遍历每个动作类别
                action_dirs = os.listdir(view_path)
                for action in action_dirs:
                    action_path = os.path.join(view_path, action)
                    if not os.path.isdir(action_path):
                        continue
                    
                    # 遍历每个物体类别
                    object_dirs = os.listdir(action_path)
                    for obj in object_dirs:
                        object_path = os.path.join(action_path, obj)
                        if not os.path.isdir(object_path):
                            continue
                        
                        # 创建对应的输出目录
                        output_dir = os.path.join(output_root, divide, set_type, view, action, obj)
                        os.makedirs(output_dir, exist_ok=True)
                        
                        # 遍历每个图像
                        images = os.listdir(object_path)
                        for img_name in images:
                            if img_name.endswith(('.jpg', '.jpeg', '.png')):
                                img_path = os.path.join(object_path, img_name)
                                
                                # 生成深度图
                                try:
                                    depth = generate_depth_map(pipe, img_path)
                                    
                                    # 保存深度图
                                    depth_name = os.path.splitext(img_name)[0] + '_depth.png'
                                    depth_path = os.path.join(output_dir, depth_name)
                                    depth.save(depth_path)
                                    
                                    print(f"Generated depth map for: {img_path} -> {depth_path}")
                                except Exception as e:
                                    print(f"Error processing {img_path}: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--agd20k_root', type=str, default='/path/to/AGD20K', help='AGD20K dataset root directory')
    parser.add_argument('--output_root', type=str, default='/path/to/AGD20K-Depth', help='Output directory for depth maps')
    args = parser.parse_args()
    
    # 处理数据集
    process_agd20k_dataset(args.agd20k_root, args.output_root)
