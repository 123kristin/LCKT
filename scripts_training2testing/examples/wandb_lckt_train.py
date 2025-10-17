import os
import sys
import argparse

# 先解析参数，设置环境变量
parser = argparse.ArgumentParser()
# dataset config
parser.add_argument("--dataset_name", type=str, default="XES3G5M")
parser.add_argument("--fold", type=int, default=0)

# train config
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--seed", type=int, default=2025)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--num_epochs", type=int, default=200)

# log config & save config
parser.add_argument("--use_wandb", type=int, default=0)
parser.add_argument("--add_uuid", type=int, default=0)
parser.add_argument("--save_dir", type=str, default="saved_model")

# model config
parser.add_argument("--model_name", type=str, default="lckt")
parser.add_argument("--emb_type", type=str, default='qkcs')

parser.add_argument("--dropout", type=float, default=0.5)
parser.add_argument("--dim_qc", type=int, default=200, help="dimension of q and c embedding")

# 内容嵌入相关参数 - 新增
parser.add_argument("--use_content_emb", type=int, default=1, 
                   help="是否使用内容嵌入: 0=不使用, 1=使用")
parser.add_argument("--content_type", type=str, default="text", 
                   choices=["text", "image"],
                   help="选择内容嵌入类型: text使用文本内容嵌入, image使用图像内容嵌入")
parser.add_argument("--content_dim", type=int, default=1536,
                   help="内容嵌入维度")
parser.add_argument("--trainable_content_emb", type=int, default=1,
                   help="内容嵌入是否可训练: 0=固定, 1=可训练")

# KC嵌入相关参数 - 新增
parser.add_argument("--use_kc_emb", type=int, default=1,
                   help="是否使用KC嵌入: 0=不使用, 1=使用")
parser.add_argument("--kc_dim", type=int, default=1600,
                   help="KC嵌入原始维度")
parser.add_argument("--trainable_kc_emb", type=int, default=1,
                   help="KC嵌入是否可训练: 0=固定, 1=可训练")

# 解析嵌入相关参数 - 新增
parser.add_argument("--use_analysis_emb", type=int, default=1,
                   help="是否使用解析嵌入: 0=不使用, 1=使用")
parser.add_argument("--analysis_type", type=str, default="generated",
                   choices=["generated", "original"],
                   help="选择解析嵌入类型: generated使用生成解析, original使用原始解析")
parser.add_argument("--analysis_dim", type=int, default=1536,
                   help="解析嵌入维度")
parser.add_argument("--trainable_analysis_emb", type=int, default=1,
                   help="解析嵌入是否可训练: 0=固定, 1=可训练")

parser.add_argument("--analysis_contrastive", type=int, default=1,
                   help="解析嵌入是否参与对比学习: 0=不参与, 1=参与")

# 对比学习相关参数 - 新增
parser.add_argument("--contrastive_weight", type=float, default=0.1,
                   help="对比学习损失权重")

# 难度对比学习参数 - 新增
parser.add_argument("--use_difficulty_contrastive", type=int, default=1,
                   help="是否使用基于难度的对比学习: 0=不使用, 1=使用")
parser.add_argument("--difficulty_contrastive_weight", type=float, default=0.05,
                   help="难度对比学习损失权重")
parser.add_argument("--difficulty_proj_dim", type=int, default=64,
                   help="难度对比投影维度")
parser.add_argument("--difficulty_temperature", type=float, default=0.1,
                   help="难度对比温度系数")

# GPU选择参数 - 新增
parser.add_argument("--gpu_id", type=str, default="0",
                   help="指定使用的GPU ID，如'0','1','2'等")

args = parser.parse_args()

# 设置环境变量 - 只设置CURRENT_GPU_ID，不设置CUDA_VISIBLE_DEVICES
os.environ['CURRENT_GPU_ID'] = args.gpu_id

# 现在才可以导入 torch 及其依赖
from wandb_train import main
from utils4running import Tee

if __name__ == "__main__":
    params = vars(args)
    
    print(f"实验配置:")
    print(f"  数据集: {args.dataset_name}")
    print(f"  模型名称: {args.model_name}")
    print(f"  使用GPU: cuda:{args.gpu_id}")
    print(f"  学习率: {args.learning_rate}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  训练轮数: {args.num_epochs}")
    print(f"  使用内容嵌入: {args.use_content_emb}")
    if args.use_content_emb:
        print(f"  内容类型: {args.content_type}")
        print(f"  内容嵌入维度: {args.content_dim}")
        print(f"  内容嵌入可训练: {args.trainable_content_emb}")
    print(f"  使用解析嵌入: {args.use_analysis_emb}")
    if args.use_analysis_emb:
        print(f"  解析类型: {args.analysis_type}")
        print(f"  解析嵌入维度: {args.analysis_dim}")
        print(f"  解析嵌入可训练: {args.trainable_analysis_emb}")
        # print(f"  解析嵌入参与主任务: {args.analysis_main_task}")  # 移除这行，解析嵌入不参与主任务
        print(f"  解析嵌入参与对比学习: {args.analysis_contrastive}")
    print(f"  对比学习权重: {args.contrastive_weight}")
    print(f"  使用难度对比学习: {args.use_difficulty_contrastive}")
    if args.use_difficulty_contrastive:
        print(f"  难度对比权重: {args.difficulty_contrastive_weight}")
        print(f"  难度对比投影维度: {args.difficulty_proj_dim}")
        print(f"  难度对比温度: {args.difficulty_temperature}")
    print(f"  使用KC嵌入: {args.use_kc_emb}")
    if args.use_kc_emb:
        print(f"  KC嵌入维度: {args.kc_dim}")
        print(f"  KC嵌入可训练: {args.trainable_kc_emb}")
    
    with Tee(f"{args.save_dir}/training_log/{args.model_name}_training.log"):
        main(params)
