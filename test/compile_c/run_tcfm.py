# run.py
import sys
import train_qwen_image_edit  # 导入编译后的 .so 模块

if __name__ == '__main__':
    # 直接调用 main 模块中的 main() 函数
    # 它内部会自动使用 sys.argv（和原来行为一致）
    args, config_args = train_qwen_image_edit.parse_args()
    train_qwen_image_edit.main(args, config_args)