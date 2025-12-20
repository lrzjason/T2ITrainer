from setuptools import setup
from Cython.Build import cythonize
import numpy  # 如果用到 numpy，否则可删

setup(
    ext_modules=cythonize(
        "train_qwen_image_edit.py",
        compiler_directives={
            'language_level': 3,
            'boundscheck': False,      # 关闭边界检查（提速+混淆）
            'wraparound': False,       # 关闭负索引支持
            'initializedcheck': False, # 关闭内存初始化检查
            'nonecheck': False,        # 关闭 None 检查
            'overflowcheck': False,
            'cdivision': True,         # 快速除法
            'infer_types': True,       # 类型推断，减少 Python 对象使用
        },
        annotate=False,                # 不生成 .html 注释文件（避免泄露逻辑）
        exclude_failures=True,
    ),
    zip_safe=False,
)

# python setup.py build_ext --inplace