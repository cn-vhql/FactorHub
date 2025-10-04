"""
修复导入问题的脚本
"""
import os
import sys

def setup_path():
    """设置Python路径"""
    # 获取项目根目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    # 添加到Python路径
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    print(f"项目根目录: {project_root}")
    print(f"Python路径: {sys.path[:3]}...")

def test_imports():
    """测试关键模块导入"""
    setup_path()

    modules_to_test = [
        'utils.config',
        'utils.logger',
        'data_processor',
        'factor_manager',
        'analyzer',
        'backtester',
        'factor_miner'
    ]

    print("\n测试模块导入:")
    print("-" * 40)

    for module_name in modules_to_test:
        try:
            __import__(module_name)
            print(f"✓ {module_name}")
        except ImportError as e:
            print(f"✗ {module_name}: {e}")

if __name__ == "__main__":
    test_imports()