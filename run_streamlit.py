"""
简化的Streamlit启动脚本
"""
import os
import sys
import subprocess

def main():
    """启动Streamlit应用"""
    # 获取项目根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = script_dir

    # 设置环境变量
    env = os.environ.copy()
    env['PYTHONPATH'] = project_root
    env['STREAMLIT_SERVER_PORT'] = '8501'
    env['STREAMLIT_SERVER_ADDRESS'] = '0.0.0.0'

    # 优先使用简化版本，避免导入问题
    app_path = os.path.join(project_root, 'simple_app.py')

    # 如果简化版本不存在，则尝试完整版本
    if not os.path.exists(app_path):
        app_path = os.path.join(project_root, 'ui', 'app.py')

    if not os.path.exists(app_path):
        print(f"错误: 找不到应用文件 {app_path}")
        return 1

    print(f"启动FactorHub应用...")
    print(f"项目目录: {project_root}")
    print(f"应用文件: {app_path}")
    print(f"访问地址: http://0.0.0.0:8501")
    print("按 Ctrl+C 停止应用")
    print("-" * 50)

    # 检查Python模块
    try:
        import streamlit
        print(f"✓ Streamlit 版本: {streamlit.__version__}")
    except ImportError:
        print("✗ 错误: Streamlit 未安装，请运行: pip install streamlit")
        return 1

    try:
        import pandas as pd
        print(f"✓ Pandas 版本: {pd.__version__}")
    except ImportError:
        print("✗ 错误: Pandas 未安装，请运行: pip install pandas")
        return 1

    print("-" * 50)

    try:
        # 启动Streamlit
        cmd = [
            sys.executable, '-m', 'streamlit', 'run', app_path,
            '--server.port=8501',
            '--server.address=0.0.0.0',
            '--server.headless=true'
        ]

        process = subprocess.Popen(cmd, env=env, cwd=project_root)
        process.wait()

    except KeyboardInterrupt:
        print("\n应用已停止")
        return 0
    except Exception as e:
        print(f"启动应用时发生错误: {e}")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())