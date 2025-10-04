#!/usr/bin/env python3
"""
FactorHub 启动脚本
"""
import os
import sys
import subprocess

def main():
    """启动FactorHub应用"""
    print("🚀 FactorHub 量化因子分析平台")
    print("=" * 50)

    # 检查当前目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)

    print(f"工作目录: {current_dir}")

    # 优先使用完整版应用
    full_app = os.path.join(current_dir, 'ui', 'app.py')
    simple_app = os.path.join(current_dir, 'simple_app.py')

    app_to_use = None
    if os.path.exists(full_app):
        app_to_use = full_app
        print("✓ 使用完整版FactorHub应用")
    elif os.path.exists(simple_app):
        app_to_use = simple_app
        print("✓ 使用简化版应用 (后备)")
    else:
        print("❌ 错误: 找不到应用文件")
        print("请确保 ui/app.py 或 simple_app.py 存在")
        return 1

    print(f"应用文件: {app_to_use}")
    print("-" * 50)

    # 启动命令
    cmd = [
        sys.executable, '-m', 'streamlit', 'run', app_to_use,
        '--server.port=8501',
        '--server.address=0.0.0.0',
        '--server.headless=false'  # 允许在浏览器中打开
    ]

    print("启动命令:", ' '.join(cmd))
    print("访问地址: http://0.0.0.0:8501")
    print("本地访问: http://localhost:8501")
    print("按 Ctrl+C 停止应用")
    print("-" * 50)

    try:
        # 启动应用
        subprocess.run(cmd, cwd=current_dir)
    except KeyboardInterrupt:
        print("\n应用已停止")
        return 0
    except Exception as e:
        print(f"启动失败: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())