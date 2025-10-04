"""
启动FactorHub应用的脚本
"""
import subprocess
import sys
import os
import argparse

def check_dependencies():
    """检查依赖是否安装"""
    required_packages = [
        'streamlit',
        'pandas',
        'numpy',
        'plotly',
        'akshare'
    ]

    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print("缺少以下依赖包:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\n请运行以下命令安装依赖:")
        print("pip install -r requirements.txt")
        return False

    return True

def start_streamlit_app(port=8501, host="0.0.0.0"):
    """启动Streamlit应用"""
    app_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ui", "app.py")

    if not os.path.exists(app_path):
        print(f"错误: 找不到应用文件 {app_path}")
        return False

    print(f"启动FactorHub应用...")
    print(f"访问地址: http://{host}:{port}")
    print("按 Ctrl+C 停止应用")
    print("-" * 50)

    try:
        # 设置环境变量
        env = os.environ.copy()
        env["STREAMLIT_SERVER_PORT"] = str(port)
        env["STREAMLIT_SERVER_ADDRESS"] = host

        # 启动Streamlit
        cmd = [
            sys.executable, "-m", "streamlit", "run", app_path,
            "--server.port", str(port),
            "--server.address", host,
            "--server.headless", "true"
        ]

        process = subprocess.Popen(cmd, env=env)
        process.wait()

    except KeyboardInterrupt:
        print("\n应用已停止")
        return True
    except Exception as e:
        print(f"启动应用时发生错误: {e}")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="FactorHub 应用启动器")
    parser.add_argument("--port", type=int, default=8501, help="端口号 (默认: 8501)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="主机地址 (默认: 0.0.0.0)")
    parser.add_argument("--check-only", action="store_true", help="只检查依赖，不启动应用")

    args = parser.parse_args()

    print("FactorHub 量化因子分析平台")
    print("=" * 40)

    # 检查依赖
    print("检查依赖...")
    if not check_dependencies():
        sys.exit(1)

    print("依赖检查通过!")
    print()

    if args.check_only:
        print("依赖检查完成，所有必需包已安装。")
        return

    # 启动应用
    start_streamlit_app(args.port, args.host)

if __name__ == "__main__":
    main()