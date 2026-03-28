"""
启动 FactorFlow 完整服务（后端API + React前端）
"""
import subprocess
import sys
import webbrowser
import time
import os
import signal
from pathlib import Path

def get_pnpm_cmd():
    if os.name == "nt":
        return "pnpm.cmd"
    else:
        return "pnpm"

def check_pnpm_installed():
    """检查pnpm是否已安装"""
    try:
        result = subprocess.run(
            [get_pnpm_cmd(), "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return True, result.stdout.strip()
        return False, None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False, None

def main():
    print("=" * 60)
    print("启动 FactorFlow 完整服务")
    print("=" * 60)

    # 项目根目录
    project_root = Path(__file__).parent
    frontend_dir = project_root / "frontend" / "react-antd"

    # 检查前端目录
    if not frontend_dir.exists():
        print(f"❌ 错误: 前端目录不存在: {frontend_dir}")
        print("请确保 frontend/react-antd 目录存在")
        return

    # 检查pnpm是否安装
    print("\n检查环境...")
    pnpm_installed, pnpm_version = check_pnpm_installed()
    if not pnpm_installed:
        print("❌ 错误: pnpm 未安装")
        print("请先安装 pnpm: npm install -g pnpm")
        print("或者访问: https://pnpm.io/installation")
        return

    print(f"✓ pnpm 版本: {pnpm_version}")

    # 检查node_modules
    node_modules = frontend_dir / "node_modules"
    if not node_modules.exists():
        print("\n⚠ 警告: node_modules 不存在")
        print("首次运行需要安装依赖，正在自动安装...")
        print("这可能需要几分钟，请耐心等待...")
        try:
            install_result = subprocess.run(
                ["pnpm", "install"],
                cwd=str(frontend_dir),
                check=True
            )
            if install_result.returncode == 0:
                print("✓ 依赖安装完成")
            else:
                print("❌ 依赖安装失败")
                return
        except Exception as e:
            print(f"❌ 依赖安装出错: {e}")
            return

    processes = []

    try:
        # 启动后端 API 服务
        print("\n[1/2] 启动后端 API 服务...")
        print("  执行: uv run python start_api.py")

        # 检查uv是否可用
        use_uv = subprocess.run(
            ["uv", "--version"],
            capture_output=True,
            timeout=5
        ).returncode == 0

        if use_uv:
            api_cmd = ["uv", "run", "python", "start_api.py"]
        else:
            api_cmd = [sys.executable, "start_api.py"]

        api_process = subprocess.Popen(
            api_cmd,
            cwd=str(project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        processes.append(("Backend API", api_process))

        # 等待 API 服务启动
        print("  等待 API 服务启动...")
        time.sleep(3)

        # 检查API进程是否还在运行
        if api_process.poll() is not None:
            print("❌ API 服务启动失败")
            print("请检查 start_api.py 是否可以正常运行")
            return

        print("  ✓ API 服务已启动")

        # 启动前端开发服务器
        print("\n[2/2] 启动 React 前端开发服务器...")
        print("  执行: pnpm dev")

        frontend_process = subprocess.Popen(
            [get_pnpm_cmd(), "dev"],
            cwd=str(frontend_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        processes.append(("Frontend Dev Server", frontend_process))

        # 等待前端服务启动
        print("  等待前端服务启动...")
        time.sleep(5)

        # 检查前端进程是否还在运行
        if frontend_process.poll() is not None:
            print("❌ 前端服务启动失败")
            print("请检查 frontend/react-antd 目录")
            api_process.terminate()
            return

        print("  ✓ 前端服务已启动")

        # 打开浏览器
        print("\n" + "=" * 60)
        print("✓ 所有服务启动完成!")
        print("=" * 60)
        print(f"🌐 前端地址: http://localhost:5173")
        print(f"🔌 API 地址: http://localhost:8000")
        print(f"📚 API 文档: http://localhost:8000/docs")
        print("=" * 60)
        print("\n正在打开浏览器...")

        time.sleep(1)
        webbrowser.open("http://localhost:5173")

        print("\n💡 提示:")
        print("  - 前端支持热更新，修改代码会自动刷新")
        print("  - API 支持自动重载，修改代码会自动重启")
        print("  - 按 Ctrl+C 停止所有服务")
        print("-" * 60)

        # 设置信号处理，确保优雅退出
        def signal_handler(sig, frame):
            print("\n\n收到停止信号，正在关闭所有服务...")
            for name, process in processes:
                try:
                    if process.poll() is None:
                        print(f"  停止 {name}...")
                        process.terminate()
                        process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print(f"  强制停止 {name}...")
                    process.kill()
                except Exception as e:
                    print(f"  停止 {name} 时出错: {e}")
            print("✓ 所有服务已停止")
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        if sys.platform == "win32":
            signal.signal(signal.SIGBREAK, signal_handler)
        else:
            signal.signal(signal.SIGTERM, signal_handler)

        # 等待用户中断
        try:
            while True:
                time.sleep(1)

                # 检查进程状态
                for name, process in processes:
                    if process.poll() is not None:
                        print(f"\n⚠ {name} 已意外停止")
                        print("正在停止所有服务...")
                        raise KeyboardInterrupt

        except KeyboardInterrupt:
            signal_handler(None, None)

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()

        # 清理已启动的进程
        print("\n正在清理已启动的服务...")
        for name, process in processes:
            try:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=3)
            except:
                process.kill()

if __name__ == "__main__":
    main()
