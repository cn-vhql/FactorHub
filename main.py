"""
FactorHub 主程序入口
"""
import streamlit.web.cli as stcli
import sys
import os

def main():
    """启动FactorHub应用"""
    os.environ["STREAMLIT_SERVER_PORT"] = "8501"
    os.environ["STREAMLIT_SERVER_ADDRESS"] = "0.0.0.0"
    sys.argv = ["streamlit", "run", "ui/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
    sys.exit(stcli.main())

if __name__ == "__main__":
    main()