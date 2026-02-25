# check_env.py

import sys
import subprocess

print("=" * 50)
print("Python 解释器信息")
print("=" * 50)
print("Python 版本:", sys.version)
print("Python 可执行路径:", sys.executable)

print("\n" + "=" * 50)
print("pip 信息")
print("=" * 50)

try:
    pip_version = subprocess.check_output(
        [sys.executable, "-m", "pip", "--version"],
        stderr=subprocess.STDOUT
    )
    print(pip_version.decode())
except Exception as e:
    print("无法获取 pip 信息:", e)

print("\n" + "=" * 50)
print("模块检测")
print("=" * 50)

def check_module(module_name):
    try:
        module = __import__(module_name)
        print("✓ 已安装:", module_name)
        print("  路径:", module.__file__)
    except ModuleNotFoundError:
        print("✗ 未安装:", module_name)
    except Exception as e:
        print("⚠ 检测异常:", module_name, e)

check_module("flask")
check_module("flask_mail")

print("\n检测完成")