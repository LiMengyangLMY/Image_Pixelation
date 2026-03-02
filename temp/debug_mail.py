# debug_mail.py
import socket
import os

# 1. 强制修复主机名问题（最常见的 Windows 报错原因）
socket.gethostname = lambda: "localhost"

from flask import Flask
from flask_mail import Mail, Message

app = Flask(__name__)

# ——————————————————————————————————————
#      使用你 app.py 里的配置进行测试
# ——————————————————————————————————————
app.config['MAIL_SERVER'] = 'smtp.qq.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USE_SSL'] = True
app.config['MAIL_USERNAME'] = '1364527938@qq.com'  # 你的账号
app.config['MAIL_PASSWORD'] = 'klmwjzlnsgsngeab'  # 你的授权码
# 注意：为了测试，我先把发件人简化为纯邮箱，排除名字格式问题
app.config['MAIL_DEFAULT_SENDER'] = app.config['MAIL_USERNAME'] 

mail = Mail(app)

def test_send():
    print("🚀 正在尝试连接 QQ 邮箱服务器...")
    print(f"📧 发送账号: {app.config['MAIL_USERNAME']}")
    
    with app.app_context():
        try:
            msg = Message(
                subject="拼豆项目测试邮件 (Debug)", 
                recipients=[app.config['MAIL_USERNAME']], # 发给自己
                body="恭喜！如果你收到这封信，说明邮件配置完全正确。"
            )
            mail.send(msg)
            print("\n✅✅✅ 发送成功！")
            print("请立即去查看你的 QQ 邮箱收件箱（包括垃圾箱）。")
            print("如果收到了，说明问题出在 app.py 代码没加 socket 补丁。")
            
        except Exception as e:
            print("\n❌❌❌ 发送失败！报错信息如下：")
            print("————————————————————————————————")
            print(e)
            print("————————————————————————————————")
            
            # 智能分析报错
            err_str = str(e)
            if "Authentication failed" in err_str or "535" in err_str:
                print("👉 分析：授权码错误。请重新去 QQ 邮箱生成一个新的授权码。")
            elif "timed out" in err_str:
                print("👉 分析：网络连接超时。可能是公司网络拦截了 465 端口。")
            elif "ascii" in err_str:
                print("👉 分析：依然是编码问题，请确保文件名或路径不含中文。")

if __name__ == "__main__":
    test_send()