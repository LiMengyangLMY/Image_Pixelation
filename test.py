from flask import Flask
from flask_mail import Mail, Message

app = Flask(__name__)

# ———— 请务必修改为你的真实配置 ————
app.config['MAIL_SERVER'] = 'smtp.qq.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USE_SSL'] = True
app.config['MAIL_USERNAME'] = '1364527938@qq.com'
app.config['MAIL_PASSWORD'] = 'klmwjzlnsgsngeab' 

# 【关键修改 1】把 "测试员" 改成 "PinDou Admin" (纯英文)
app.config['MAIL_DEFAULT_SENDER'] = 'PinDou Admin <1364527938@qq.com>'

mail = Mail(app)

if __name__ == '__main__':
    with app.app_context():
        try:
            print("Trying to send email...")
            # 【关键修改 2】标题和内容也暂时改成纯英文
            msg = Message(subject="Test Email", recipients=["你的QQ号@qq.com"])
            msg.body = "If you see this, the config is correct!"
            mail.send(msg)
            print("✅ Success! Email sent.")
        except Exception as e:
            print("\n❌ Failed again:")
            print(e)