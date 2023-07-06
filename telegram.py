import requests
from datetime import datetime, time
import pandas as pd
import time
import math
import os

# Telegram setup for TUMEWKbot
BOT_TOKEN = "1857233194:AAHF5Fi8IQnGeICsbqrDpGBjgpYajLJOgMk"
CHAT_ID = "827047006"


class Telegram:
    def __init__(self, bot_token: str = BOT_TOKEN, chat_id: str = CHAT_ID):
        self.bot_token = bot_token
        self.chat_id = chat_id

    def send_message(self, msg: str):
        """Send a message to the bot on Telegram"""
        msg = msg.replace("_", "\_")
        send_text = 'https://api.telegram.org/bot' + self.bot_token + '/sendMessage?chat_id=' + self.chat_id \
                    + '&parse_mode=Markdown&text=' + msg

        requests.get(send_text)

    def send_photo(self, path: str, caption: str = ""):
        """Send a photo to the bot on Telegram"""
        send_photo = 'https://api.telegram.org/bot' + self.bot_token + '/sendPhoto?chat_id=' + self.chat_id
        files = {'photo': open(path, 'rb')}
        data = {'caption': caption}
        requests.post(send_photo, files=files, data=data)

