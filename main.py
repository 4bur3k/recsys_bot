import telebot
from telebot.types import InputMediaPhoto
import os
import json
import re
import recsys


def validate_articule(text):
    match = re.search(r'\d{2,4}', text)
    return match[0]


df = recsys.init_data()
model = recsys.init_model()
embeddings = recsys.get_embedings()

with open('./strings.json', encoding='utf-8') as strings_file:
    STRINGS = json.load(strings_file)

with open('./tokens.json') as token_file:
    token = json.load(token_file)['Tokens']['Telegram']['APIKey']

bot = telebot.TeleBot(token)

@bot.message_handler(commands=['start'])
def start(message):
    reply = STRINGS['bot_answers']['start']
    bot.send_message(message.chat.id, reply)

@bot.message_handler(content_types=['text'])
def handle_text(message):
    article = 1000
    try:
        article = int(message.text)
        print('message.text = ', message.text)
    except BaseException: pass

    flag = False
    try:
        flag = recsys.recomendation(article, df, model, embeddings)
    except BaseException as e:
        print(e)

    if flag:
        reply = STRINGS['bot_answers']['submission']
        bot.send_photo(message.chat.id, open('original.jpg', 'rb'), caption=reply)

        media_group = []
        print(os.listdir('./tmp/'))
        for i, name in enumerate(os.listdir('./tmp/')):
            try:
                f = open(f'./tmp/{name}', 'rb')
              
                media_group.append(InputMediaPhoto(f))
            except BaseException:
                pass
        print(media_group)
        reply = STRINGS['bot_answers']['recomendation']
        bot.send_message(message.chat.id, reply)
        bot.send_media_group(message.chat.id, media=media_group)
    else: 
        reply =  STRINGS['bot_answers']['aborted']
        bot.send_message(message.chat.id, reply)

bot.polling(none_stop=True, interval=0)
