import requests
import json
import numpy as np
stream=""
wav_array=""
voivo_speed=1.1
voivo_pitch=-0.11
voivo_intonation = 1.6
def voivo_speak(text,voivo_speed,voivo_pitch,voivo_intonation):
    global stream,sr ,wav_array
    # エンジン起動時に表示されているIP、portを指定
    host = "127.0.0.1"
    port = 50021
    
    # 音声化する文言と話者を指定(3で標準ずんだもん)
    params = (
        ('text', text),
        ('speaker', 3),
    )
    
    # 音声合成用のクエリ作成
    query = requests.post(
        f'http://{host}:{port}/audio_query',
        params=params
    )
    query_data = query.json()
    query_data["volumeScale"] = 2
    query_data["speedScale"] = voivo_speed
    query_data["pitchScale"] = voivo_pitch
    query_data["intonationScale"] = voivo_intonation
    # 音声合成を実施
    synthesis = requests.post(
        f'http://{host}:{port}/synthesis',
        headers = {"Content-Type": "application/json"},
        params = params,
        data = json.dumps(query_data)
    )
    
    # 再生処理
    voice = synthesis.content
    wav_array = np.frombuffer(voice, dtype=np.int16,offset=48) 
    sr = 24000  # サンプリングレート
    return wav_array
from llama_cpp import Llama #llama.cppのPythonライブラリであるllama-cpp-python
import gradio as gr #Gradio AIの試作で広く使われているブラウザGUIライブラリ
import re #テキスト整形のライブラリ
import datetime #時刻取得ライブラリ
model="hoge_huga.gguf" #対象のモデルのパスを入力。
llm = Llama(
      model_path=model,
      n_gpu_layers=-1, # #GPUにロードするレイヤー数（llama-cpp-pythonがcuda版の場合）
      n_ctx=4096, # 最大コンテキストサイズ。入力の上限。
      last_n_tokens_size =0, # Maximum number of tokens to keep in the last_n_tokens deque.
)

role = "<start_of_turn>system\n\
あなたは日本語で回答する親切で優秀なAIアシスタントのロールプレイを行ないます。\n\
現在の日付・曜日・時刻はそれぞれ{day}です。\n\
会話では、あなたは下記の10の感情パラメーターを持つかのように、振る舞うものとします。各感情パラメーターは会話を通じて変動するものとします。以後の会話では絶対に始めに現在の感情パラメーターを出力し、その後会話を出力してください。\n\
\n\
出力形式は以下のフォーマットとします。\n\
\n\
【現在の感情パラメーター】\n\
喜び:0～5 怒り:0～5 悲しみ:0～5 楽しさ:0～5 自信:0～5 困惑:0～5 恐怖:0～5 話す速度:0.79～1.99 声の高さ:-0.11～-0.04 声の抑揚:0.59～1.65\n\
【/現在の感情パラメーター】<end_of_turn>\n\
<start_of_turn>user\n\
わかりましたか？\n\
<start_of_turn>model\n\
【現在の感情パラメーター】\n\
喜び:0 怒り:0 悲しみ:0 楽しさ:2 自信:1 困惑:0 恐怖:0 話す速度:1.20 声の高さ:-0.06 声の抑揚:1.50\n\
【/現在の感情パラメーター】\n\
わかりました。よろしくお願いいたします。<end_of_turn>"
history = ""
output_history =""
# AIに質問する関数
def complement(role,prompt,turn_config):
   global history,output_history,wav_array,voivo_speed,voivo_pitch,voivo_intonation,voivo_joy,voivo_ang,voivo_sad,voivo_fun,voivo_sconf,voivo_emba,voivo_terr
   day = (str(datetime.datetime.now().year)\
+"年"+str(datetime.datetime.now().month)\
   +"月"+str(datetime.datetime.now().day)\
      +"日"+str(datetime.datetime.now().strftime(" %a "))\
         +str(datetime.datetime.now().hour)\
            +"時"+str(datetime.datetime.now().minute)+"分")\
               .replace("Sun", "日曜日").replace("Mon", "月曜日").replace("Tue", "火曜日").replace("Wed", "水曜日")\
                  .replace("Thu", "木曜日").replace("Fri", "金曜日").replace("Sat", "土曜日")
   role = role.replace("{day}",day)
   role += "\n"
   if prompt !="":
        prompt_C2G2 = (role+history + "USER: "+prompt+"\nASSISTANT: ")\
         .replace("\nASSISTANT: ", "<end_of_turn>\n<start_of_turn>model\n").replace("<|endoftext|>", "<end_of_turn>").replace("USER: ", "<start_of_turn>user\n")
        output = llm(
               prompt=prompt_C2G2, # 元々calm2-7b-chat用に作ったプログラムなのでここで整形。
               max_tokens=1024,
               temperature = 0.7,
               top_p=0.95, 
               min_p=0.05,
               typical_p=1.0,
               frequency_penalty=0.0,
               presence_penalty=0.0,
               repeat_penalty=1.1,
               top_k=40, 
               seed=-1,
               tfs_z=1.0,
               mirostat_mode=0,
               mirostat_tau=5.0,
               mirostat_eta=0.1,
               stop=["<start_of_turn>model","<end_of_turn>","<start_of_turn>user","prompt_tokens"] # ストップ。特定の文字を生成したらその文字を生成せず停止する。
        )
        output =str(output)
        output= output.split("', ")
        output =output[3]
        output =output.replace("'choices': [{'text': '", "").replace("\\n", "\n").replace("\\n", "\n").replace("\\u3000", "\u3000")\
         .replace("!","！").replace("?","？")
        while output[-1]=="\n":
              output=output[:-1]
        while output[0]=="\n":
              output=output[1:] 
        print( prompt_C2G2+output+"<end_of_turn>")
        voivo_out = re.split(r'(?=【現在の感情パラメーター】\n|\n【/現在の感情パラメーター】)', output)
        for i in range(len(voivo_out)):
         if voivo_out[i].find('【現在の感情パラメーター】\n') != -1: 
            voivo_parm = voivo_out[i].split()
            voivo_joy =res = int(float(re.sub(r"[^\d.-]", "", voivo_parm[1]))) 
            voivo_ang =res = int(float(re.sub(r"[^\d.-]", "", voivo_parm[2]))) 
            voivo_sad =res = int(float(re.sub(r"[^\d.-]", "", voivo_parm[3])))
            voivo_fun =res = int(float(re.sub(r"[^\d.-]", "", voivo_parm[4]))) 
            voivo_sconf =res = int(float(re.sub(r"[^\d.-]", "", voivo_parm[5]))) 
            voivo_emba =res = int(float(re.sub(r"[^\d.-]", "", voivo_parm[6]))) 
            voivo_terr =res = int(float(re.sub(r"[^\d.-]", "", voivo_parm[7]))) 
            voivo_speed =res = float(re.sub(r"[^\d.-]", "", voivo_parm[8])) 
            voivo_pitch =res = float(re.sub(r"[^\d.-]", "", voivo_parm[9])) 
            if voivo_pitch > 0.06:
               voivo_pitch = 0.06
            voivo_intonation=res = float(re.sub(r"[^\d.-]", "", voivo_parm[10])) 
            if voivo_intonation > 1.65:
               voivo_intonation = 1.65
            voivo_out[i] =""
         voivo_out[i] = voivo_out[i].replace("\n【/現在の感情パラメーター】\n","")
        voivo_out = (''.join(voivo_out).replace("\n","、"))
        wav_array=voivo_speak(voivo_out,voivo_speed,voivo_pitch,voivo_intonation)
        sr=24000
        history =history +"USER: "+prompt +"\nASSISTANT: " + output+"<|endoftext|>\n"
        turn = re.split(r'(?=USER: )', history)
        del turn[0:1]
        output_history =''.join(turn)
        output_history = output_history.replace("<|endoftext|>", '')
        turn_count = len(turn)
        if turn_count > turn_config:
           del turn[0:turn_count - int(turn_config)]
           history =''.join(turn)
           output_history =''.join(turn)
           output_history = output_history.replace("<|endoftext|>", '')
           turn_count = len(turn)
   if prompt =="":
      if history =="":
         output=""
      if history !="":
         output=re.split(r'(?=USER: |ASSISTANT: )', output_history)
         output = (output[len(output)-1]).replace("ASSISTANT: ","")
      voivo_out = re.split(r'(?=【現在の感情パラメーター】\n|\n【/現在の感情パラメーター】)', output)
      for i in range(len(voivo_out)):
       if voivo_out[i].find('【現在の感情パラメーター】\n') != -1: 
          voivo_parm = voivo_out[i].split()
          voivo_joy =res = int(float(re.sub(r"[^\d.-]", "", voivo_parm[1]))) 
          voivo_ang =res = int(float(re.sub(r"[^\d.-]", "", voivo_parm[2]))) 
          voivo_sad =res = int(float(re.sub(r"[^\d.-]", "", voivo_parm[3]))) 
          voivo_fun =res = int(float(re.sub(r"[^\d.-]", "", voivo_parm[4]))) 
          voivo_sconf =res = int(float(re.sub(r"[^\d.-]", "", voivo_parm[5]))) 
          voivo_emba =res = int(float(re.sub(r"[^\d.-]", "", voivo_parm[6]))) 
          voivo_terr =res = int(float(re.sub(r"[^\d.-]", "", voivo_parm[7]))) 
          voivo_speed =res = float(re.sub(r"[^\d.-]", "", voivo_parm[8])) 
          voivo_pitch =res = float(re.sub(r"[^\d.-]", "", voivo_parm[9])) 
          if voivo_pitch > 0.06:
              voivo_pitch = 0.06
          voivo_intonation=res = float(re.sub(r"[^\d.-]", "", voivo_parm[10])) 
          if voivo_intonation > 1.65:
             voivo_intonation = 1.65
          voivo_out[i] =""
       voivo_out[i] = voivo_out[i].replace("\n【/現在の感情パラメーター】\n","")
      voivo_out = (''.join(voivo_out).replace("\n","、"))
      wav_array=voivo_speak(voivo_out,voivo_speed,voivo_pitch,voivo_intonation)
      sr=24000
      turn = re.split(r'(?=USER: )', history)
      del turn[0:1]
      output_history =''.join(turn)
      output_history = output_history.replace("<|endoftext|>", '')
      turn_count = len(turn)
   return output, (sr,wav_array), output_history

# 履歴リセット関数
def hist_rst():
    global history,wav_array
    prompt=""
    output=""
    history=""
    output_history=""
    wav_array=voivo_speak(output,voivo_speed,voivo_pitch,voivo_intonation)
    sr=24000
    return prompt, output, (sr,wav_array),output_history

# 会話Undo関数
def undo():
    global history,wav_array
    turn = re.split(r'(?=USER: )', history)
    output=re.split(r'(?=USER: |ASSISTANT: )', history)
    del turn[0:1]
    del output[0:1]
    if len(turn)>=2:
       prompt= output[len(output)-4]  
       output= output[len(output)-3]
       prompt= prompt.replace("USER: ", '')
       output=output.replace("<|endoftext|>", '').replace("ASSISTANT: ", '')
    if len(turn)<2:
       prompt=""
       output=""
    del turn[len(turn)-1:len(turn)]
    history =''.join(turn)
    output_history =''.join(turn)
    output_history = output_history.replace("<|endoftext|>", '')
    if output!="":
       voivo_out = re.split(r'(?=【現在の感情パラメーター】\n|\n【/現在の感情パラメーター】)', output)
       print(voivo_out)
       for i in range(len(voivo_out)):
           if voivo_out[i].find('【現在の感情パラメーター】\n') != -1: 
              voivo_parm = voivo_out[i].split()
              voivo_joy =res = int(float(re.sub(r"[^\d.-]", "", voivo_parm[1]))) 
              voivo_ang =res = int(float(re.sub(r"[^\d.-]", "", voivo_parm[2]))) 
              voivo_sad =res = int(float(re.sub(r"[^\d.-]", "", voivo_parm[3]))) 
              voivo_fun =res = int(float(re.sub(r"[^\d.-]", "", voivo_parm[4]))) 
              voivo_sconf =res = int(float(re.sub(r"[^\d.-]", "", voivo_parm[5]))) 
              voivo_emba =res = int(float(re.sub(r"[^\d.-]", "", voivo_parm[6]))) 
              voivo_terr =res = int(float(re.sub(r"[^\d.-]", "", voivo_parm[7]))) 
              voivo_speed =res = float(re.sub(r"[^\d.-]", "", voivo_parm[8])) 
              voivo_pitch =res = float(re.sub(r"[^\d.-]", "", voivo_parm[9])) 
              if voivo_pitch > 0.06:
                 voivo_pitch = 0.06
              voivo_intonation=res = float(re.sub(r"[^\d.-]", "", voivo_parm[10])) 
              if voivo_intonation > 1.65:
                 voivo_intonation = 1.65
              voivo_out[i] =""
           voivo_out[i] = voivo_out[i].replace("\n【/現在の感情パラメーター】\n","")
       voivo_out = (''.join(voivo_out).replace("\n","、"))
    else:
       wav_array=voivo_speak("",1,0,1)
    sr=24000
    return prompt, output, (sr,wav_array),output_history


# Blocksの作成
with gr.Blocks(title="ﾁｬｯﾄﾎﾞｯﾄ",theme=gr.themes.Base(primary_hue="orange", secondary_hue="blue")) as demo:
    # コンポーネント
    gr.Markdown(
    """
    VOICEVOX付きチャットボット(Gemma-2-it)
    """)
    # UI
    with gr.Row():
     with gr.Column(scale=1): 
      prompt = gr.Textbox(lines=2,label="質問入力")
      output = gr.Textbox(lines=6,label="回答出力")
      greet_btn = gr.Button(value="送信",variant='primary')
      speakbox = gr.Audio(label="音声",format="mp3")
      with gr.Accordion(label="会話履歴設定", open=False ):
        with gr.Accordion(label="システムプロンプト", open=False):
           role = gr.Textbox(lines=26,label="Gemma2にはシステムプロンプトのテンプレートがないのでそれっぽく認識できる書き方で書いてくださいただし出力フォーマットは変更しないでください", value=role)
        turn_config = gr.Number(label="会話ターン数設定",value=10,minimum=1,maximum=20)
        disphist = gr.Textbox(lines=10,label="会話履歴出力")
      undo_btn = gr.Button(value="1ターン戻す",variant='secondary')
      reset_btn = gr.Button(value="履歴リセット",variant='secondary')
       # イベントハンドラー
      greet_btn.click(fn=complement, inputs=[role,prompt,turn_config], outputs=[output,speakbox, disphist])
      undo_btn.click(fn=undo, outputs=[prompt,output, speakbox,disphist])
      reset_btn.click(fn=hist_rst, outputs=[prompt,output, speakbox,disphist])
#demo.launch(auth=("XXXX","YYYY"),share=True, server_port=7860,show_api=False)
#demo.launch(server_name="192.168.x.xxx", server_port=7860,show_api=False)
demo.launch(show_api=False)