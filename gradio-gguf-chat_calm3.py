from llama_cpp import Llama #llama.cppのPythonライブラリであるllama-cpp-python
import gradio as gr #Gradio AIの試作で広く使われているブラウザGUIライブラリ
import re #テキスト整形のライブラリ
import datetime #時刻取得ライブラリ
model="hoge_huga.gguf" #対象のモデルのパスを入力。
llm = Llama(
      model_path=model,
      n_gpu_layers=-1, # #GPUにロードするレイヤー数（llama-cpp-pythonがcuda版の場合）
      n_ctx=4096, # 最大コンテキストサイズ。入力の上限。
      flash_attn=True,
      last_n_tokens_size =0, # Maximum number of tokens to keep in the last_n_tokens deque.
)

role = "<|im_start|>system\n\
あなたは日本語で回答する親切で優秀なAIアシスタントのロールプレイを行ないます。\n\
現在の日付・曜日・時刻はそれぞれ{day}です。\n\
回答に確信が持てない質問、または自信を持って答えるのに十分な情報が無い質問は回答を拒否してください。<|im_end|>"
history = ""
output_history =""
# AIに質問する関数
def complement(role,prompt,turn_config):
   global history,output_history
   day = (str(datetime.datetime.now().year)\
+"年"+str(datetime.datetime.now().month)\
   +"月"+str(datetime.datetime.now().day)\
      +"日"+str(datetime.datetime.now().strftime(" %a "))\
         +str(datetime.datetime.now().hour)\
            +"時"+str(datetime.datetime.now().minute)+"分")\
               .replace("Sun", "日曜日").replace("Mon", "月曜日").replace("Tue", "火曜日").replace("Wed", "水曜日")\
                  .replace("Thu", "木曜日").replace("Fri", "金曜日").replace("Sat", "土曜日")
   role = role.replace("{day}",day)
   #day=""
   role += "\n"
   if prompt !="":
        prompt_C2C3 = (role+history + "USER: "+prompt+"\nASSISTANT: ")\
         .replace("ASSISTANT: ", "<|im_start|>assistant").replace("<|endoftext|>", "<|im_end|>").replace("USER: ", "<|im_start|>user")
        output = llm(
               prompt=prompt_C2C3, # 元々calm2-7b-chat用に作ったプログラムなのでここで整形。
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
               stop=["<|im_","prompt_tokens"] # ストップ。特定の文字を生成したらその文字を生成せず停止する。
        )
        output= output["choices"][0]["text"]
        output =output.replace("\\n", "\n").replace("\\u3000", "\u3000").replace("!","！").replace("?","？")
        while output[-1]=="\n":
              output=output[:-1]
        while output[0]=="\n":
              output=output[1:] 
        print( prompt_C2C3+output+"<|im_end|>")
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
      turn = re.split(r'(?=USER: )', history)
      del turn[0:1]
      output_history =''.join(turn)
      output_history = output_history.replace("<|endoftext|>", '')
      turn_count = len(turn)
   return output, output_history

# 履歴リセット関数
def hist_rst():
    global history
    prompt=""
    output=""
    history=""
    output_history=""
    return prompt, output, output_history

# 会話Undo関数
def undo():
    global history
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
    return prompt, output,output_history
    

# Blocksの作成
with gr.Blocks(title="ﾁｬｯﾄﾎﾞｯﾄ",theme=gr.themes.Base(primary_hue="orange", secondary_hue="blue")) as demo:
    # コンポーネント
    gr.Markdown(
    """
    チャットボット(calm3-22b-chat)
    """)
    # UI
    with gr.Row():
     with gr.Column(scale=1): 
      prompt = gr.Textbox(lines=2,label="質問入力")
      output = gr.Textbox(label="回答出力")
      greet_btn = gr.Button(value="送信",variant='primary')
      with gr.Accordion(label="会話履歴設定", open=False ):
        with gr.Accordion(label="システムプロンプト", open=False):
           role = gr.Textbox(lines=26,label="calm3-chatのチャットテンプレーで書いてください", value=role)
        turn_config = gr.Number(label="会話ターン数設定",value=10,minimum=1,maximum=20)
        disphist = gr.Textbox(lines=10,label="会話履歴出力")
      undo_btn = gr.Button(value="1ターン戻す",variant='secondary')
      reset_btn = gr.Button(value="履歴リセット",variant='secondary')
       # イベントハンドラー
      greet_btn.click(fn=complement, inputs=[role,prompt,turn_config], outputs=[output,disphist])
      undo_btn.click(fn=undo, outputs=[prompt,output, disphist])
      reset_btn.click(fn=hist_rst, outputs=[prompt,output, disphist])
#demo.launch(auth=("XXXX","YYYY"),share=True, server_port=7860,show_api=False)
#demo.launch(server_name="192.168.x.xxx", server_port=7860,show_api=False)
demo.launch(show_api=False)