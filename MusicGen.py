from flask import Flask, request, send_file, render_template
from audiocraft.models import MusicGen
from audiocraft.modules.transformer import set_efficient_attention_backend
import scipy.io.wavfile
import numpy as np
import torch

app = Flask(__name__)

# GPU 확인
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"사용 디바이스: {device}")
if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print("모델 로딩 중...")
model = MusicGen.get_pretrained('facebook/musicgen-small')
#model = MusicGen.get_pretrained('facebook/musicgen-medium')  # 1.5GB
#model = MusicGen.get_pretrained('facebook/musicgen-large')  # 3.3GB
model.set_generation_params(
    duration=15,      # 30초로 늘리기
    top_k=250,
    top_p=0.0,
    temperature=0.9,  # 조금 낮춰서 안정적으로
    cfg_coef=5.0      # 프롬프트 충실도 높이기
)

set_efficient_attention_backend('torch')
for module in model.lm.modules():
    if hasattr(module, 'memory_efficient'):
        module.memory_efficient = False

print("모델 로딩 완료!")

@app.route('/generate', methods=['POST'])
def generate():
    prompt = request.form.get('prompt', 'happy jazz music')
    print(f"생성 중: {prompt}")

    wav = model.generate([prompt])

    wav_np = wav[0].cpu().numpy()
    wav_file = "output.wav"
    scipy.io.wavfile.write(wav_file, model.sample_rate, wav_np.T)

    print("생성 완료!")
    return send_file(wav_file, mimetype='audio/wav')

@app.route('/')
def index():
    return render_template('index.html', device=device)

if __name__ == "__main__":
    app.run(debug=True)