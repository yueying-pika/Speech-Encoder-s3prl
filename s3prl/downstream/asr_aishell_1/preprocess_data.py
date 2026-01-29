import os
import librosa
from tqdm import tqdm  # 引入 tqdm 用于显示进度条

# 定义路径
audio_path = "/CDShare3/ASR/aishell1/data_aishell/wav"
transcript_file = "/CDShare3/ASR/aishell1/data_aishell/transcript/aishell_transcript_v0.8.txt"
output_dir = "/Work21/2026/xuzhao/s3prl/s3prl/downstream/asr_aishell_1/data"  # 修改为你的本地路径

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 读取转录文本文件
with open(transcript_file, 'r', encoding='utf-8') as f:
    transcript_lines = f.readlines()

# 创建转录字典
transcripts = {}
for line in transcript_lines:
    parts = line.strip().split(' ', 1)
    utt_id = parts[0]
    text = parts[1].replace(" ", "")  # 删除中文空格
    transcripts[utt_id] = text

# 获取音频时长的函数
def get_audio_duration(audio_file_path):
    try:
        y, sr = librosa.load(audio_file_path, sr=None)  # sr=None 保持原始采样率
        duration = librosa.get_duration(y=y, sr=sr)
        return duration
    except Exception as e:
        print(f"Error reading {audio_file_path}: {e}")
        return 0  # 如果读取失败，返回0

# 生成txt文件的函数
def generate_file_set(set_name):
    # 遍历文件夹及子文件夹
    set_path = os.path.join(audio_path, set_name)
    output_file = os.path.join(output_dir, f"{set_name}.txt")
    
    # 获取所有 .wav 文件的路径
    wav_files = []
    for root, dirs, files in os.walk(set_path):
        for file_name in files:
            if file_name.endswith(".wav"):  # 只处理 .wav 文件
                wav_files.append(os.path.join(root, file_name))
    
    # 使用 tqdm 显示进度条
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for audio_file_path in tqdm(wav_files, desc=f"Processing {set_name}", unit="file"):
            utt_id = os.path.basename(audio_file_path).split('.')[0]  # 提取文件名前缀作为utt_id
            if utt_id in transcripts:
                # 获取对应的文本
                text = transcripts[utt_id]
                
                # 获取音频文件时长
                duration = get_audio_duration(audio_file_path)
                
                # 如果音频时长为0，跳过该文件
                if duration == 0:
                    print(f"Skipping {utt_id} due to error in audio file.")
                    continue
                
                # 准备输出行格式
                line = f"utt:{utt_id} feat:{audio_file_path} feat_shape:{duration:.2f} text:{text}\n"
                out_f.write(line)

# 生成train、dev、test文件
generate_file_set("train")
generate_file_set("dev")
generate_file_set("test")

# 提示生成完成
print("train.txt, dev.txt, test.txt 文件已生成！")
