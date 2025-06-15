import os
import re
import json
import requests
from lxml import etree
import subprocess
import time


class BilibiliClipDownloader:
    def __init__(self):
        self.headers = {
            "Referer": "https://www.bilibili.com",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0",
            "Origin": "https://www.bilibili.com",
            "Accept": "*/*",
            "Accept-Language": "zh-CN,zh;q=0.9",
            "Accept-Encoding": "gzip, deflate, br"
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def _sanitize_filename(self, filename):
        """Clean invalid filename characters"""
        return re.sub(r'[\\/:*?"<>|]', '', filename)[:100]

    def get_play_info(self, bvid):
        """Get video play info and title"""
        url = f"https://www.bilibili.com/video/{bvid}"
        try:
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    r = self.session.get(url, timeout=10)
                    r.raise_for_status()
                    break
                except requests.RequestException as e:
                    if attempt == max_retries - 1:
                        raise e
                    time.sleep(2 * (attempt + 1))

            html = etree.HTML(r.text)
            title = html.xpath('//h1/@title')
            if not title:
                title = html.xpath('//h1/text()')
            title = title[0].strip() if title else "未知标题"
            title = self._sanitize_filename(title)

            if "你所访问的视频不存在" in r.text:
                raise ValueError("视频不存在或已被删除")

            play_info = re.search(r'window\.__playinfo__=({.*?})</script>', r.text)
            if not play_info:
                raise ValueError("无法提取播放信息，可能需要登录或视频已失效")

            data = json.loads(play_info.group(1))
            
            videos = data["data"]["dash"]["video"]
            audios = data["data"]["dash"]["audio"]
            
            video_url = max(videos, key=lambda x: x.get("bandwidth", 0))["baseUrl"]
            audio_url = max(audios, key=lambda x: x.get("bandwidth", 0))["baseUrl"]

            return video_url, audio_url, title

        except Exception as e:
            print(f"❌ 获取视频信息失败: {str(e)}")
            return None, None, None

    def download_clip(self, bvid, start_sec, end_sec, base_output_dir="downloads"):
        """下载指定片段（静默模式）"""
        # 创建目录结构
        video_dir = os.path.join(base_output_dir, "video")  # 无声视频
        audio_dir = os.path.join(base_output_dir, "audio")  # 纯音频
        output_dir = os.path.join(base_output_dir, "output")  # 合成视频
        for dir_path in [video_dir, audio_dir, output_dir]:
            os.makedirs(dir_path, exist_ok=True)

        # 获取视频信息
        video_url, audio_url, title = self.get_play_info(bvid)
        if not video_url or not audio_url:
            print(f"❌ 无法获取视频流信息: {bvid}")
            return False

        # 准备临时文件路径
        temp_video = os.path.join(base_output_dir, f"temp_{bvid}_video.m4s")
        temp_audio = os.path.join(base_output_dir, f"temp_{bvid}_audio.m4s")
        clip_duration = end_sec - start_sec
        base_filename = f"{self._sanitize_filename(title)}_{bvid}_{int(start_sec)}-{int(end_sec)}s"

        try:
            # 下载临时文件（静默模式）
            print(f"⏳ 正在下载 {title}...")
            for url, temp_file in [(video_url, temp_video), (audio_url, temp_audio)]:
                with requests.get(url, headers=self.headers, stream=True) as r:
                    with open(temp_file, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)

            # 1. 提取音频（静默模式）
            audio_output = os.path.join(audio_dir, f"{base_filename}.mp3")
            subprocess.run([
                "ffmpeg",
                "-loglevel", "error",  # 隐藏技术输出
                "-ss", str(start_sec),
                "-i", temp_audio,
                "-t", str(clip_duration),
                "-c:a", "libmp3lame",
                "-y", audio_output
            ], check=True, stderr=subprocess.DEVNULL)
            print(f"✅ 音频已保存: {os.path.relpath(audio_output)}")

            # 2. 提取无声视频（静默模式）
            video_output = os.path.join(video_dir, f"{base_filename}.mp4")
            subprocess.run([
                "ffmpeg",
                "-loglevel", "error",
                "-ss", str(start_sec),
                "-i", temp_video,
                "-t", str(clip_duration),
                "-c:v", "copy",
                "-an",
                "-y", video_output
            ], check=True, stderr=subprocess.DEVNULL)
            print(f"✅ 无声视频已保存: {os.path.relpath(video_output)}")

            # 3. 合成完整视频（静默模式）
            final_output = os.path.join(output_dir, f"{base_filename}.mp4")
            subprocess.run([
                "ffmpeg",
                "-loglevel", "error",
                "-ss", str(start_sec),
                "-i", temp_video,
                "-ss", str(start_sec),
                "-i", temp_audio,
                "-t", str(clip_duration),
                "-c:v", "copy",
                "-c:a", "aac",
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-y", final_output
            ], check=True, stderr=subprocess.DEVNULL)
            print(f"✅ 完整视频已保存: {os.path.relpath(final_output)}\n")

            return True

        except subprocess.CalledProcessError as e:
            print(f"❌ 处理失败（FFmpeg错误代码 {e.returncode}）")
        except Exception as e:
            print(f"❌ 发生意外错误: {str(e)}")
        finally:
            # 清理临时文件
            for temp_file in [temp_video, temp_audio]:
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except:
                        pass

        return False


def main():
    downloader = BilibiliClipDownloader()

    # Read video info from JSONL file
    with open("bilibili_emotion_videos.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                bvid = data["bilibili_id"]
                start = float(data["start_seconds"])
                end = float(data["end_seconds"])

                print(f"\n▶ 正在处理 {bvid} ({start}-{end}秒)")
                success = downloader.download_clip(bvid, start, end, "downloads")

                if not success:
                    print(f"⚠️ 跳过失败的视频: {bvid}")

                time.sleep(2)  # Avoid being blocked

            except json.JSONDecodeError:
                print("⚠️ 跳过无效的JSON行")
            except KeyError as e:
                print(f"⚠️ 数据格式错误，缺少字段: {str(e)}")


if __name__ == "__main__":
    main()