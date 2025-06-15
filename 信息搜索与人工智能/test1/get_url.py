import json
import requests
from typing import List, Dict
import time
import random
import re


def get_bilibili_cookies():
    """获取基本的B站cookies"""
    return {
        "buvid3": str(random.randint(1000000000000000, 9999999999999999)),
        "fingerprint": str(random.randint(1000000000000000, 9999999999999999)),
        "sid": ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=8))
    }


def is_valid_bvid(bvid: str) -> bool:
    """检查是否为有效的B站视频ID格式"""
    return bool(re.match(r'^BV[a-zA-Z0-9]{10}$', bvid))


def search_bilibili_emotion_videos(keywords: List[str], max_results: int = 100) -> List[Dict]:
    """
    搜索B站上与自然主题相关的视频
    :param keywords: 情绪相关的关键词列表
    :param max_results: 最大返回结果数
    :return: 视频信息列表
    """
    base_url = "https://api.bilibili.com/x/web-interface/search/all/v2"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Referer": "https://www.bilibili.com/",
        "Origin": "https://www.bilibili.com",
    }

    cookies = get_bilibili_cookies()

    videos = []
    for keyword in keywords:
        params = {
            "keyword": keyword,
            "page": 1,
            "pagesize": max(5, max_results // len(keywords))  # 每关键词至少5个结果
        }

        try:
            # 添加随机延迟避免被封
            time.sleep(random.uniform(1, 2))

            response = requests.get(
                base_url,
                headers=headers,
                params=params,
                cookies=cookies
            )

            # 检查响应状态
            if response.status_code == 412:
                print(f"关键词 '{keyword}' 可能被限制，尝试下一个关键词...")
                continue

            response.raise_for_status()
            data = response.json()

            if data.get("code") == 0:
                for item in data["data"]["result"]:
                    if item["result_type"] == "video":
                        for v in item["data"]:
                            # 检查bvid是否存在且有效
                            bvid = v.get("bvid", "")
                            if not is_valid_bvid(bvid):
                                print(f"跳过无效视频ID: {bvid} (标题: {v.get('title', '无标题')})")
                                continue

                            # 模拟情绪标签
                            labels = [f"/m/{keyword}"] + [
                                f"/m/{random.randint(1000, 9999)}"
                                for _ in range(random.randint(2, 5))
                            ]

                            # 处理duration字段
                            duration_str = v.get("duration", "0:0")
                            duration_parts = duration_str.split(":")
                            if len(duration_parts) == 2:
                                duration_sec = int(duration_parts[0]) * 60 + int(duration_parts[1])
                            else:
                                duration_sec = 60  # 默认值

                            # 随机选择视频中的10秒片段
                            max_start = max(0, duration_sec - 10)
                            start = random.uniform(0, max_start)

                            video_info = {
                                "bilibili_id": bvid,
                                "start_seconds": round(start, 1),
                                "end_seconds": round(start + 10, 1),
                                "label": labels,
                                "title": v.get("title", ""),
                                "duration": duration_sec
                            }
                            videos.append(video_info)

                            # 达到最大结果数时停止
                            if len(videos) >= max_results:
                                return videos

        except Exception as e:
            print(f"搜索关键词 '{keyword}' 时出错: {e}")

    return videos


def save_to_jsonl(videos: List[Dict], filename: str):
    """
    将视频信息保存为JSONL文件
    :param videos: 视频信息列表
    :param filename: 输出文件名
    """
    valid_count = 0
    with open(filename, 'w', encoding='utf-8') as f:
        for video in videos:
            # 再次检查bvid有效性
            if not is_valid_bvid(video["bilibili_id"]):
                print(f"警告: 跳过无效视频ID: {video['bilibili_id']} (标题: {video.get('title', '无标题')})")
                continue

            # 只保留与示例JSONL中相同的字段
            output = {
                "bilibili_id": video["bilibili_id"],
                "start_seconds": video["start_seconds"],
                "end_seconds": video["end_seconds"],
                "label": ",".join(video["label"])  # 将标签列表转为逗号分隔字符串
            }
            f.write(json.dumps(output, ensure_ascii=False) + '\n')
            valid_count += 1

    return valid_count


if __name__ == "__main__":
    # 自然相关的关键词 - 使用更中性的词汇
    emotion_keywords = [
        "流水", "自然", "鸟鸣", "犬吠", "乡村",
        "噪声", "车流", "鼓掌", "放松时刻","交谈","海浪","强烈情绪","喊叫","自然风光"
    ]

    print("开始搜索B站自然主题视频...")

    # 搜索视频
    videos = search_bilibili_emotion_videos(emotion_keywords, max_results=100)

    # 保存为JSONL文件并获取有效视频数
    valid_count = save_to_jsonl(videos, "bilibili_emotion_videos.jsonl")

    print(f"已保存 {valid_count} 个有效视频信息到 bilibili_emotion_videos.jsonl")