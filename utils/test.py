import requests
import sys

# ==========================================
# ⚠️ 请在这里填入您刚刚重新生成的 API Key
# ==========================================
API_KEY = "0502195f6ecb8fa1d60ac1fe46b4f2e0" 

def test_connection():
    print("------------------------------------------------")
    print("📡 正在测试 TMDB API 连接...")
    print(f"🔑 使用 Key: {API_KEY[:6]}******") # 只显示前几位，保护隐私
    
    # 1. 设置请求目标 (搜索 'Inception')
    url = "https://api.themoviedb.org/3/search/movie"
    params = {
        'api_key': API_KEY,
        'query': 'Inception',
        'language': 'en-US'
    }
    
    try:
        # 2. 发起请求
        # timeout=10 意味着如果 10秒内连不上，就报错，防止无限等待
        response = requests.get(url, params=params, timeout=10)
        
        # 3. 检查 HTTP 状态码
        print(f"📥 HTTP 状态码: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            results = data.get('results', [])
            
            if results:
                first_movie = results[0]
                print("\n 测试成功！API 工作正常。")
                print(f"🎬 搜索到的电影: {first_movie['title']}")
                print(f"🆔 TMDB ID: {first_movie['id']}")
                print(f"Nw 简介片段: {first_movie['overview'][:50]}...")
            else:
                print("❓ 连接成功，但没有返回结果。请检查搜索词。")
                
        elif response.status_code == 401:
            print("\n❌ 认证失败 (401)")
            print("请检查您的 API Key 是否填写正确，或者是否刚刚生成（可能需要几分钟生效）。")
            
        elif response.status_code == 404:
            print("\n❌ 找不到资源 (404)")
            print("API 路径可能变了，请检查 URL。")
            
        else:
            print(f"\n❌ 请求失败: {response.text}")

    except requests.exceptions.ConnectionError:
        print("\n [致命错误] 无法连接到 api.themoviedb.org")
        print("原因：网络不可达。")
        print("排查建议：")
        print("1. 检查服务器是否有外网访问权限。")
        print("2. 检查防火墙设置。")
        print("3. 如果在公司内网，可能需要配置 HTTP_PROXY。")
        
    except requests.exceptions.Timeout:
        print("\n [超时错误] 连接 TMDB 响应太慢。")
        print("建议：网络可能拥堵，请稍后再试。")
        
    except Exception as e:
        print(f"\n 发生了未知的 Python 错误: {e}")

if __name__ == "__main__":
    if "YOUR_NEW" in API_KEY:
        print(" 请先修改脚本中的 API_KEY 变量！")
    else:
        test_connection()