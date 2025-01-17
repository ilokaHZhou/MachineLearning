import requests
from bs4 import BeautifulSoup

# 公众号文章页面的URL，这里需要替换成你想要爬取的公众号文章页面
url = 'https://mp.weixin.qq.com/s/your_article_url'

# 发送HTTP请求
response = requests.get(url)
response.encoding = 'utf-8'  # 根据实际情况设置编码

# 检查请求是否成功
if response.status_code == 200:
    # 使用BeautifulSoup解析HTML内容
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # 假设文章内容在某个特定的class或id中，这里需要根据实际页面结构进行调整
    # 例如，如果文章内容在一个id为'article_content'的div中
    article_content = soup.find('div', id='article_content')
    
    # 提取文章标题
    title = soup.find('h1').text
    