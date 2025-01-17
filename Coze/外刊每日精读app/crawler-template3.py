import requests
from bs4 import BeautifulSoup

# 目标网页URL
url = "https://www.nationalgeographic.com/history/article/origin-pajamas-pjs"

# 发送HTTP请求获取网页内容
response = requests.get(url)
response.raise_for_status()  # 检查请求是否成功

# 使用BeautifulSoup解析HTML
soup = BeautifulSoup(response.text, 'html.parser')

# 提取文章标题
title = soup.find('h1').get_text()
print(f"Title: {title}")

# 提取文章插图（假设插图的标签是<img>，并且有特定的class或属性）
# images = soup.find_all('img')
# for img in images:
#     print(f"Image URL: {img['src']}")

subtitle = soup.find('div', attrs={'data-testid': 'prism-GridColumn'})
print(subtitle)

# 提取文章正文（假设正文在某个特定的div或article标签内）
# 需要根据实际网页结构调整选择器
content = soup.find('div', attrs={'data-testid': 'prism-article-body'})  # 这里的class_需要根据实际网页结构调整
if content:
    print(content)
    paragraphs = content.find_all('p')
    for p in paragraphs:
        print(p.get_text())
else:
    print("Content not found")