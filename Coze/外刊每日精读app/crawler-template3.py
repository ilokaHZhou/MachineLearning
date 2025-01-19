import requests
from bs4 import BeautifulSoup

# 目标网页URL
url = "https://www.nationalgeographic.com/history/article/origin-pajamas-pjs"

result = {}

# 发送HTTP请求获取网页内容
response = requests.get(url)
response.raise_for_status()  # 检查请求是否成功

# 使用BeautifulSoup解析HTML
soup = BeautifulSoup(response.text, 'html.parser')

# 提取文章标题
caption_tag = soup.find('div', attrs={'data-testid': 'prism-GridColumn'})
if caption_tag:
    h1_tag = caption_tag.find('h1')  # 查找 <h1> 标签
    if h1_tag and not isinstance(h1_tag, int):  # 确保 <h1> 标签存在
        title = h1_tag.get_text().strip()  # 提取标题文本
        result['title'] = title
    p_tag = caption_tag.find('p')  # 查找 <p> 标签
    if p_tag and not isinstance(p_tag, int):  # 确保 <p> 标签存在
        subtitle = p_tag.get_text().strip()  # 提取副标题文本
        result['subtitle'] = subtitle
    print(result)


# 提取文章插图（假设插图的标签是<img>，并且有特定的class或属性）
main_img = soup.find('img', src=True, attrs={'data-testid': 'prism-image'}) 
result['picture'] = main_img['src']

# 提取文章正文（假设正文在某个特定的div或article标签内）
# 需要根据实际网页结构调整选择器
result['paragraphs'] = []
body = soup.find('div', attrs={'data-testid': 'prism-article-body'})  # 这里的class_需要根据实际网页结构调整
if body:
    body_content = body.find('div', attrs={'data-testid': 'prism-GridColumn'})
    if body_content and not isinstance(body_content, int):
        paragraphs = body_content.find_all('p')
        for p in paragraphs:
            result['paragraphs'].append(p.get_text())
