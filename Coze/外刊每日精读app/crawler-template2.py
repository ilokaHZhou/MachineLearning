import requests
from bs4 import BeautifulSoup
import json

"""
Each file needs to export a function named `handler`. This function is the entrance to the Tool.

Parameters:
args: parameters of the entry function.
args.input - input parameters, you can get test input value by args.input.xxx.
args.logger - logger instance used to print logs, injected by runtime.

Remember to fill in input/output in Metadata, it helps LLM to recognize and use tool.

Return:
The return data of the function, which should match the declared output parameters.
"""
cookies = {
    '_ga': 'GA1.2.1029573471.1726262426',
    '_gid': 'GA1.2.1394220240.1735013115',
    'Hm_lvt_5a48fb280b334d499dae14e06d7bcbb5': '1733580187,1735013117',
    'HMACCOUNT': '2D3985402338B97B',
    'Hm_lpvt_5a48fb280b334d499dae14e06d7bcbb5': '1735119738',
    '_ga_6P7RXJKJRT': 'GS1.2.1735119738.10.0.1735119738.0.0.0',
}

headers = {
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
    'Accept-Language': 'zh-CN,zh;q=0.9',
    'Cache-Control': 'max-age=0',
    'Connection': 'keep-alive',
    # 'Cookie': '_ga=GA1.2.1029573471.1726262426; _gid=GA1.2.1394220240.1735013115; Hm_lvt_5a48fb280b334d499dae14e06d7bcbb5=1733580187,1735013117; HMACCOUNT=2D3985402338B97B; Hm_lpvt_5a48fb280b334d499dae14e06d7bcbb5=1735119738; _ga_6P7RXJKJRT=GS1.2.1735119738.10.0.1735119738.0.0.0',
    'Sec-Fetch-Dest': 'document',
    'Sec-Fetch-Mode': 'navigate',
    'Sec-Fetch-Site': 'none',
    'Sec-Fetch-User': '?1',
    'Upgrade-Insecure-Requests': '1',
    'User-Agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Mobile Safari/537.36',
    'sec-ch-ua': '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
    'sec-ch-ua-mobile': '?1',
    'sec-ch-ua-platform': '"Android"',
}

response = requests.get('https://www.qbitai.com/', cookies=cookies, headers=headers)


def scrape_news(url):
    print(response)
    soup = BeautifulSoup(response.text, 'html.parser')
    print(soup)
    result = []
    swiper_slides = soup.find_all('div', class_='swiper-slide')

    # 提取每个 swiper-slide 中的标题、链接和图片
    for slide in swiper_slides:
        link_tag = slide.find('a', href=True)  # 找到 <a> 标签，带有 href 属性
        img_tag = slide.find('img', src=True)  # 找到 <img> 标签，带有 src 属性
        caption_tag = slide.find('div', class_='carousel-caption')  # 查找标题所在的 <div> 标签

        if link_tag and img_tag and caption_tag:
            # 提取信息
            title = caption_tag.find('h3').get_text().strip() if caption_tag.find('h3') else ''
            link = link_tag['href']
            picture = img_tag['src']

            # 将提取到的信息添加到结果列表
            result.append({
                'title': title,
                'link': link,
                'picture': picture,
                'author': '',
                'time': ''
            })

    picture_text = soup.find_all('div', class_='picture_text')
    for slide in picture_text:
        link_tag = slide.find('a', href=True)  # 找到 <a> 标签，带有 href 属性
        img_tag = slide.find('img', src=True)  # 找到 <img> 标签，带有 src 属性
        caption_tag = slide.find('div', class_='text_box')  # 查找标题所在的 <div> 标签

        if link_tag and img_tag and caption_tag:
            # 提取信息
            title = caption_tag.find('h4').get_text().strip() if caption_tag.find('h4') else ''
            link = link_tag['href']
            picture = img_tag['src']
            author = slide.find('span', class_='author').get_text().strip() if slide.find('span', class_='author') else None
            time = slide.find('span', class_='time').get_text().strip() if slide.find('span', class_='time') else None

            # 将提取到的信息添加到结果列表
            result.append({
                'title': title,
                'link': link,
                'picture': picture,
                'author': author,
                'time': time
            })
    return result


url = ''
news_data = scrape_news(url)
print(json.dumps(news_data, indent=4, ensure_ascii=False))