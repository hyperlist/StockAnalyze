# -*- coding: utf-8 -*-
import scrapy
import re
from myspider.items import ArticlesItem,ShortNewsItem
import time

str = time.strftime("%Y", time.localtime()) + "年" + time.strftime("%m", time.localtime()) + "年" + time.strftime("%d", time.localtime()) + "日 "
                    
class DongfangSpider(scrapy.Spider):
    name = 'dongfang'
    allowed_domains = ['http://www.eastmoney.com/']
    def start_requests(self):
        start_url = 'http://kuaixun.eastmoney.com/'
        headers = {
            'Connection': 'keep - alive',
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/52.0.2743.82 Safari/537.36'
        }
        yield scrapy.Request(url=start_url)

    def parse(self, response):
        #print(response)
        list = response.xpath('//div[@id="livenews-list"]/div[@class="livenews-media"]')
        for news in list:
            link = news.xpath('div/h2/a/@href').extract_first()
            if link:
                #print(link)
                yield scrapy.Request(url=link, callback=self.parse_article,dont_filter=True)
            else:
                time = news.xpath('div/span/text()').extract_first()
                review = news.xpath('div/h2/span/text()').extract_first()
                id = news.xpath('@id').extract_first().replace('livenews-id-2-','').replace('livenews-id-1-','')
                if time and review:
                    #print(time,review)
                    ShortNews = ShortNewsItem()
                    ShortNews['id'] = id
                    ShortNews['time'] = str + time
                    ShortNews['review'] = review
                    #print(ShortNews)
                    yield ShortNews
                
                
            
    def parse_article(self, response):
        url = response.url
        id = re.search('/a/(\d+).html', url).group(1)
        info = response.xpath('//div[@class="time-source"]')
        title = response.xpath('//div[@class="newsContent"]/h1/text()').extract_first()
        time = info.xpath('div[@class="time"]/text()').extract_first()
        source = info.xpath('div[@class="source data-source"]/@data-source').extract_first()
        #print(url,time,source,title)
        context = response.xpath('//div[@id="ContentBody"]')
        review = context.xpath('div[2]/text()').extract_first()
        data = context.xpath('//p')
        text = ''
        for p in data:
            if p.xpath('@class'):
                continue
            text += p.xpath('string(.)').extract_first().replace('\u3000','').replace('\r\n|\n|\b|\t',' ')
        Articles = ArticlesItem()
        Articles['id'] = id
        Articles['url'] = url
        Articles['title'] = title
        Articles['time'] = time
        Articles['source'] = source
        Articles['review'] = review
        Articles['full_text'] = text
        #print(Articles)
        return Articles
        