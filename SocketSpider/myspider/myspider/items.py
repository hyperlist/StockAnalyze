# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://doc.scrapy.org/en/latest/topics/items.html

import scrapy



class ArticlesItem(scrapy.Item):
    id = scrapy.Field()
    url = scrapy.Field()
    title = scrapy.Field()
    time = scrapy.Field()
    source = scrapy.Field()
    review = scrapy.Field()
    full_text = scrapy.Field()
    
class ShortNewsItem(scrapy.Item):
    id = scrapy.Field()
    time = scrapy.Field()
    review = scrapy.Field()