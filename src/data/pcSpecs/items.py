# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

from scrapy import Item, Field


class GPU(Item):
    rank = Field()
    url = Field()
    productname = Field()
    price = Field()
    rating = Field()
    brand = Field()
    chipmake = Field()
    

class AMAZON(Item):
    name = Field()
    

class CPU(Item):
    productname = Field()
    brand = Field()
    series = Field()
    name = Field()
    socket = Field()
    corename = Field()
    core = Field()
    freq = Field()
    l3cache = Field()
    l2cache = Field()
    power = Field()
    rating = Field()
    price = Field()
    url = Field()
    rank = Field()