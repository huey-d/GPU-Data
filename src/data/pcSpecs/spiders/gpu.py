import scrapy
from scrapy.http import Request
from pcSpecs.items import GPU

class gpuSpider(scrapy.Spider):
    name = "gpu"
    page_number = 2
    allowed_domains = ["newegg.com"]
    rank = 1

    def start_requests(self):
        start_urls = [
            "https://www.newegg.com/Desktop-Graphics-Cards/SubCategory/ID-48?Tid=7709&Order=3&PageSize=96"
        ]
        for url in start_urls:
            yield Request(url = url, callback = self.parse)

    def parse(self, response):

        self.logger.info('Scraping GPU Data...')

        products = response.xpath('//div[@class = "item-container"]')
        
        
        for product in products:
            
            item = GPU()

            rank = self.rank
            url = product.xpath('div[@class = "item-info"]/a/@href').extract_first()
            productname = product.xpath('div[@class = "item-info"]/a/text()').extract_first()
            price = product.xpath('div[@class = "item-action"]/ul/li[@class = "price-current"]/strong/text()').extract_first()
            rating = product.xpath('div[@class = "item-info"]/div[@class = "item-branding"]/a/@title').extract_first()
            if rating == []:
                item['rating'] = 'no_rating'
            else:
                item['rating'] = rating
            item['rank'] = rank
            
            item['url'] = url
            item['productname'] = productname
            item['price'] = price
            
            product_URL = product.xpath('div[@class = "item-info"]/a/@href').get()
            request = Request(product_URL, callback = self.productpage)
            request.meta['item'] = item
            self.rank += 1

            yield request
            
        next_page = 'https://www.newegg.com/Desktop-Graphics-Cards/SubCategory/ID-48/Page-' + str(gpuSpider.page_number) + '?Tid=7709&Order=3&PageSize=96'
        if gpuSpider.page_number <= 74:
            gpuSpider.page_number += 1
            yield response.follow(next_page, callback = self.parse)
            
    def productpage(self, response):
        item = response.meta['item']
        
        Brand = ['ASUS', 'MSI', 'EVGA', 'GIGABYTE', 'NVIDIA', 'Sapphire Tech', 'ASRock', 'PNY Technologies, Inc.', 'ZOTAC', 'XFX', 'VisionTek', 'DELL', 'HP', 'MAXSUN', 'PowerColor']
        manufacturers = ['AMD', 'Intel', 'NVIDIA']
                

        specs = response.xpath('//div[@class="tab-pane"]')
        for info in specs:
            brand = info.xpath('//*[@id="product-details"]/div[2]/div[2]/table[2]/tbody/tr[1]/td/text()').extract_first()
            # model = info.xpath('//*[@id="product-details"]/div[2]/div[2]/table[2]/tbody/tr[3]/td/text()').extract_first()
            chipmake = info.xpath('//*[@id="product-details"]/div[2]/div[2]/table[4]/tbody/tr[1]/td/text()').extract_first()
            # date = info.xpath('//*[@id="product-details"]/div[2]/div[2]/table[12]/tbody/tr/td/text()').extract_first()


        if brand not in Brand:
            item['brand'] = None
        else:
            item['brand'] = brand
        
        # item['model'] = model

        if chipmake not in manufacturers:
            item['chipmake'] = None
        else:
            item['chipmake'] = chipmake
            
        # item['date'] = date

        yield item